import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Any, Union, List
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass

from .FastPLMs.esm_plusplus.modeling_esm_plusplus import ESMplusplusModel, ESMplusplusConfig, UnifiedTransformerBlock
from .generate_mixin import GenerateMixin
from .FastPLMs.embedding_mixin import Pooler


class DSM2Config(ESMplusplusConfig):
    model_type = "dsm2"
    def __init__(
        self,
        teacher_hidden_size: int = 768,
        expansion_ratio: float = 8 / 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.teacher_hidden_size = teacher_hidden_size
        self.expansion_ratio = expansion_ratio


@dataclass
class DSM2Output(ModelOutput):
    logits: Optional[torch.Tensor] = None
    mask_labels: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    ce_loss: Optional[torch.Tensor] = None
    contrastive_loss: Optional[torch.Tensor] = None
    jepa_loss: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None
    student_hidden_states: Optional[Tuple[torch.Tensor]] = None
    t: Optional[torch.Tensor] = None



class LMHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, soft_logit_cap: float = 30.0):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.soft_logit_cap = soft_logit_cap
        self.act = nn.GELU()
    
    @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)
        x = self.act(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return self.soft_logit_cap * torch.tanh(x / self.soft_logit_cap)


def pool_states(hidden_states: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """
    Pools a tuple of hidden states natively using mean and var pooling.
    Returns stacked pooled states of shape (num_layers, b, 2d).
    """
    pooler = Pooler(pooling_types=["mean", "var"])
    stacked = torch.stack(hidden_states)
    pooled = []
    for layer in stacked:
        pooled.append(pooler(layer)) # (b, 2d)
    return torch.stack(pooled) # (num_layers, b, 2d)


def contrastive_loss_from_pooled(
    s_pooled: torch.Tensor,
    t_pooled: torch.Tensor,
) -> torch.Tensor:
    """
    Computes depth-weighted contrastive loss from pre-pooled student and teacher representations.
    s_pooled, t_pooled: (num_layers, b, 2d)
    """
    num_layers = s_pooled.shape[0]

    # (num_layers, b, b)
    intra_student_reps = torch.bmm(s_pooled, s_pooled.transpose(1, 2)).softmax(dim=-1)
    intra_teacher_reps = torch.bmm(t_pooled, t_pooled.transpose(1, 2)).softmax(dim=-1)

    # (num_layers, b, b)
    squared_diff = (intra_student_reps - intra_teacher_reps) ** 2

    # Weight by depth
    depth_weights = torch.arange(1, num_layers + 1, device=squared_diff.device, dtype=squared_diff.dtype) / num_layers
    # (num_layers, 1, 1)
    depth_weights = depth_weights.view(num_layers, 1, 1)

    weighted_squared_diff = squared_diff * depth_weights

    # Average over layers and batch pairs
    layer_batch_loss = weighted_squared_diff.mean(dim=0).mean() # scalar

    return layer_batch_loss


def contrastive_loss(
    student_hidden_states: Tuple[torch.Tensor, ...],
    teacher_hidden_states: Tuple[torch.Tensor, ...],
    p_masks: torch.Tensor,
) -> torch.Tensor:
    """
    Computes a depth-weighted contrastive loss mapping student representations
    to teacher representations, scaled by the inverse of the mask rate.
    """
    assert len(student_hidden_states) == len(teacher_hidden_states), "Student and teacher hidden states must have the same number of layers"
    s_pooled = pool_states(student_hidden_states)
    t_pooled = pool_states(teacher_hidden_states)
    return contrastive_loss_from_pooled(s_pooled, t_pooled, p_masks)


def jepa_loss(
    student_hidden_states: Tuple[torch.Tensor, ...],
    teacher_hidden_states: Tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    p_masks: torch.Tensor,
) -> torch.Tensor:
    """
    Computes depth-weighted MSE between student and teacher hidden states for unmasked tokens,
    scaled by inverse mask rate.
    """
    assert len(student_hidden_states) == len(teacher_hidden_states), "Student and teacher hidden states must have the same number of layers"
    num_layers = len(student_hidden_states)
    
    mask = attention_mask.bool() # (b, seq_len)

    # Stack to (num_layers, b, seq_len, d)
    s_stacked = torch.stack(student_hidden_states)
    t_stacked = torch.stack(teacher_hidden_states)

    # MSE per token, (num_layers, b, seq_len, d)
    squared_diff = (s_stacked - t_stacked) ** 2
    
    # Mean over hidden dimension, (num_layers, b, seq_len)
    mse_per_token = squared_diff.mean(dim=-1)

    # Weight by layer depth, (num_layers, 1, 1)
    depth_weights = torch.arange(1, num_layers + 1, device=squared_diff.device, dtype=squared_diff.dtype) / num_layers
    depth_weights = depth_weights.view(num_layers, 1, 1)

    # Scale the Loss by 1 / p_masks
    # p_masks is (b, seq_len)
    # We want 1 / p_masks (where we enforce an eps threshold inside DSM2)
    inv_p_masks = (1.0 / p_masks).unsqueeze(0) # (1, b, seq_len)

    weighted_mse = mse_per_token * depth_weights * inv_p_masks

    # Filter out padding tokens and sum up
    valid_mse = weighted_mse[:, mask]

    # Average over valid tokens and layers
    return valid_mse.mean()


class DSM2(ESMplusplusModel, GenerateMixin):
    config_class = DSM2Config
    def __init__(self, config: DSM2Config, **kwargs):
        ESMplusplusModel.__init__(self, config, **kwargs)
        GenerateMixin.__init__(self)
        self.config = config
        self.vocab_size = config.vocab_size
        
        self.lm_head = LMHead(config.hidden_size, config.vocab_size)
        
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        self.mask_token_id = self.tokenizer.mask_token_id

        # Replace transformer blocks with custom expansion size since ESM++ natively hardcodes 8 / 3
        # and doesn't pass expansion_ratio down to UnifiedTransformerBlock through TransformerStack.
        self.transformer.blocks = nn.ModuleList([
            UnifiedTransformerBlock(
                config.hidden_size,
                config.num_attention_heads,
                residue_scaling_factor=math.sqrt(config.num_hidden_layers / 36),
                expansion_ratio=config.expansion_ratio,
                dropout=config.dropout,
                attn_backend=config.attn_backend,
            )
            for _ in range(config.num_hidden_layers)
        ])
        # Re-initialize the weights for these dynamically created blocks
        self.apply(self._init_weights)

        # Projection layers to align student hidden states to teacher hidden size
        self.teacher_projections = nn.ModuleList([
            nn.Linear(config.hidden_size, config.teacher_hidden_size)
            for _ in range(config.num_hidden_layers)
        ])
        
        self.special_token_ids = self.get_special_token_ids()

    def get_special_token_ids(self, extra_tokens: Optional[List[str]] = None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mask_token = self.tokenizer.mask_token
        self.special_token_ids = [self.tokenizer.convert_tokens_to_ids(v) for k, v in self.tokenizer.special_tokens_map.items() if v != mask_token]
        if extra_tokens is not None:
            self.special_token_ids.extend([self.tokenizer.convert_tokens_to_ids(v) for v in extra_tokens])
        self.special_token_ids = list(set(self.special_token_ids))
        self.special_token_ids = torch.tensor(self.special_token_ids, device=device).flatten()
        return self.special_token_ids

    @torch.no_grad()
    def _get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Any
    ) -> torch.Tensor:
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            output_attentions=False,
        )
        x = outputs.last_hidden_state
        return self.lm_head(x)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        teacher_hidden_states: Optional[Tuple[torch.Tensor, ...]] = None,
        alpha_ce: float = 1.0,
        alpha_jepa: float = 1.0,
        alpha_contrastive: float = 1.0,
        **kwargs: Any
    ) -> DSM2Output:
        
        eps = 1e-3
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)

        if self.training:
            t = torch.rand(batch_size, device=device)
            t = (1 - eps) * t + eps
        else:
            t = torch.full((batch_size,), 0.15, device=device)

        p_mask = t[:, None].repeat(1, seq_len)
        mask_indices = torch.rand(batch_size, seq_len, device=device) < p_mask
        
        # ensure special_token_ids is on the same device
        if getattr(self, 'special_token_ids', None) is not None:
            special_mask = torch.isin(input_ids, self.special_token_ids.to(device))
            mask_indices = mask_indices & ~special_mask & attention_mask.bool()
        else:
            mask_indices = mask_indices & attention_mask.bool()

        noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)
        labels = input_ids.clone()
        non_mask_indices = ~mask_indices | (attention_mask == 0)
        labels[non_mask_indices] = -100

        outputs = super().forward(
            input_ids=noisy_batch,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=False,
        )

        all_hidden_states = outputs.hidden_states
        # Outputs from TransformerStack (in modeling_esm_plusplus.py) includes `hidden_states` as a tuple.
        # But ESMplusplusModel forward returns `TransformerOutput` passing along `hidden_states`.
        # Note: the first element in hidden_states is usually the embeddings. We iterate from index 1.
        if len(all_hidden_states) == self.config.num_hidden_layers + 1:
            all_hidden_states = all_hidden_states[1:]
            
        projected_student_states = []
        for state, proj in zip(all_hidden_states, self.teacher_projections):
            projected_student_states.append(proj(state))
        
        projected_student_states = tuple(projected_student_states)
        
        x = outputs.last_hidden_state
        lm_logits = self.lm_head(x)

        joint_mask = mask_indices & attention_mask.bool()
        if not joint_mask.any():
            joint_mask = attention_mask.bool()

        token_loss = self.ce_loss(
            lm_logits[joint_mask].view(-1, self.vocab_size),
            input_ids[joint_mask].view(-1)
        ) / p_mask[joint_mask]

        ce_loss_val = token_loss.sum() / (batch_size * seq_len)
        total_loss = alpha_ce * ce_loss_val

        jepa_loss_val = None
        contrastive_loss_val = None

        if teacher_hidden_states is not None:
            if alpha_jepa > 0.0:
                jepa_loss_val = jepa_loss(
                    student_hidden_states=projected_student_states,
                    teacher_hidden_states=teacher_hidden_states,
                    attention_mask=attention_mask,
                    p_masks=p_mask,
                )
                total_loss = total_loss + (alpha_jepa * jepa_loss_val)
            
            if alpha_contrastive > 0.0:
                contrastive_loss_val = contrastive_loss(
                    student_hidden_states=projected_student_states,
                    teacher_hidden_states=teacher_hidden_states,
                )
                total_loss = total_loss + (alpha_contrastive * contrastive_loss_val)

        return DSM2Output(
            loss=total_loss,
            ce_loss=ce_loss_val,
            contrastive_loss=contrastive_loss_val,
            jepa_loss=jepa_loss_val,
            logits=lm_logits,
            mask_labels=labels,
            last_hidden_state=x,
            student_hidden_states=projected_student_states,
            t=t,
        )

if __name__ == "__main__":
    # Test vectorization locally
    from FastPLMs.embedding_mixin import Pooler
    
    b = 2
    l = 16
    d = 32
    num_layers = 4
    
    # student & teacher mock
    s_states = tuple(torch.randn(b, l, d) for _ in range(num_layers))
    t_states = tuple(torch.randn(b, l, d) for _ in range(num_layers))
    
    attn_mask = torch.ones(b, l)
    p_mask = torch.full((b, l), 0.15)
    
    # naive jepa loop
    def old_jepa(s_states, t_states, attn_mask):
        mask = attn_mask.bool()
        loss = 0
        for depth, (s, t) in enumerate(zip(s_states, t_states)):
            s_masked = s[mask]
            t_masked = t[mask]
            layer_loss = F.mse_loss(s_masked, t_masked)
            layer_loss *= (depth + 1) / len(s_states)
            loss += layer_loss
        return loss
    
    # Run
    old_j_res = old_jepa(s_states, t_states, attn_mask)
    new_j_res = jepa_loss(s_states, t_states, attn_mask, p_mask)
    print("Old JEPA form (unscaled by 1/p):", old_j_res.item())
    # New JEPA scaled by ~ 6.66 on average
    print("New Vectorized JEPA (scaled by 1/p):", new_j_res.item())
    
    def old_contrastive(s_states, t_states):
        pooler = Pooler(pooling_types=["mean", "var"])
        loss = 0
        for depth, (s, t) in enumerate(zip(s_states, t_states)):
            sp = pooler(s)
            tp = pooler(t)
            intra_s = sp.matmul(sp.T)
            intra_t = tp.matmul(tp.T)
            layer_loss = F.mse_loss(intra_s, intra_t)
            layer_loss *= (depth + 1) / len(s_states)
            loss += layer_loss
        return loss
    
    old_c_res = old_contrastive(s_states, t_states)
    new_c_res = contrastive_loss(s_states, t_states)
    print("Old Contrastive form (unscaled):", old_c_res.item())
    print("New Vectorized Contrastive form (scaled weight):", new_c_res.item())