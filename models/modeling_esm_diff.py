import torch
import torch.nn as nn
from typing import Optional, Tuple, Any, Union
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from .FastPLMs.modeling_fastesm import FastEsmModel, FastEsmConfig
from .generate_mixin import GenerateMixin
from .modeling_transformer import Transformer
from .modeling_nw_transformer import NWTransformerCross
from .alignment_helpers import AlignmentLossLike


class ESMDiffConfig(FastEsmConfig):
    model_type = "esm_diff"
    def __init__(
        self,
        num_at_layers: int = 1,
        at_vocab_size: int = 30000,
        alignment_loss: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_at_layers = num_at_layers
        self.at_vocab_size = at_vocab_size
        self.alignment_loss = alignment_loss


@dataclass
class EsmDiffOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None
    last_hidden_state: Optional[torch.Tensor] = None
    t: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None


class LMHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, soft_logit_cap: float = 30.0):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.soft_logit_cap = soft_logit_cap
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)
        x = self.act(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return self.soft_logit_cap * torch.tanh(x / self.soft_logit_cap)


class ESM_Diff(FastEsmModel, GenerateMixin): # FastEsmModel already inherits EmbeddingMixin
    config_class = ESMDiffConfig
    def __init__(self, config: ESMDiffConfig, **kwargs):
        FastEsmModel.__init__(self, config, **kwargs)
        GenerateMixin.__init__(self, self.tokenizer)
        self.config = config
        self.vocab_size = config.vocab_size
        self.lm_head = LMHead(config.hidden_size, config.vocab_size)
        # tie to word embeddings
        self.lm_head.decoder.weight = self.esm.embeddings.word_embeddings.weight
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        self.mask_token_id = self.tokenizer.mask_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.alignment_loss = config.alignment_loss
        if self.alignment_loss:
            self.alignment_scorer = NWTransformerCross.from_pretrained('GleghornLab/AlignmentTransformer')
            for param in self.alignment_scorer.parameters():
                param.requires_grad = False
            self.l1_loss = nn.L1Loss()

    def _get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Any
    ) -> torch.Tensor:
        with torch.no_grad():
            x = self.esm.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state # (b, L, d)
            logits = self.lm_head(x) # (b, L, v)
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> EsmDiffOutput:
        eps = 1e-3
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if self.training: # sample uniform between 0 and 1
            t = torch.rand(batch_size, device=device)
            t = (1 - eps) * t + eps
        else: # evaluate at classic 15%
            t = torch.full((batch_size,), 0.15, device=device)
        
        p_mask = t[:, None].repeat(1, seq_len)
        mask_indices = torch.rand(batch_size, seq_len, device=device) < p_mask
        # prevent cls and eos from being masked
        cls_mask = input_ids == self.cls_token_id
        eos_mask = input_ids == self.eos_token_id
        mask_indices = mask_indices & ~cls_mask & ~eos_mask

        noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)
        labels = input_ids.clone()
        non_mask_indices = ~mask_indices | (attention_mask == 0)
        labels[non_mask_indices] = -100

        x = super().forward(
            input_ids=noisy_batch,
            attention_mask=attention_mask,
        ).last_hidden_state # (b, L, d)
        lm_logits = self.lm_head(x) # (b, L, v)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)

        joint_mask = mask_indices & attention_mask.bool()
        token_loss = self.ce_loss(
            lm_logits[joint_mask].view(-1, self.vocab_size),
            input_ids[joint_mask].view(-1)) / p_mask[joint_mask]

        loss = token_loss.sum() / (batch_size * seq_len)

        if self.alignment_loss:
            self.alignment_scorer.eval()
            pred_alignment = self.alignment_scorer.scoring(
                input_ids_a=input_ids,
                logits_b=lm_logits,
                attention_mask_a=attention_mask,
                attention_mask_b=attention_mask,
            )
            ideal_labels = torch.ones_like(pred_alignment, device=device)
            alignment_loss = self.l1_loss(pred_alignment.view(-1), ideal_labels.view(-1))
            ce_alpha = alignment_loss.clone().detach()
            loss = alignment_loss + ce_alpha * loss
            
            with torch.no_grad():
                pred_labels = self.alignment_scorer(
                input_ids_a=input_ids,
                input_ids_b=lm_logits.argmax(dim=-1),
                attention_mask_a=attention_mask,
                    attention_mask_b=attention_mask,
                ).logits
                _, scores = AlignmentLossLike()(lm_logits, input_ids)
                print('-' * 100)
                print(pred_labels.flatten()[:10])
                print(labels.flatten()[:10])
                print(scores.flatten()[:10])
                print('-' * 100)

            return EsmDiffOutput(
                loss=loss,
                logits=(lm_logits, labels, pred_alignment),
                last_hidden_state=x,
                hidden_states=None,
                attentions=None,
                t=t,
            )

        return EsmDiffOutput(
            loss=loss,
            logits=(lm_logits, labels),
            last_hidden_state=x,
            hidden_states=None,
            attentions=None,
            t=t,
        )


class ESM_Diff_AV(ESM_Diff):
    config_class = ESMDiffConfig
    def __init__(self, config: ESMDiffConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.at_embedding = nn.Embedding(config.at_vocab_size, config.hidden_size)
        self.at = Transformer(
            hidden_size=config.hidden_size,
            n_heads=config.num_attention_heads,
            expansion_ratio=2.0,
            dropout=0.1,
            rotary=True,
            causal=False,
            n_layers=config.num_at_layers
        )

    def _get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_tokens: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        **kwargs: Any
    ) -> torch.Tensor:
        x_at = self.at_embedding(prompt_tokens)
        x_at = self.at(x_at, prompt_attention_mask)
        _, seq_len = input_ids.shape
        x = self.esm.embeddings.word_embeddings(input_ids)
        x_hat = torch.cat([x_at, x], dim=1)
        attention_mask_hat = torch.cat([prompt_attention_mask, attention_mask], dim=1)
        x = self.esm.forward(
            inputs_embeds=x_hat,
            attention_mask=attention_mask_hat,
        ).last_hidden_state # (b, L, d)
        x = x[:, -seq_len:]
        logits = self.lm_head(x) # (b, L, v)
        return logits

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        at_ids: Optional[torch.Tensor] = None,
        at_attention_mask: Optional[torch.Tensor] = None,
    ) -> EsmDiffOutput:
        x_at = self.at_embedding(at_ids)
        x_at = self.at(x_at, at_attention_mask)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        eps = 1e-3
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        t = torch.rand(batch_size, device=device)
        t = (1 - eps) * t + eps
        p_mask = t[:, None].repeat(1, seq_len)
        mask_indices = torch.rand(batch_size, seq_len, device=device) < p_mask
        # prevent cls and eos from being masked
        cls_mask = input_ids == self.cls_token_id
        eos_mask = input_ids == self.eos_token_id
        mask_indices = mask_indices & ~cls_mask & ~eos_mask

        noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)
        labels = input_ids.clone()
        non_mask_indices = ~mask_indices | (attention_mask == 0)
        labels[non_mask_indices] = -100
        x = self.esm.embeddings.word_embeddings(noisy_batch)

        x_hat = torch.cat([x_at, x], dim=1)
        attention_mask_hat = torch.cat([at_attention_mask, attention_mask], dim=1)

        x = self.esm.forward(
            inputs_embeds=x_hat,
            attention_mask=attention_mask_hat,
        ).last_hidden_state # (b, L, d)
        x = x[:, -seq_len:]
        logits = self.lm_head(x) # (b, L, v)

        joint_mask = mask_indices & attention_mask.bool()
        token_loss = self.ce_loss(
            logits[joint_mask].view(-1, self.vocab_size),
            input_ids[joint_mask].view(-1)) / p_mask[joint_mask]

        loss = token_loss.sum() / (batch_size * seq_len)

        return EsmDiffOutput(
            loss=loss,
            logits=(logits, labels),
            last_hidden_state=x,
            hidden_states=None,
            attentions=None,
            t=t,
        )


if __name__ == "__main__":
    # py -m models.esm_diff.modeling_esm_diff
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ESM_Diff.from_pretrained('Synthyra/ESM2-8M', alignment_loss=True).to(device)
    print(model)

    # test forward
    input_ids = torch.randint(0, 33, (8, 256)).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    output = model(input_ids, attention_mask)
    lm_logits, module_pred, module_labels = output.logits
    print(output.loss, lm_logits.shape, module_pred.shape, module_labels.shape, output.t)
    
    # test generate
    length = 64
    steps = 16
    block_length = 16
