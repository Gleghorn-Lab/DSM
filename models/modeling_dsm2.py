import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from transformers.modeling_outputs import ModelOutput

from dsm2.e1 import E1Config, E1ForMaskedLM
from dsm2.losses import contrastive_loss, jepa_loss
from .generate_mixin import GenerateMixin


class DSM2Config(E1Config):
    model_type = "dsm2"
    def __init__(
        self,
        teacher_hidden_size: int = 768,
        use_teacher_projections: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.teacher_hidden_size = teacher_hidden_size
        self.use_teacher_projections = use_teacher_projections


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
    s_max: Optional[Tuple[List[torch.Tensor]]] = None


class DSM2(E1ForMaskedLM, GenerateMixin):
    config_class = DSM2Config
    def __init__(self, config: DSM2Config, **kwargs):
        E1ForMaskedLM.__init__(self, config, **kwargs)
        GenerateMixin.__init__(self)
        self.config = config
        self.vocab_size = config.vocab_size
                
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        self.mask_token_id = self.tokenizer.mask_token_id

        # Projection layers align student hidden states to teacher hidden size when distillation dims differ.
        if config.use_teacher_projections:
            self.teacher_projections = nn.ModuleList([
                nn.Linear(config.hidden_size, config.teacher_hidden_size)
                for _ in range(config.num_hidden_layers)
            ])
        else:
            self.teacher_projections = None
        
        self.special_token_ids = self.get_special_token_ids()

    def get_special_token_ids(self, extra_tokens: Optional[List[str]] = None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mask_token = self.tokenizer.mask_token
        collected_tokens: list[str] = []
        for token_value in self.tokenizer.special_tokens_map.values():
            if isinstance(token_value, str):
                if token_value != mask_token:
                    collected_tokens.append(token_value)
            elif isinstance(token_value, list):
                for token in token_value:
                    if token != mask_token:
                        collected_tokens.append(token)
        self.special_token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in collected_tokens]
        if extra_tokens is not None:
            self.special_token_ids.extend([self.tokenizer.convert_tokens_to_ids(token) for token in extra_tokens])
        self.special_token_ids = list(set(self.special_token_ids))
        self.special_token_ids = torch.tensor(self.special_token_ids, device=device).flatten()
        return self.special_token_ids

    @torch.no_grad()
    def _get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        model_kwargs: dict[str, Any] = {}
        if token_type_ids is not None:
            model_kwargs["token_type_ids"] = token_type_ids
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            output_attentions=False,
            **model_kwargs,
        )
        assert outputs.logits is not None, "Student model must return logits."
        return outputs.logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
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
        
        special_mask = torch.isin(input_ids, self.special_token_ids.to(device))
        mask_indices = mask_indices & ~special_mask & attention_mask.bool()

        noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)
        labels = input_ids.clone()
        non_mask_indices = ~mask_indices | (attention_mask == 0)
        labels[non_mask_indices] = -100

        output_s_max = self.training
        if "output_s_max" in kwargs:
            output_s_max = kwargs["output_s_max"]

        model_kwargs: dict[str, Any] = {}
        if token_type_ids is not None:
            model_kwargs["token_type_ids"] = token_type_ids
        if "within_seq_position_ids" in kwargs:
            model_kwargs["within_seq_position_ids"] = kwargs["within_seq_position_ids"]
        if "global_position_ids" in kwargs:
            model_kwargs["global_position_ids"] = kwargs["global_position_ids"]
        if "sequence_ids" in kwargs:
            model_kwargs["sequence_ids"] = kwargs["sequence_ids"]

        outputs = super().forward(
            input_ids=noisy_batch,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=False,
            output_s_max=output_s_max,
            **model_kwargs,
        )

        all_hidden_states = outputs.hidden_states
        assert all_hidden_states is not None, "DSM2 requires hidden_states from the student model."
        expected_hidden_state_count = self.config.num_hidden_layers + 1
        assert len(all_hidden_states) == expected_hidden_state_count, (
            f"Expected {expected_hidden_state_count} hidden-state tensors (embedding + {self.config.num_hidden_layers} layers), "
            f"got {len(all_hidden_states)}."
        )
        student_states_for_distill = tuple(all_hidden_states[1:expected_hidden_state_count])
        assert len(student_states_for_distill) == self.config.num_hidden_layers, (
            f"Expected {self.config.num_hidden_layers} student distillation layers, got {len(student_states_for_distill)}."
        )

        if self.teacher_projections is None:
            projected_student_states = student_states_for_distill
        else:
            projected_student_states = []
            for state, proj in zip(student_states_for_distill, self.teacher_projections):
                projected_student_states.append(proj(state))
            projected_student_states = tuple(projected_student_states)
        
        last_hidden_state = outputs.last_hidden_state
        lm_logits = self.mlm_head(last_hidden_state)

        joint_mask = mask_indices & attention_mask.bool()
        if not joint_mask.any():
            joint_mask = attention_mask.bool()
        distill_mask = (~mask_indices) & attention_mask.bool() & ~special_mask
        assert distill_mask.any(), "distill_mask must include at least one non-special, non-padding token."

        token_loss = self.ce_loss(
            lm_logits[joint_mask].view(-1, self.vocab_size),
            input_ids[joint_mask].view(-1)
        ) / p_mask[joint_mask]

        ce_loss_val = token_loss.sum() / (batch_size * seq_len)
        total_loss = alpha_ce * ce_loss_val

        jepa_loss_val = None
        contrastive_loss_val = None

        if teacher_hidden_states is not None:
            teacher_hidden_states = tuple(teacher_hidden_states)
            assert len(projected_student_states) == len(teacher_hidden_states), (
                f"Student hidden states ({len(projected_student_states)}) and teacher hidden states ({len(teacher_hidden_states)}) "
                "must have the same number of layers."
            )
            for layer_idx in range(len(projected_student_states)):
                student_shape = projected_student_states[layer_idx].shape
                teacher_shape = teacher_hidden_states[layer_idx].shape
                assert student_shape == teacher_shape, (
                    f"Layer {layer_idx} hidden-state shape mismatch: student {student_shape} vs teacher {teacher_shape}."
                )
            if self.teacher_projections is None and ((alpha_jepa > 0.0) or (alpha_contrastive > 0.0)):
                assert projected_student_states[0].shape[-1] == teacher_hidden_states[0].shape[-1], (
                    f"Teacher hidden size ({teacher_hidden_states[0].shape[-1]}) does not match student hidden size "
                    f"({projected_student_states[0].shape[-1]}). Set use_teacher_projections=True, or disable distillation losses."
                )
            if alpha_jepa > 0.0:
                jepa_loss_val = jepa_loss(
                    student_hidden_states=projected_student_states,
                    teacher_hidden_states=teacher_hidden_states,
                    distill_mask=distill_mask,
                )
                total_loss = total_loss + (alpha_jepa * jepa_loss_val)
            
            if alpha_contrastive > 0.0:
                contrastive_loss_val = contrastive_loss(
                    student_hidden_states=projected_student_states,
                    teacher_hidden_states=teacher_hidden_states,
                    attention_mask=attention_mask,
                )
                total_loss = total_loss + (alpha_contrastive * contrastive_loss_val)

        return DSM2Output(
            loss=total_loss,
            ce_loss=ce_loss_val,
            contrastive_loss=contrastive_loss_val,
            jepa_loss=jepa_loss_val,
            logits=lm_logits,
            mask_labels=labels,
            last_hidden_state=last_hidden_state,
            student_hidden_states=projected_student_states,
            t=t,
            s_max=outputs.s_max,
        )
