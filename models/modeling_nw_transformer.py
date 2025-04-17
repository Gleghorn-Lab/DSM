import torch
import torch.nn as nn
from typing import Optional
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from .attention.cross_attention import CrossAttention
from .MLP import swiglu_ffn
from .modeling_transformer import TransformerBlock
from .utils import pad_and_concatenate_dimer
from .pooler import Pooler


class NWTransformerConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size: int = 33,
        hidden_size: int = 64,
        n_heads: int = 2,
        n_layers: int = 1,
        head_dim: int = 256,
        max_length: int = 4096,
        expansion_ratio: float = 8/3,
        dropout: float = 0.1,
        loss_type: str = "l1",
        pooling_type: str = "max",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.head_dim = head_dim
        self.max_length = max_length
        self.loss_type = loss_type
        self.expansion_ratio = expansion_ratio
        self.dropout = dropout
        self.pooling_type = pooling_type


class NWTransformerFull(PreTrainedModel):
    config_class = NWTransformerConfig
    def __init__(self, config: NWTransformerConfig):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.transformer = TransformerBlock(
            hidden_size=config.hidden_size,
            n_heads=config.n_heads,
            expansion_ratio=config.expansion_ratio,
            dropout=config.dropout,
            rotary=True,
            causal=False
        )
        self.pooler = Pooler(config.pooling_type)
        self.regression_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.head_dim),
            nn.ReLU(),
            nn.Linear(config.head_dim, 1)
        )

        if config.loss_type == "l1":
            self.loss_fct = nn.L1Loss()
        elif config.loss_type == "l2":
            self.loss_fct = nn.MSELoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> SequenceClassifierOutput:
        x = self.embedding(input_ids)
        x = self.transformer(x, attention_mask)
        x = self.pooler(x, attention_mask)
        
        logits = self.regression_head(x)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )


class NWTransformerCross(PreTrainedModel):
    config_class = NWTransformerConfig
    def __init__(self, config: NWTransformerConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.cross_ab = CrossAttention(config.hidden_size, config.n_heads)
        self.cross_ba = CrossAttention(config.hidden_size, config.n_heads)
        self.ffn = swiglu_ffn(config.hidden_size, config.expansion_ratio, config.dropout)
        self.pooler = Pooler(config.pooling_type)
        self.regression_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.head_dim),
            nn.ReLU(),
            nn.Linear(config.head_dim, 1)
        )
        if config.loss_type == "l1":
            self.loss_fct = nn.L1Loss()
        elif config.loss_type == "l2":
            self.loss_fct = nn.MSELoss()

    def _low_temp_softmax(self, logits, eps=1e-8):
        return (logits / eps).softmax(dim=-1)

    def _get_logits(
            self,
            x_a: torch.Tensor,
            x_b: torch.Tensor,
            attention_mask_a: torch.Tensor,
            attention_mask_b: torch.Tensor
        ) -> torch.Tensor:
        x_a = self.cross_ab(x_a, x_b, attention_mask_a, attention_mask_b)
        x_b = self.cross_ba(x_b, x_a, attention_mask_b, attention_mask_a)
        x, attention_mask = pad_and_concatenate_dimer(x_a, x_b, attention_mask_a, attention_mask_b)
        x = self.ffn(x)
        x = self.pooler(x, attention_mask)
        
        logits = self.regression_head(x)
        return logits

    def scoring(
            self,
            input_ids_a: torch.Tensor,
            logits_b: torch.Tensor,
            attention_mask_a: Optional[torch.Tensor] = None,
            attention_mask_b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_a = self.embedding(input_ids_a)
        logits_b = logits_b[:, :, :self.vocab_size]
        soft_argmax_b = self._low_temp_softmax(logits_b) # (b, l, v)
        x_b = soft_argmax_b @ self.embedding.weight # (b, l, d)
        logits = self._get_logits(x_a, x_b, attention_mask_a, attention_mask_b)
        return logits

    def forward(
        self,
        input_ids_a: torch.Tensor,
        input_ids_b: torch.Tensor,
        attention_mask_a: Optional[torch.Tensor] = None,
        attention_mask_b: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> SequenceClassifierOutput:
        x_a, x_b = self.embedding(input_ids_a), self.embedding(input_ids_b)
        logits = self._get_logits(x_a, x_b, attention_mask_a, attention_mask_b)
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )
