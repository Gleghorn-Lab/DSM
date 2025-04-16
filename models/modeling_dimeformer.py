import torch
import torch.nn as nn
from typing import Optional, Tuple
from .attention.cross_attention import CrossAttention
from .MLP import swiglu_ffn
from .utils import pad_and_concatenate_dimer


class DimeformerBlock(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, expansion_ratio: float, dropout: float):
        super().__init__()
        self.cross_ab = CrossAttention(hidden_size, n_heads)
        self.cross_ba = CrossAttention(hidden_size, n_heads)
        self.ffn_a = swiglu_ffn(hidden_size, expansion_ratio, dropout)
        self.ffn_b = swiglu_ffn(hidden_size, expansion_ratio, dropout)

    def forward(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        attention_mask_a: Optional[torch.Tensor] = None,
        attention_mask_b: Optional[torch.Tensor] = None,
    ):
        res_a, res_b = x_a, x_b
        x_a = self.cross_ab(res_a, res_b, attention_mask_a, attention_mask_b) + res_a
        x_b = self.cross_ba(res_b, res_a, attention_mask_b, attention_mask_a) + res_b
        x_a = self.ffn_a(x_a) + x_a
        x_b = self.ffn_b(x_b) + x_b
        return x_a, x_b


class Dimeformer(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, expansion_ratio: float, dropout: float, n_layers: int):
        super().__init__()
        self.blocks = nn.ModuleList([DimeformerBlock(hidden_size, n_heads, expansion_ratio, dropout) for _ in range(n_layers)])

    def forward(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        attention_mask_a: Optional[torch.Tensor] = None,
        attention_mask_b: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # (b, L1, d), (b, L2, d) -> (b, L, d)
        for block in self.blocks:
            x_a, x_b = block(x_a, x_b, attention_mask_a, attention_mask_b)
        return pad_and_concatenate_dimer(x_a, x_b, attention_mask_a, attention_mask_b)
