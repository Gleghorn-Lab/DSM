import torch
import torch.nn as nn
from typing import Optional
from attention.self_attention import MultiHeadAttention
from MLP import swiglu_ffn


class TransformerBlock(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            n_heads: int,
            expansion_ratio: float,
            dropout: float,
            rotary: bool,
            causal: bool,
            **kwargs
    ):
        super().__init__()
        self.attn = MultiHeadAttention(hidden_size, n_heads, rotary, causal)
        self.ffn = swiglu_ffn(hidden_size, expansion_ratio, dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attn(x, attention_mask) + x
        x = self.ffn(x) + x
        return x
    

class Transformer(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            n_heads: int,
            expansion_ratio: float,
            dropout: float,
            rotary: bool,
            causal: bool,
            n_layers: int,
            **kwargs
    ):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(
            hidden_size, n_heads, expansion_ratio, dropout, rotary, causal
        ) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, attention_mask)
        return x