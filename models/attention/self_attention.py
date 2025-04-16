import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from functools import partial
from einops import rearrange
from .rotary import RotaryEmbedding


Linear = partial(nn.Linear, bias=False)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, rotary: bool = True, causal: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_head = self.hidden_size // self.n_heads
        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(hidden_size), Linear(hidden_size, hidden_size * 3)
        )
        self.out_proj = Linear((hidden_size // n_heads) * n_heads, hidden_size)
        self.q_ln = nn.LayerNorm(hidden_size, bias=False)
        self.k_ln = nn.LayerNorm(hidden_size, bias=False)
        self.reshaper = partial(rearrange, pattern="b s (h d) -> b h s d", h=n_heads)
        self.rotary = RotaryEmbedding(hidden_size // n_heads) if rotary else None
        self.causal = causal

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # attention mask already prepped for sdpa shape (bs, 1, seq_len, seq_len)
        b, L, _ = x.shape
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :].expand(b, 1, L, L).bool()
        qkv = self.layernorm_qkv(x) # (bs, seq_len, hidden_size * 3)
        q, k, v = torch.chunk(qkv, 3, dim=-1) # (bs, seq_len, hidden_size)
        q, k = self.q_ln(q).to(q.dtype), self.k_ln(k).to(q.dtype)
        if self.rotary:
            q, k = self._apply_rotary(q, k)
        q, k, v = map(self.reshaper, (q, k, v)) # (bs, n_heads, seq_len, d_head)
        a = F.scaled_dot_product_attention(q, k, v, attention_mask if not self.causal else None, is_causal=self.causal) # (bs, n_heads, seq_len, d_head)
        a = rearrange(a, "b h s d -> b s (h d)") # (bs, seq_len, n_heads * d_head)
        return self.out_proj(a) # (bs, seq_len, hidden_size)


if __name__ == '__main__':
    ### py -m models.attention.multihead_attention
    # Test MultiHeadAttention
    batch_size, seq_len, hidden_size, n_heads = 2, 10, 256, 8
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}, hidden_size: {hidden_size}, n_heads: {n_heads}")
    mha = MultiHeadAttention(hidden_size, n_heads)
    x = torch.randn(batch_size, seq_len, hidden_size)
    out = mha(x)
    print(f"MultiHeadAttention output shape: {out.shape}")