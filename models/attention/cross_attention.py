import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from functools import partial
from einops import rearrange
from .rotary import RotaryEmbeddingCross


Linear = partial(nn.Linear, bias=False)


class CrossAttention(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, rotary: bool = True, causal: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_head = self.hidden_size // self.n_heads
        self.Wq = Linear(hidden_size, hidden_size)
        self.Wk = Linear(hidden_size, hidden_size)
        self.Wv = Linear(hidden_size, hidden_size)
        self.out_proj = Linear((hidden_size // n_heads) * n_heads, hidden_size)
        self.q_ln = nn.LayerNorm(hidden_size, bias=False)
        self.k_ln = nn.LayerNorm(hidden_size, bias=False)
        self.reshaper = partial(rearrange, pattern="b s (h d) -> b h s d", h=n_heads)
        self.rotary = RotaryEmbeddingCross(hidden_size // n_heads) if rotary else None
        self.causal = causal

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
            attention_mask_1: Optional[torch.Tensor] = None,
            attention_mask_2: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        # For cross attention, we need a mask that goes from L1 (query) to L2 (key)
        attention_mask = None
        if attention_mask_1 is not None and attention_mask_2 is not None:
            # Create cross attention mask: queries from x1 attending to keys from x2
            # Shape: (bs, 1, L1, L2)
            attention_mask = torch.einsum('bi,bj->bij', attention_mask_1, attention_mask_2)
            attention_mask = attention_mask.unsqueeze(1).bool()
        q = self.Wq(x1) # (bs, L1, hidden_size)
        k = self.Wk(x2) # (bs, L2, hidden_size)
        v = self.Wv(x2) # (bs, L2, hidden_size)
        q, k = self.q_ln(q).to(q.dtype), self.k_ln(k).to(q.dtype)
        if self.rotary:
            q, k = self._apply_rotary(q, k)
        q, k, v = map(self.reshaper, (q, k, v)) # (bs, n_heads, L1, d_head) (bs, n_heads, L2, d_head) (bs, n_heads, L2, d_head)
        a = F.scaled_dot_product_attention(q, k, v, attention_mask if not self.causal else None, is_causal=self.causal) # (bs, n_heads, L1, d_head)
        a = rearrange(a, "b h s d -> b s (h d)") # (bs, L1, n_heads * d_head)
        return self.out_proj(a) # (bs, L1, hidden_size)


if __name__ == '__main__':
    ### py -m models.attention.cross_attention
    # Test CrossAttention
    batch_size, seq_len, hidden_size, n_heads = 2, 10, 256, 8
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}, hidden_size: {hidden_size}, n_heads: {n_heads}")
    mha = CrossAttention(hidden_size, n_heads)
    x1 = torch.randn(batch_size, seq_len, hidden_size)
    x2 = torch.randn(batch_size, seq_len, hidden_size)
    out = mha(x1, x2)
    print(f"CrossAttention output shape: {out.shape}")