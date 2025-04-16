import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from typing import Optional


Linear = partial(nn.Linear, bias=False)


class AttentionPooler(nn.Module):
    """
    Cross-attention mechanism for pooling (b, L, d) -> (b, n_tokens, d_pooled)
    """
    def __init__(
            self,
            hidden_size: int,
            n_tokens: int = 1,
            n_heads: int = 16,
    ):
        super(AttentionPooler, self).__init__()
        assert hidden_size % n_heads == 0, "hidden_size must be divisible by n_heads"
        self.n_tokens = n_tokens
        self.d_head = hidden_size // n_heads
        self.Q = nn.Parameter(torch.randn(1, n_tokens, hidden_size))
        self.Wq = Linear(hidden_size, hidden_size)
        self.Wv = Linear(hidden_size, hidden_size)
        self.Wk = Linear(hidden_size, hidden_size)
        self.Wo = Linear((hidden_size // n_heads) * n_heads, hidden_size)
        self.reshaper = partial(rearrange, pattern="b s (h d) -> b h s d", h=n_heads)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, L, d = x.size()
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :].expand(b, 1, self.n_tokens, L).bool()
        q = self.Wq(self.Q).expand(b, -1, -1)  # (b, n_tokens, d)
        v = self.Wv(x)  # (b, L, d)
        k = self.Wk(x)  # (b, L, d)
        q, k, v = map(self.reshaper, (q, k, v))  # (b, n_heads, n_tokens, d_head) (b, n_heads, L, d_head)
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, is_causal=False
        ) # (b, n_heads, n_tokens, d_head)
        attn = rearrange(attn, "b h s d -> b s (h d)")  # (b, n_tokens, n_heads * d_head)
        return self.Wo(attn)  # (b, n_tokens, d_pooled)


if __name__ == '__main__':
    ### py -m models.attention.attention_pooler
    # Test AttentionPooler
    batch_size, seq_len, hidden_size, n_tokens, n_heads = 2, 64, 256, 32, 8
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}, hidden_size: {hidden_size}, n_tokens: {n_tokens}, n_heads: {n_heads}")
    ap = AttentionPooler(hidden_size, n_tokens, n_heads)
    x = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)
    test_mask = torch.ones(batch_size, seq_len)
    test_mask[:, :-10] = 0
    a = ap(x, attention_mask)
    b = ap(x, None)
    c = ap(x, test_mask)
    print(a.isclose(b).all())
    print(b.isclose(c).all())

