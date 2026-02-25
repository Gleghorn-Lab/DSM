import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path
from typing import Optional, Tuple, Union, List
from einops import rearrange, repeat
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedModel, PreTrainedTokenizerFast, PretrainedConfig
from transformers.modeling_outputs import ModelOutput
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.attention.flex_attention import flex_attention

from .FastPLMs.embedding_mixin import EmbeddingMixin, Pooler


class PLMConfig(PretrainedConfig):
    """Configuration class for PLM model.
    
    Args:
        vocab_size: Size of the vocabulary
        hidden_size: Dimension of hidden layers
        num_attention_heads: Number of attention heads
        num_hidden_layers: Number of transformer layers
        num_labels: Number of output labels for classification
        problem_type: Type of problem - regression, single/multi label classification
    """
    model_type = "PLM"
    def __init__(
        self,
        vocab_size: int = 64,
        hidden_size: int = 960,
        expansion_ratio: float = 8 / 3,
        head_size: int = 64,
        num_hidden_layers: int = 30,
        num_labels: int = 2,
        problem_type: str | None = None,
        dropout: float = 0.0,
        initializer_range: float = 0.02,
        attn_backend: str = "flex",
        sliding_window_size: int = 512,
        dilation: int = 16,
        soft_logit_cap: float = 30.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.expansion_ratio = expansion_ratio
        self.head_size = head_size
        self.num_attention_heads = hidden_size // head_size
        self.num_hidden_layers = num_hidden_layers
        self.num_labels = num_labels
        self.problem_type = problem_type
        self.dropout = dropout
        self.initializer_range = initializer_range
        self.sliding_window_size = sliding_window_size
        self.dilation = dilation
        self.tie_word_embeddings = False
        self.attn_backend = attn_backend
        self.soft_logit_cap = soft_logit_cap


class LMHead(nn.Module):
    def __init__(self, config: PLMConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.soft_logit_cap = config.soft_logit_cap
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)
        x = self.act(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return self.soft_logit_cap * torch.tanh(x / self.soft_logit_cap)


### Rotary Embeddings
def rotate_half(x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(
            torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2
        )


def apply_rotary_emb_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
    _inplace: bool = False,
) -> torch.Tensor:
    """Apply rotary embeddings to input based on cos and sin."""
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    seqlen = x.size(1)
    cos = cos[:seqlen]
    sin = sin[:seqlen]
    cos = repeat(cos, "s d -> s 1 (2 d)")
    sin = repeat(sin, "s d -> s 1 (2 d)")
    return torch.cat(
        [
            x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )


class RotaryEmbedding(torch.nn.Module):
    """Rotary position embeddings.
    
    Based on the paper "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    
    Args:
        dim: Dimension of the embedding
        base: Base for computing angular frequencies
        interleaved: Whether to use interleaved rotations
        scale_base: Base for scaling
        scaling_factor: Factor for scaling positions
        pos_idx_in_fp32: Whether to compute position indices in fp32
        device: Computation device
    """
    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        interleaved: bool = False,
        scale_base: Optional[float] = None,
        scaling_factor: float = 1.0,
        pos_idx_in_fp32: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        self.interleaved = interleaved
        self.scale_base = scale_base
        self.scaling_factor = scaling_factor
        self.device = device

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None
        self._inv_freq_compute_device: Optional[torch.device] = None
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters of the embedding."""
        if "inv_freq" in self._buffers and isinstance(self._buffers["inv_freq"], torch.Tensor):
            buffer_device = self._buffers["inv_freq"].device
        else:
            buffer_device = self.device
        inv_freq = self._compute_inv_freq(buffer_device)
        self._inv_freq_compute_device = inv_freq.device
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        arange = torch.arange(0, self.dim, 2, device=buffer_device, dtype=torch.float32)
        scale = (
            (arange + 0.4 * self.dim) / (1.4 * self.dim)
            if self.scale_base is not None
            else None
        )
        self.register_buffer("scale", scale)

    def _compute_inv_freq(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Compute inverse frequency bands."""
        return 1 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )

    def _update_cos_sin_cache(self, seqlen: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """Update the cached cosine and sine values."""
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                t /= self.scaling_factor
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self.inv_freq.to(torch.float32)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                t /= self.scaling_factor
                inv_freq = self.inv_freq
            freqs = torch.outer(t, inv_freq)

            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(
                        seqlen, dtype=self.scale.dtype, device=self.scale.device
                    )
                    - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** power.unsqueeze(-1)
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to queries and keys.
        
        Args:
            q: Query tensor of shape (batch, seqlen, nheads, headdim)
            k: Key tensor of shape (batch, seqlen, nheads, headdim)
            
        Returns:
            Tuple of rotated query and key tensors
        """
        assert self._inv_freq_compute_device is not None, "Rotary inv_freq compute device should be set after initialization."
        if self._inv_freq_compute_device != q.device:
            self.reset_parameters()
        self._update_cos_sin_cache(q.shape[1], device=q.device, dtype=q.dtype)
        assert self._cos_cached is not None
        assert self._sin_cached is not None
        if self.scale is None:
            return (
                apply_rotary_emb_torch(
                    q,
                    self._cos_cached,
                    self._sin_cached,
                    self.interleaved,
                    True,  # inplace=True
                ),
                apply_rotary_emb_torch(
                    k,
                    self._cos_cached,
                    self._sin_cached,
                    self.interleaved,
                    True,  # inplace=True
                ),
            )  # type: ignore
        else:
            assert False


def correction_fn(expansion_ratio: float, d_model: int) -> int:
    """Compute corrected dimension for SwiGLU."""
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class MLP(nn.Module):
    def __init__(self, config: PLMConfig, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.expansion_ratio = config.expansion_ratio
        self.bias = bias
        self.layernorm = nn.LayerNorm(config.hidden_size, bias=bias)
        self.up_proj = nn.Linear(config.hidden_size, correction_fn(config.expansion_ratio, config.hidden_size), bias=bias)
        self.down_proj = nn.Linear(correction_fn(config.expansion_ratio, config.hidden_size), config.hidden_size, bias=bias)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.up_proj(self.layernorm(x))))


class MultiHeadAttention(nn.Module):
    """Multi-head attention with rotary embeddings.
    
    Args:
        config: PLMConfig
    """
    def __init__(
        self,
        config: PLMConfig,
    ):
        super().__init__()
        self.d_model = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.d_head = self.d_model // self.n_heads
        self.attn_backend = config.attn_backend
        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_o = nn.Linear(self.d_model, self.d_model, bias=False)
        self.q_ln = nn.LayerNorm(self.d_model, bias=False)
        self.k_ln = nn.LayerNorm(self.d_model, bias=False)
        self.reshaper = partial(rearrange, pattern="b s (h d) -> b h s d", h=self.n_heads)
        self.rotary = RotaryEmbedding(self.d_model // self.n_heads)
        self.scale = 1.0 / math.sqrt(self.d_head)

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key."""
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        flex_block_mask: object,
        output_attentions: bool = False,
        output_s_max: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        """
        Args:
            x: Input tensor
            attention_mask: 4D attention mask
            flex_block_mask: Flex attention block mask
            output_attentions: Whether to return attention weights
            
        Returns:
            Output tensor after self attention, and optionally attention weights
        """
        attn_weights = None
        query_BLD = self.q_ln(self.W_q(x))
        key_BLD = self.k_ln(self.W_k(x))
        value_BLD = self.W_v(x)
        query_BLD, key_BLD = self._apply_rotary(query_BLD, key_BLD)
        query_BHLD, key_BHLD, value_BHLD = map(self.reshaper, (query_BLD, key_BLD, value_BLD))

        if output_attentions: # Manual attention computation
            attn_weights = torch.matmul(query_BHLD, key_BHLD.transpose(-2, -1)) * self.scale
            attn_weights = attn_weights.masked_fill(attention_mask.logical_not(), float('-inf'))
            attn_weights = F.softmax(attn_weights, dim=-1)
            context_BHLD = torch.matmul(attn_weights, value_BHLD)
        else:
            if self.attn_backend == "flex":
                assert flex_attention is not None, "Flex attention backend requested but torch.flex_attention is unavailable."
                assert query_BHLD.dtype in (torch.float16, torch.bfloat16), f"Flex attention backend requires float16 or bfloat16, got {query_BHLD.dtype}."
                assert flex_block_mask is not None, "Flex attention backend requires a block mask when attention_mask is provided."
                context_BHLD = flex_attention(
                    query_BHLD,
                    key_BHLD,
                    value_BHLD,
                    block_mask=flex_block_mask,
                    scale=self.scale,
                )
            else:
                context_BHLD = F.scaled_dot_product_attention(
                    query_BHLD,
                    key_BHLD,
                    value_BHLD,
                    attn_mask=attention_mask,
                    scale=self.scale,
                )
            
        context_BLD = rearrange(context_BHLD, "b h s d -> b s (h d)")
        output = self.W_o(context_BLD)
        
        s_max = None
        if output_s_max:
            with torch.no_grad():
                q_norm = torch.linalg.vector_norm(query_BHLD, dim=-1)
                k_norm = torch.linalg.vector_norm(key_BHLD, dim=-1)
                s_max_bound = (q_norm.max(dim=-1).values * k_norm.max(dim=-1).values).max(dim=0).values * self.scale
                s_max = [s_max_bound[h] for h in range(self.n_heads)]
                
        return output, attn_weights, s_max


### Regression Head
def RegressionHead(d_model: int, output_dim: int, hidden_dim: Optional[int] = None) -> nn.Module:
    """Create a regression head with optional hidden dimension.
    
    Args:
        d_model: Input dimension
        output_dim: Output dimension
        hidden_dim: Optional hidden dimension (defaults to d_model)
    """
    hidden_dim = hidden_dim if hidden_dim is not None else d_model
    return nn.Sequential(
        nn.Linear(d_model, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, output_dim),
    )


class TransformerBlock(nn.Module):
    """Transformer block with attention and feedforward layers.
    
    Args:
        config: PLMConfig
    """
    def __init__(
        self,
        config: PLMConfig,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.mlp = MLP(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        flex_block_mask: object,
        output_attentions: bool = False,
        output_s_max: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        """
        Args:
            x: Input tensor
            attention_mask: 4D attention mask
            flex_block_mask: Flex attention block mask
            output_attentions: Whether to return attention weights
            
        Returns:
            Output tensor after transformer block, and optionally attention weights
        """
        attn_output, attn_weights, s_max = self.attn(
            x,
            attention_mask,
            flex_block_mask,
            output_attentions,
            output_s_max,
        )
        x = x + self.dropout(attn_output)
        x = x + self.dropout(self.mlp(x))
        return x, attn_weights, s_max


### Model Outputs
@dataclass
class TransformerOutput(ModelOutput):
    """Output type for transformer encoder."""
    last_hidden_state: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    s_max: Optional[Tuple[List[torch.Tensor]]] = None


@dataclass
class PLMOutput(ModelOutput):
    """Output type for PLM models."""
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    s_max: Optional[Tuple[List[torch.Tensor]]] = None


### Transformer Stack
class TransformerStack(nn.Module):
    """Stack of transformer blocks.
    
    Args:
        config: PLMConfig
    """
    def __init__(
        self,
        config: PLMConfig,
    ):
        super().__init__()
        self._attn_backend = config.attn_backend
        self.sliding_window_size = config.sliding_window_size
        self.dilation = config.dilation
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.hidden_size, bias=False)
        self.gradient_checkpointing = False
        self.attn_backend = config.attn_backend
        self.sliding_window_size = config.sliding_window_size
        self.dilation = config.dilation

    @property
    def attn_backend(self) -> str:
        return self._attn_backend

    @attn_backend.setter
    def attn_backend(self, backend: str) -> None:
        assert backend in ("sdpa", "flex"), f"Unsupported attn_backend: {backend}"
        self._attn_backend = backend
        for block in self.blocks:
            block.attn.attn_backend = backend

    def get_attention_mask(
        self,
        batch_size: int, 
        seq_len: int, 
        device: torch.device,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if attention_mask is None:
            token_attention_mask = torch.ones((batch_size, seq_len), device=device).bool() 
        else:
            token_attention_mask = attention_mask.bool()
        
        if self.attn_backend == "flex":
            if attention_mask is None:
                flex_block_mask = None
            else:
                def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
                    diff = torch.abs(q_idx - kv_idx)
                    in_window = diff <= self.sliding_window_size
                    is_dilated = (diff % self.dilation) == 0
                    document_mask = (token_attention_mask[batch_idx, q_idx] == token_attention_mask[batch_idx, kv_idx]) & (token_attention_mask[batch_idx, q_idx] != 0)
                    return (in_window | is_dilated) & document_mask
        
                flex_block_mask = create_block_mask(
                    mask_mod,
                    batch_size,
                    1,
                    seq_len,
                    seq_len,
                    device=device,
                )
            extended_attention_mask = None
        else:
            flex_block_mask = None
            extended_attention_mask = token_attention_mask[:, None, :, None] & token_attention_mask[:, None, None, :]

        return extended_attention_mask, flex_block_mask

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_s_max: Optional[bool] = False,
    ) -> TransformerOutput:
        """
        Args:
            x: Input tensor
            attention_mask: Optional 2D attention mask
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights
            
        Returns:
            TransformerOutput containing last hidden state and optionally all hidden states and attention weights
        """
        hidden_states = () if output_hidden_states else None
        attentions = () if output_attentions else None
        s_max_list = () if output_s_max else None
        
        # move to 4D attention mask or flex block mask
        attention_mask, flex_block_mask = self.get_attention_mask(
            batch_size=x.shape[0],
            seq_len=x.shape[1],
            device=x.device,
            attention_mask=attention_mask,
        )
            
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x, attn_weights = self._gradient_checkpointing_func(
                    block.__call__,
                    x=x,
                    attention_mask=attention_mask,
                    flex_block_mask=flex_block_mask,
                    output_attentions=output_attentions,
                )
            else:
                x, attn_weights, s_max = block(
                    x=x,
                    attention_mask=attention_mask,
                    flex_block_mask=flex_block_mask,
                    output_attentions=output_attentions,
                    output_s_max=output_s_max,
                )

            if attentions is not None:
                attentions += (attn_weights,)
            if s_max_list is not None:
                s_max_list += (s_max,)
                
            if output_hidden_states:
                assert hidden_states is not None
                hidden_states += (x,)
        
        last_hidden_state = self.norm(x)
        if output_hidden_states:
            hidden_states += (last_hidden_state,)
                
        return TransformerOutput(
            last_hidden_state=last_hidden_state, 
            hidden_states=hidden_states,
            attentions=attentions,
            s_max=s_max_list,
        )


class PreTrainedPLM(PreTrainedModel):
    config_class = PLMConfig
    base_model_prefix = "plm"
    supports_gradient_checkpointing = True
    all_tied_weights_keys = {}

    def _init_weights(self, module):
        """Initialize the weights"""
        # HF from_pretrained marks loaded parameters with `_is_hf_initialized`.
        # Skip this module if any local parameter is already marked as loaded.
        for parameter in module.parameters(recurse=False):
            if "_is_hf_initialized" in parameter.__dict__ and parameter.__dict__["_is_hf_initialized"]:
                return

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    @property
    def attn_backend(self) -> str:
        return self.config.attn_backend

    @attn_backend.setter
    def attn_backend(self, backend: str) -> None:
        assert backend in ("sdpa", "flex"), f"Unsupported attn_backend: {backend}"
        self.config.attn_backend = backend
        for module in self.modules():
            if isinstance(module, TransformerStack):
                module.attn_backend = backend

    def _reset_rotary_embeddings(self):
        """Refresh non-persistent rotary buffers after checkpoint loading."""
        for module in self.modules():
            if isinstance(module, RotaryEmbedding):
                module.reset_parameters()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        output_loading_info = bool(kwargs["output_loading_info"]) if "output_loading_info" in kwargs else False
        loaded = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        if output_loading_info:
            model, loading_info = loaded
            model._reset_rotary_embeddings()
            return model, loading_info
        loaded._reset_rotary_embeddings()
        return loaded


class PLM(PreTrainedPLM, EmbeddingMixin):
    """
    PLM model. transformer model with no heads
    """
    config_class = PLMConfig
    def __init__(self, config: PLMConfig, **kwargs):
        PreTrainedPLM.__init__(self, config, **kwargs)
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed = nn.Embedding(self.vocab_size, config.hidden_size)
        self.transformer = TransformerStack(config)
        self.tokenizer = EsmSequenceTokenizer()
        self.init_weights()

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed(input_ids)
        return self.transformer(
            x=x,
            attention_mask=attention_mask,
            output_hidden_states=False,
            output_attentions=False,
        ).last_hidden_state

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_s_max: Optional[bool] = None,
        return_dict: Optional[bool] = None, # to play nice with HF adjacent packages
        **kwargs,
    ) -> TransformerOutput:
        """Forward pass for masked language modeling.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            inputs_embeds: Optional precomputed embeddings
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights
            
        Returns:
            TransformerOutput containing last hidden state and optionally all hidden states and attention weights
        """
        assert input_ids is not None or inputs_embeds is not None, "You have to specify either input_ids or inputs_embeds"
        assert not (input_ids is not None and inputs_embeds is not None), "You cannot specify both input_ids and inputs_embeds at the same time"
        
        if inputs_embeds is None:
            x = self.embed(input_ids)
        else:
            x = inputs_embeds
        
        transformer_output = self.transformer(
            x=x,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
        )
        return PLMOutput(
            last_hidden_state=transformer_output.last_hidden_state,
            hidden_states=transformer_output.hidden_states,
            attentions=transformer_output.attentions,
            s_max=transformer_output.s_max,
        )

class PLMForMaskedLM(PreTrainedPLM, EmbeddingMixin):
    config_class = PLMConfig
    def __init__(self, config: PLMConfig, **kwargs):
        PreTrainedPLM.__init__(self, config, **kwargs)
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed = nn.Embedding(self.vocab_size, config.hidden_size)
        self.transformer = TransformerStack(config)
        self.lm_head = LMHead(config)
        self.ce_loss = nn.CrossEntropyLoss()
        self.tokenizer = EsmSequenceTokenizer()
        self.init_weights()

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def get_output_embeddings(self):
        return self.sequence_head[-1]

    def set_output_embeddings(self, new_embeddings):
        self.sequence_head[-1] = new_embeddings

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed(input_ids)
        return self.transformer(
            x=x,
            attention_mask=attention_mask,
            output_hidden_states=False,
            output_attentions=False,
        ).last_hidden_state

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, # to play nice with HF adjacent packages
        **kwargs,
    ) -> PLMOutput:
        """Forward pass for masked language modeling.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            inputs_embeds: Optional precomputed embeddings
            labels: Optional labels for masked tokens
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights
            
        Returns:
            PLMOutput containing loss, logits, hidden states and attention weights
        """
        if inputs_embeds is None:
            x = self.embed(input_ids)
        else:
            x = inputs_embeds
    
        output = self.transformer(
            x=x,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        last_hidden_state = output.last_hidden_state
        logits = self.lm_head(last_hidden_state)
        loss = None
        if labels is not None:
            loss = self.ce_loss(logits.view(-1, self.vocab_size), labels.view(-1))
        
        return PLMOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=last_hidden_state,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


class PLMForSequenceClassification(PLM, EmbeddingMixin):
    """
    PLM model for sequence classification.
    Extends the base PLM model with a classification head.
    """
    def __init__(self, config: PLMConfig, **kwargs):
        PLM.__init__(self, config, **kwargs)
        self.config = config
        self.num_labels = config.num_labels
        self.classifier = RegressionHead(config.hidden_size * 2, config.num_labels, config.hidden_size * 4)
        # Large intermediate projections help with sequence classification tasks (*4)
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        # if kwargs has pooling_types, use them, otherwise use ['cls', 'mean']
        if 'pooling_types' in kwargs and isinstance(kwargs['pooling_types'], List[str]) and len(kwargs['pooling_types']) > 0:
            pooling_types = kwargs['pooling_types']
        else:
            pooling_types = ['mean', 'var']
        self.pooler = Pooler(pooling_types)
        self.init_weights()

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed(input_ids)
        return self.transformer(
            x=x,
            attention_mask=attention_mask,
            output_hidden_states=False,
            output_attentions=False,
        ).last_hidden_state

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, # to play nice with HF adjacent packages
        **kwargs,
    ) -> PLMOutput:
        """Forward pass for sequence classification.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            inputs_embeds: Optional precomputed embeddings
            labels: Optional labels for classification
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights
            
        Returns:
            PLMOutput containing loss, logits, and hidden states
        """
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        last_hidden_state = output.last_hidden_state
        features = self.pooler(last_hidden_state, attention_mask) # pooler expects 2d attention mask
        logits = self.classifier(features)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.num_labels == 1:
                    loss = self.mse(logits.flatten(), labels.flatten())
                else:
                    loss = self.mse(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = self.ce(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = self.bce(logits, labels)

        return PLMOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=last_hidden_state,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


class PLMForTokenClassification(PLM, EmbeddingMixin):
    """
    PLM model for token classification.
    Extends the base PLM model with a token classification head.
    """
    def __init__(self, config: PLMConfig, **kwargs):
        PLM.__init__(self, config, **kwargs)
        self.config = config
        self.num_labels = config.num_labels
        self.classifier = RegressionHead(config.hidden_size, config.num_labels, config.hidden_size * 4)
        # Large intermediate projections help with sequence classification tasks (*4)
        self.loss_fct = nn.CrossEntropyLoss()
        self.init_weights()

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed(input_ids)
        return self.transformer(x, attention_mask, output_hidden_states=False, output_attentions=False).last_hidden_state

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, # to play nice with HF adjacent packages
        **kwargs,
    ) -> PLMOutput:
        """Forward pass for token classification.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            inputs_embeds: Optional precomputed embeddings
            labels: Optional labels for token classification
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights
            
        Returns:
            PLMOutput containing loss, logits, and hidden states
        """
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        last_hidden_state = output.last_hidden_state
        logits = self.classifier(last_hidden_state)
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return PLMOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=last_hidden_state,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


### Tokenization
SEQUENCE_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
    "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z",
    "O", ".", "-", "|",
    "<mask>",
]

class EsmSequenceTokenizer(PreTrainedTokenizerFast):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        unk_token="<unk>",
        cls_token="<cls>",
        pad_token="<pad>",
        mask_token="<mask>",
        eos_token="<eos>",
        chain_break_token="|",
        **kwargs,
    ):
        all_tokens = SEQUENCE_VOCAB
        token_to_id = {tok: ind for ind, tok in enumerate(all_tokens)}

        # a character-level tokenizer is the same as BPE with no token merges
        bpe = BPE(token_to_id, merges=[], unk_token=unk_token)
        tokenizer = Tokenizer(bpe)
        special_tokens = [
            cls_token,
            pad_token,
            mask_token,
            eos_token,
            chain_break_token,
        ]
        self.cb_token = chain_break_token
        additional_special_tokens = [chain_break_token]

        tokenizer.add_special_tokens(special_tokens)

        # This is where we configure the automatic addition of special tokens when we call
        # tokenizer(text, add_special_tokens=True). Note that you can also configure how two
        # sequences are merged if you want.
        tokenizer.post_processor = TemplateProcessing(  # type: ignore
            single="<cls> $A <eos>",
            pair="<cls>:0 $A:0 <eos>:0 $B:1 <eos>:1",
            special_tokens=[
                ("<cls>", tokenizer.token_to_id("<cls>")),
                ("<eos>", tokenizer.token_to_id("<eos>")),
            ],
        )
        super().__init__(
            tokenizer_object=tokenizer,
            unk_token=unk_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            eos_token=eos_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

    # These are a footgun, we never use the `bos` token anywhere so we're just overriding it here.
    @property
    def bos_token(self):
        return self.cls_token

    @property
    def bos_token_id(self):
        return self.cls_token_id

    @property
    def chain_break_token(self):
        return self.cb_token

    @property
    def chain_break_token_id(self):
        return self.convert_tokens_to_ids(self.chain_break_token)

    @property
    def all_token_ids(self):
        return list(range(self.vocab_size))

    @property
    def special_token_ids(self):
        return self.all_special_ids
