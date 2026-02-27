import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel, PreTrainedTokenizerFast, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput
from transformers.utils import logging

from models.FastPLMs.embedding_mixin import Pooler


logger = logging.get_logger(__name__)

### Establish attention compatibility
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    logger.warning("Failed to import flash attention; Will be using PyTorch attention instead")
    flash_attn_func = None
    flash_attn_varlen_func = None

try:
    from torch.nn.attention.flex_attention import (
        BlockMask,
        create_block_mask,
        flex_attention,
    )
except ImportError:
    logger.warning("Failed to import torch flex attention; flex fallback will be unavailable.")
    BlockMask = Any  # type: ignore[assignment]
    create_block_mask = None
    flex_attention = None

try:
    from kernels import get_kernel
    try:
        flash_kernel = get_kernel("kernels-community/flash-attn3")
        logger.info(f"Using flash-attn3 kernel: {flash_kernel}")
    except Exception as e1:
        logger.warning(f"Failed to load flash-attn3 kernel: {e1}")
        try:
            flash_kernel = get_kernel("kernels-community/flash-attn2")
            logger.info(f"Using flash-attn2 kernel: {flash_kernel}")
        except Exception as e2:
            logger.warning(f"Failed to load flash-attn2 kernel: {e2}")
            flash_kernel = None
    try:
        layer_norm = get_kernel("kernels-community/triton-layer-norm")
        logger.info(f"Using triton-layer-norm kernel: {layer_norm}")
    except Exception as e3:
        logger.warning(f"Failed to load triton-layer-norm kernel: {e3}")
        layer_norm = None
        logger.warning("Will be using PyTorch RMSNorm instead")
except Exception as e:
    logger.warning(f"Failed to load kernels package components: {e}")
    flash_kernel = None
    layer_norm = None


def is_kernels_flash_attention_available() -> bool:
    return flash_kernel is not None and (os.getenv("USE_KERNELS_FLASH_ATTN", "1") == "1")


def is_flex_attention_available() -> bool:
    return flex_attention is not None


class AttentionBackend(Enum):
    AUTO = "auto"
    KERNELS_FLASH = "kernels_flash"
    FLASH_ATTN = "flash_attn"
    FLEX = "flex"
    SDPA = "sdpa"


def _validate_attention_backend(attn_backend: str) -> None:
    valid_backends = (
        AttentionBackend.AUTO.value,
        AttentionBackend.KERNELS_FLASH.value,
        AttentionBackend.FLASH_ATTN.value,
        AttentionBackend.FLEX.value,
        AttentionBackend.SDPA.value,
    )
    assert attn_backend in valid_backends, (
        f"Unsupported attention backend: {attn_backend}. "
        f"Expected one of {valid_backends}."
    )


def _build_nonpad_attention_mask(
    q_sequence_ids: torch.Tensor,
    k_sequence_ids: torch.Tensor,
) -> torch.Tensor:
    q_valid = q_sequence_ids != -1
    k_valid = k_sequence_ids != -1
    return q_valid[:, None, :, None] & k_valid[:, None, None, :]


def create_nonpad_block_mask(
    q_sequence_ids: torch.Tensor,
    k_sequence_ids: torch.Tensor,
) -> BlockMask:
    assert create_block_mask is not None, "Flex attention block mask creation is unavailable in this environment."
    batch_size, q_len = q_sequence_ids.shape
    _, k_len = k_sequence_ids.shape

    def non_pad_mask(b, h, q_idx, kv_idx):  # type: ignore[no-untyped-def]
        return (q_sequence_ids[b, q_idx] != -1) & (k_sequence_ids[b, kv_idx] != -1)

    return create_block_mask(non_pad_mask, batch_size, 1, q_len, k_len, device=q_sequence_ids.device)


def _repeat_kv_for_sdpa(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def sdpa_attention_func(
    query_states: torch.Tensor,  # (bs, q_len, nh, hs)
    key_states: torch.Tensor,  # (bs, kv_len, nkv, hs)
    value_states: torch.Tensor,  # (bs, kv_len, nkv, hs)
    q_sequence_ids: torch.Tensor,
    k_sequence_ids: torch.Tensor,
    num_key_value_groups: int,
) -> torch.Tensor:
    query_BHLD = query_states.transpose(1, 2).contiguous()
    key_BHLD = key_states.transpose(1, 2).contiguous()
    value_BHLD = value_states.transpose(1, 2).contiguous()
    key_BHLD = _repeat_kv_for_sdpa(key_BHLD, num_key_value_groups)
    value_BHLD = _repeat_kv_for_sdpa(value_BHLD, num_key_value_groups)
    attention_mask = _build_nonpad_attention_mask(q_sequence_ids, k_sequence_ids)
    out_BHLD = F.scaled_dot_product_attention(query_BHLD, key_BHLD, value_BHLD, attn_mask=attention_mask, is_causal=False)
    return out_BHLD.transpose(1, 2).contiguous()


def kernels_flash_attention_func(
    query_states: torch.Tensor,  # (bs, seqlen, nh, hs)
    key_states: torch.Tensor,  # (bs, seqlen, nkv, hs)
    value_states: torch.Tensor,  # (bs, seqlen, nkv, hs)
    q_sequence_ids: torch.Tensor,
    k_sequence_ids: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    assert is_kernels_flash_attention_available(), "Kernel Flash Attention is not available in this environment."

    if not causal:
        batch_size, q_len = query_states.shape[0], query_states.shape[1]
        (
            query_states,
            key_states,
            value_states,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        ) = _unpad_input(query_states, key_states, value_states, q_sequence_ids, k_sequence_ids)

        attn_output_unpad = flash_kernel.varlen_fwd(  # type: ignore[union-attr]
            q=query_states,
            k=key_states,
            v=value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            is_causal=False,
        )[0]
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, q_len)
    else:
        attn_output = flash_kernel.fwd(q=query_states, k=key_states, v=value_states, is_causal=True)[0]  # type: ignore[union-attr]

    return attn_output


def is_flash_attention_available() -> bool:
    return (
        flash_attn_func is not None and flash_attn_varlen_func is not None and (os.getenv("USE_FLASH_ATTN", "1") == "1")
    )


def flex_attention_func(
    query_states: torch.Tensor,  # (bs, seqlen, nh, hs)
    key_states: torch.Tensor,  # (bs, seqlen, nkv, hs)
    value_states: torch.Tensor,  # (bs, seqlen, nkv, hs)
    block_mask: BlockMask | None = None,
) -> torch.Tensor:
    assert flex_attention is not None, "Flex Attention is not available in this environment"
    query_states = query_states.transpose(1, 2).contiguous()  # (bs, nh, seqlen, hs)
    key_states = key_states.transpose(1, 2).contiguous()  # (bs, nkv, seqlen, hs)
    value_states = value_states.transpose(1, 2).contiguous()  # (bs, nkv, seqlen, hs)

    outputs = flex_attention(
        query_states,
        key_states,
        value_states,
        block_mask=block_mask,
        enable_gqa=query_states.shape[1] != key_states.shape[1],  # if nkv != nh
    )

    outputs = outputs.transpose(1, 2)  # (bs, seqlen, nh, hs)
    return outputs


def flash_attn_package_attention_func(
    query_states: torch.Tensor,  # (bs, seqlen, nh, hs)
    key_states: torch.Tensor,  # (bs, seqlen, nkv, hs)
    value_states: torch.Tensor,  # (bs, seqlen, nkv, hs)
    q_sequence_ids: torch.Tensor,
    k_sequence_ids: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:  # (bs, seqlen, nh, hs)
    # Contains at least one padding token in the sequence. Note: ignore attention mask if causal.
    if not is_flash_attention_available():
        raise ImportError("Flash Attention is not available. Please install flash-attn.")

    if not causal:
        batch_size, q_len = query_states.shape[0], query_states.shape[1]
        (
            query_states,
            key_states,
            value_states,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        ) = _unpad_input(query_states, key_states, value_states, q_sequence_ids, k_sequence_ids)

        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            causal=False,
        )
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, q_len)

    else:
        attn_output = flash_attn_func(query_states, key_states, value_states, causal=True)

    return attn_output


class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices) -> torch.Tensor:  # type: ignore[no-untyped-def]
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        # return input[indices]
        return torch.gather(rearrange(input, "b ... -> b (...)"), 0, repeat(indices, "z -> z d", d=second_dim)).reshape(
            -1, *other_shape
        )

    @staticmethod
    def backward(ctx, grad_output) -> tuple[torch.Tensor, None]:  # type: ignore[no-untyped-def]
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]], device=grad_output.device, dtype=grad_output.dtype
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # grad_input[indices] = grad_output
        grad_input.scatter_(0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim) -> torch.Tensor:  # type: ignore[no-untyped-def]
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype)
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        output[indices] = values
        # output.scatter_(0, repeat(indices, 'z -> z d', d=values.shape[1]), values)
        return output

    @staticmethod
    def backward(ctx, grad_output) -> tuple[torch.Tensor, None, None]:  # type: ignore[no-untyped-def]
        (indices,) = ctx.saved_tensors
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        grad_values = grad_output[indices]
        # grad_values = torch.gather(grad_output, 0, repeat(indices, 'z -> z d', d=grad_output.shape[1]))
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply


def pad_input(hidden_states: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    # output = torch.zeros((batch * seqlen), dim, device=hidden_states.device, dtype=hidden_states.dtype)
    # output[indices] = hidden_states
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def _get_unpad_data(sequence_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    non_pad_indices = sequence_ids != -1
    non_pad_indices = torch.nonzero(non_pad_indices.flatten(), as_tuple=False).flatten()
    sequence_ids = sequence_ids + torch.arange(len(sequence_ids), device=sequence_ids.device)[:, None] * 1e5
    sequence_ids = sequence_ids.flatten()[non_pad_indices]
    _, seqlens_in_batch = torch.unique_consecutive(sequence_ids, return_counts=True)
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return non_pad_indices, cu_seqlens, max_seqlen_in_batch


def _unpad_input(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    q_sequence_ids: torch.Tensor,
    k_sequence_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], tuple[int, int]]:
    batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape
    query_length, num_q_heads = query_layer.shape[1], query_layer.shape[2]
    assert query_layer.shape[:2] == q_sequence_ids.shape, (
        f"Shape mismatch between query layer and query sequence ids: {query_layer.shape[:2]} != {q_sequence_ids.shape}"
    )
    assert key_layer.shape[:2] == k_sequence_ids.shape, (
        f"Shape mismatch between key layer and key sequence ids: {key_layer.shape[:2]} != {k_sequence_ids.shape}"
    )
    assert query_length <= kv_seq_len, (
        f"Query length should be less than or equal to KV sequence length: {query_length} <= {kv_seq_len}"
    )

    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(k_sequence_ids)

    key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
    value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

    if torch.equal(q_sequence_ids, k_sequence_ids):
        indices_q = indices_k
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
    else:
        indices_q, cu_seqlens_q, max_seqlen_in_batch_q = _get_unpad_data(q_sequence_ids)

    query_layer = index_first_axis(query_layer.reshape(batch_size * query_length, num_q_heads, head_dim), indices_q)

    assert cu_seqlens_q.shape == cu_seqlens_k.shape, (
        f"Query and KV should have the same number of sequences: {cu_seqlens_q.shape} != {cu_seqlens_k.shape}"
    )

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


index_first_axis = IndexFirstAxis.apply
PAD_TOKEN_ID = 0


def get_tokenizer() -> Tokenizer:
    try:
        fname = os.path.join(os.path.dirname(__file__), "tokenizer.json")
        tokenizer: Tokenizer = Tokenizer.from_file(fname)
    except Exception:
        print("E1 Tokenizer not found in local directory, downloading from Hugging Face")
        from huggingface_hub import hf_hub_download
        fname = hf_hub_download(repo_id="Synthyra/Profluent-E1-150M", filename="tokenizer.json")
        tokenizer: Tokenizer = Tokenizer.from_file(fname)
    assert tokenizer.padding["pad_id"] == PAD_TOKEN_ID, (
        f"Padding token id must be {PAD_TOKEN_ID}, but got {tokenizer.padding['pad_id']}"
    )

    return tokenizer


@dataclass
class DataPrepConfig:
    max_num_sequences: int = 512
    max_num_positions_within_seq: int = 8192
    remove_X_tokens: bool = False


def get_context(sequence: str) -> str | None:
    if "," in sequence:
        return sequence.rsplit(",", 1)[0]
    return None


class E1BatchPreparer:
    def __init__(
        self,
        data_prep_config: DataPrepConfig | None = None,
        tokenizer: Tokenizer | None = None,
        preserve_context_labels: bool = False,
    ):
        self.tokenizer = tokenizer or get_tokenizer()
        self.data_prep_config = data_prep_config or DataPrepConfig()
        self.pad_token_id = self.tokenizer.token_to_id("<pad>")
        self.preserve_context_labels = preserve_context_labels
        device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
        self.boundary_token_ids = torch.tensor(
            [self.tokenizer.token_to_id(token) for token in ["<bos>", "<eos>", "1", "2", "<pad>"]], device=device
        ).long()
        self.mask_token = "?"  # nosec
        self.mask_token_id = self.tokenizer.token_to_id(self.mask_token)
        self.X_token_id = self.tokenizer.token_to_id("X")
        self.vocab = self.tokenizer.get_vocab()

    def get_batch_kwargs(  # type: ignore[override]
        self, sequences: list[str], device: torch.device = torch.device("cpu"), non_blocking: bool = False
    ) -> dict[str, torch.Tensor | list[str] | list[int]]:
        sequence_encodings = [self.prepare_multiseq(sequence) for sequence in sequences]
        return self.pad_encodings(sequence_encodings, device, non_blocking)

    def pad_encodings(
        self,
        sequence_encodings: list[dict[str, torch.Tensor]],
        device: torch.device = torch.device("cpu"),
        non_blocking: bool = False,
    ) -> dict[str, torch.Tensor | list[str] | list[int]]:
        non_blocking = non_blocking and device.type == "cuda"
        padded_encodings = {}
        # Note: We use -1 as the padding value for sequence and position ids because the 0 value
        # is a valid value for sequence and position ids. -1 is then used to distinguish valid
        # tokens from padding tokens, for example, when doing padding/unpadding for flash attention.
        for key, padding_value in {
            "input_ids": self.pad_token_id,
            "sequence_ids": -1,
            "within_seq_position_ids": -1,
            "global_position_ids": -1,
            "labels": self.pad_token_id,
        }.items():
            padded_encodings[key] = pad_sequence(
                [enc[key] for enc in sequence_encodings], batch_first=True, padding_value=padding_value
            ).to(device=device, dtype=torch.long, non_blocking=non_blocking)

        padded_encodings["context"] = [enc["context"] for enc in sequence_encodings]
        padded_encodings["context_len"] = [enc["context_len"] for enc in sequence_encodings]

        return padded_encodings

    def prepare_multiseq(self, sequence: str) -> dict[str, torch.Tensor | str | int]:
        single_sequences = sequence.split(",")
        if len(single_sequences) > self.data_prep_config.max_num_sequences:
            raise ValueError(
                f"Number of sequences {len(single_sequences)} exceeds max number of sequences {self.data_prep_config.max_num_sequences}"
                " in the provided multi-sequence instance. Please remove some homologous sequences before trying again."
            )

        single_sequence_encodings = [self.prepare_singleseq(sequence) for sequence in single_sequences]

        num_tokens = [len(x["input_ids"]) for x in single_sequence_encodings]
        input_ids = torch.cat([x["input_ids"] for x in single_sequence_encodings])
        labels = torch.cat([x["labels"] for x in single_sequence_encodings])

        within_seq_position_ids = torch.cat([encoding["position_ids"] for encoding in single_sequence_encodings])
        global_position_ids, ctx_len = [], 0
        for encoding in single_sequence_encodings:
            global_position_ids.append(encoding["position_ids"] + ctx_len)
            ctx_len = max(ctx_len, encoding["position_ids"].max().item() + ctx_len + 1)
        global_position_ids = torch.cat(global_position_ids)

        sequence_ids = torch.repeat_interleave(torch.tensor(num_tokens))

        # Get multi-seq context & mask out all but last sequence in multi-seq instance if desired
        context_len = sum(num_tokens[:-1])
        context = self.tokenizer.decode(input_ids[:context_len].tolist(), skip_special_tokens=False)
        if not self.preserve_context_labels:
            labels[:context_len] = self.pad_token_id

        assert (
            input_ids.shape
            == sequence_ids.shape
            == within_seq_position_ids.shape
            == global_position_ids.shape
            == labels.shape
        ), "Input ids, sequence ids, within seq position ids, global position ids, and labels must have the same shape"

        assert input_ids.shape[0] >= context_len, "Input ids must have at least as many tokens as the context length"

        return {
            "input_ids": input_ids,
            "sequence_ids": sequence_ids,
            "within_seq_position_ids": within_seq_position_ids,
            "global_position_ids": global_position_ids,
            "labels": labels,
            "context": context,
            "context_len": context_len,
        }

    def prepare_singleseq(self, sequence: str) -> dict[str, torch.Tensor]:
        if not self.validate_sequence(sequence):
            raise ValueError(f"Invalid sequence: {sequence}; Input sequence should contain [A-Z] or ? characters only")

        if len(sequence) > self.data_prep_config.max_num_positions_within_seq:
            raise ValueError(
                f"Sequence length {len(sequence)} exceeds max length {self.data_prep_config.max_num_positions_within_seq}"
            )

        # Can also use `tokens = torch.tensor(self.tokenizer.encode(f"<bos>1{sequence}2<eos>").ids)`
        # but following is faster since our vocabulary is simple.
        tokens = torch.tensor([self.vocab[token] for token in ["<bos>", "1", *sequence, "2", "<eos>"]])
        position_ids = torch.arange(len(tokens))

        if self.data_prep_config.remove_X_tokens:
            X_positions = torch.where(tokens != self.X_token_id)[0]
            tokens = tokens[X_positions]
            position_ids = position_ids[X_positions]

        return {"input_ids": tokens, "labels": tokens, "position_ids": position_ids}

    def get_boundary_token_mask(self, tokens: torch.Tensor) -> torch.BoolTensor:
        return torch.isin(tokens, self.boundary_token_ids.to(tokens.device))

    def get_mask_positions_mask(self, tokens: torch.Tensor) -> torch.BoolTensor:
        return tokens == self.mask_token_id

    def validate_sequence(self, sequence: str) -> bool:
        assert isinstance(sequence, str), "Sequence must be a string"
        sequence = sequence.replace(self.mask_token, "")
        return sequence.isalpha() and sequence.isupper()


class E1Config(PretrainedConfig):
    model_type = "E1"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(  # type: ignore
        self,
        # Model architecture/initialization
        vocab_size=None,
        hidden_size=4096,
        intermediate_size=16384,
        gated_mlp=False,
        num_hidden_layers=40,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        dtype="bfloat16",
        gradient_checkpointing=False,
        no_ffn_gradient_checkpointing=False,
        # Tokenization
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        tie_word_embeddings=False,
        # Attention implementation & rotary positional embeddings
        global_attention_every_n_layers=0,
        max_num_sequences=512,
        max_num_positions_within_seq=8192,
        max_num_positions_global=1024 * 128,
        rope_theta_within_seq=10000.0,
        rope_theta_global=100000.0,
        attn_backend="auto",
        clip_qkv=None,
        **kwargs,
    ) -> None:
        tokenizer = get_tokenizer()
        super().__init__(
            pad_token_id=tokenizer.token_to_id("<pad>"),
            bos_token_id=tokenizer.token_to_id("<bos>"),
            eos_token_id=tokenizer.token_to_id("<eos>"),
            tie_word_embeddings=tie_word_embeddings,
            dtype=dtype,
            **kwargs,
        )

        self.hidden_size = hidden_size
        if intermediate_size is None:
            intermediate_size = 3 * hidden_size if gated_mlp else 4 * hidden_size
        self.intermediate_size = intermediate_size
        self.gated_mlp = gated_mlp
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_num_positions_within_seq = max_num_positions_within_seq
        self.max_num_positions_global = max_num_positions_global

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta_within_seq = rope_theta_within_seq
        self.rope_theta_global = rope_theta_global
        self.max_num_sequences = max_num_sequences
        _validate_attention_backend(attn_backend)
        self.attn_backend = attn_backend
        assert clip_qkv is None or clip_qkv > 0
        self.clip_qkv = clip_qkv
        self.global_attention_every_n_layers = global_attention_every_n_layers

        self.vocab_size = tokenizer.get_vocab_size()
        self.gradient_checkpointing = gradient_checkpointing
        self.no_ffn_gradient_checkpointing = no_ffn_gradient_checkpointing

        if vocab_size is not None:
            if vocab_size < self.vocab_size:
                logger.warning(
                    f"Using vocab_size {vocab_size} smaller than {self.vocab_size} from tokenizer. MAKE SURE THIS IS INTENTIONAL."
                )
                self.vocab_size = vocab_size
            elif vocab_size > self.vocab_size:
                logger.warning(f"Using vocab_size {vocab_size} instead of smaller {self.vocab_size} from tokenizer.")
                self.vocab_size = vocab_size
        if pad_token_id is not None and pad_token_id != self.pad_token_id:
            logger.warning(f"Ignoring pad_token_id. Using {self.pad_token_id} from tokenizer")
        if bos_token_id is not None and bos_token_id != self.bos_token_id:
            logger.warning(f"Ignoring bos_token_id. Using {self.bos_token_id} from tokenizer")
        if eos_token_id is not None and eos_token_id != self.eos_token_id:
            logger.warning(f"Ignoring eos_token_id. Using {self.eos_token_id} from tokenizer")


class DynamicCache:
    """
    A cache layer that grows dynamically as more tokens are generated. This is the default for generative models.
    It stores the key and value states as tensors of shape `[batch_size, seq_len, num_heads, head_dim]`.

    Args:
        key_cache (`list[torch.Tensor]`): The list of key states.
        value_cache (`list[torch.Tensor]`): The list of value states.
    """

    def __init__(self) -> None:
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necessary keys and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache of shape [batch_size, seq_len, num_heads, head_dim]
            value_states (`torch.Tensor`): The new value states to cache of shape [batch_size, seq_len, num_heads, head_dim]
            layer_idx (`int`): The index of the layer to update.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states of shape [batch_size, seq_len, num_heads, head_dim].
        """
        # Lazy initialization
        if len(self.key_cache) <= layer_idx:
            # There may be skipped layers, fill them with empty lists
            for _ in range(len(self.key_cache), layer_idx):
                self.key_cache.append(torch.tensor([]))
                self.value_cache.append(torch.tensor([]))
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        elif (
            not self.key_cache[layer_idx].numel()  # prefers not t.numel() to len(t) == 0 to export the model
        ):  # fills previously skipped layers; checking for tensor causes errors
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=1)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=1)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or not self.key_cache[layer_idx].numel()  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[1] if not is_empty_layer else 0
        return layer_seq_length

    def crop(self, max_length: int) -> None:
        """Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens. This is used in assisted decoding and contrastive search."""
        assert max_length > 0, "max_length must be positive"

        if self.get_seq_length() <= max_length:
            return

        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx].numel():
                self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :max_length, ...]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :max_length, ...]

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx].numel():
                self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeats, dim=0)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx].numel():
                self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]


class KVCache:
    def __init__(self, cache_size: int = 4) -> None:
        self.cache_size = cache_size
        self.tensor_input_field_names = [
            "input_ids",
            "within_seq_position_ids",
            "global_position_ids",
            "sequence_ids",
            "labels",
        ]
        self.tensor_output_field_names = ["logits", "embeddings"]
        self.cache_dict: dict[str, DynamicCache] = {}
        self.cache_queue: list[str] = []

    def reset(self) -> None:
        for k in list(self.cache_dict.keys()):
            del self.cache_dict[k]
        del self.cache_dict
        self.cache_dict = {}
        self.cache_queue = []

        torch.cuda.empty_cache()

    def before_forward(self, batch: dict[str, torch.Tensor]) -> None:
        contexts: list[str] | None = batch.get("context", None)
        if contexts is None or "context_len" not in batch:
            logger.warning_once(
                "KVCache requires the batch dict to have both `context` and `context_len` keys to trigger. Skipping."
            )
            return

        context_lens: list[int] = list(set(batch["context_len"]))
        contexts: list[str] = list(set(contexts))  # type: ignore[no-redef]
        if len(contexts) != 1 or len(context_lens) != 1:
            logger.warning(
                "SingleContextKVCache requires a single context and context length. "
                "Multiple contexts or context lengths found in a single batch. Skipping."
            )
            return

        batch_size = batch["input_ids"].shape[0]

        unique_context = contexts[0]
        unique_context_len = context_lens[0]
        batch["use_cache"] = True

        if unique_context not in self.cache_dict:
            return

        self.cache_dict[unique_context].batch_repeat_interleave(batch_size)
        past_key_values = self.cache_dict[unique_context]
        batch["past_key_values"] = past_key_values

        # Remove context from the input fields
        for field_name in self.tensor_input_field_names:
            if batch.get(field_name, None) is not None:
                batch[field_name] = batch[field_name][:, unique_context_len:]

    def after_forward(self, batch: dict[str, Any], outputs: ModelOutput) -> None:
        contexts = batch.get("context", None)
        context_lens = batch.get("context_len", [])
        if contexts is None or len(set(contexts)) != 1 or len(set(context_lens)) != 1 or context_lens[0] == 0:
            return

        assert batch["use_cache"]
        unique_context = contexts[0]
        unique_context_len = context_lens[0]

        past_key_values = getattr(outputs, "past_key_values", None)
        if not isinstance(past_key_values, DynamicCache):
            logger.warning_once("KVCache is incompatible with models that don't return a DynamicCache. Skipping.")
            return

        if "past_key_values" not in batch:
            if len(self.cache_queue) == self.cache_size:
                last_context = self.cache_queue.pop(0)
                if last_context not in self.cache_queue:
                    del self.cache_dict[last_context]
                    torch.cuda.empty_cache()

            self.cache_dict[unique_context] = past_key_values
            self.cache_queue.append(unique_context)

            # Remove context from the input fields
            for field_name in self.tensor_input_field_names:
                if field_name in batch and batch[field_name] is not None:
                    batch[field_name] = batch[field_name][:, unique_context_len:]

            # Remove context from the output fields
            for field_name in self.tensor_output_field_names:
                if field_name in outputs and outputs[field_name] is not None:
                    outputs[field_name] = outputs[field_name][:, unique_context_len:]
            if "hidden_states" in outputs and outputs["hidden_states"] is not None:
                outputs["hidden_states"] = [h[:, unique_context_len:] for h in outputs["hidden_states"]]

        self.cache_dict[unique_context].crop(unique_context_len)
        self.cache_dict[unique_context].batch_select_indices([0])


class AttentionLayerType(Enum):
    WITHIN_SEQ = "within_seq"
    GLOBAL = "global"


class AttentionArgs(TypedDict, total=False):
    block_mask: BlockMask | None


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).

    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to (batch,
    num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self, dim: int, max_position_embeddings: int = 2048, base: int = 10000, device: torch.device | None = None
    ):
        super().__init__()

        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        inv_freq = base ** -(torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_sin_cos_cache(seq_len=max_position_embeddings, device=self.inv_freq.device)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _set_sin_cos_cache(self, seq_len: int, device: torch.device) -> None:
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        angles = torch.outer(t, self.inv_freq.to(device))
        angles = torch.cat((angles, angles), dim=1)
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.LongTensor, seq_len: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [bsz, seq_len, num_attention_heads, head_size]
        device, dtype = q.device, q.dtype
        seq_len = position_ids.max().item() + 1 if seq_len is None else seq_len

        if seq_len > self.max_seq_len_cached:
            self._set_sin_cos_cache(seq_len=seq_len, device=device)

        # angles_cached[position_ids] gets us something of shape (batch_size, seq_len, head_dim),
        # so unsqueeze dimension -2 to broadcast to (batch_size, seq_len, n_heads, head_dim).
        idxs = position_ids.to(device)
        cos = self.cos_cached.to(device=device, dtype=dtype).unsqueeze(-2)[idxs]
        sin = self.sin_cached.to(device=device, dtype=dtype).unsqueeze(-2)[idxs]

        # Apply rotary positional embeddings to q and k (treating them as complex numbers). The first half is
        # Re[x exp(it)] = Re[x] cos(t) - Im[x] sin(t), while the second half is
        # Im[x exp(it)] = Im[x] cos(t) + Re[x] sin(t). This works b/c both halves of cos/sin are the same.
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed


class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init__(self, config: E1Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.max_num_seqs = config.max_num_sequences
        self.clip_qkv = config.clip_qkv
        self.scale = 1.0 / (self.head_dim**0.5)

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        if self.config.global_attention_every_n_layers > 0:
            self.layer_type = (
                AttentionLayerType.GLOBAL
                if (self.layer_idx + 1) % self.config.global_attention_every_n_layers == 0
                else AttentionLayerType.WITHIN_SEQ
            )
        else:
            self.layer_type = AttentionLayerType.WITHIN_SEQ

        self.rope_theta = (
            config.rope_theta_within_seq
            if self.layer_type == AttentionLayerType.WITHIN_SEQ
            else config.rope_theta_global
        )
        self.max_position_embeddings = (
            config.max_num_positions_within_seq
            if self.layer_type == AttentionLayerType.WITHIN_SEQ
            else config.max_num_positions_global
        )

        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta
        )

    def prepare_qkv(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_value: DynamicCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = hidden_states.size()
        query_states: torch.Tensor = self.q_proj(hidden_states)
        key_states: torch.Tensor = self.k_proj(hidden_states)
        val_states: torch.Tensor = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim)
        val_states = val_states.view(bsz, q_len, self.num_kv_heads, self.head_dim)

        if self.clip_qkv is not None:
            query_states = query_states.clamp(-self.clip_qkv, self.clip_qkv)
            key_states = key_states.clamp(-self.clip_qkv, self.clip_qkv)
            val_states = val_states.clamp(-self.clip_qkv, self.clip_qkv)

        query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)

        if use_cache and past_key_value is not None:
            key_states, val_states = past_key_value.update(key_states, val_states, self.layer_idx)

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        else:
            target_dtype = self.q_proj.weight.dtype
        if input_dtype != target_dtype:
            logger.warning_once(
                f"The input hidden states seems to be silently casted in {input_dtype}. "
                f"This might be because you have upcasted embedding or layer norm layers "
                f"in {input_dtype}. We will cast back the input in {target_dtype}."
            )
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            val_states = val_states.to(target_dtype)

        return query_states, key_states, val_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        within_seq_position_ids: torch.LongTensor,
        global_position_ids: torch.LongTensor,
        sequence_ids: torch.LongTensor,
        attention_args: AttentionArgs | None = None,
        past_key_value: DynamicCache | None = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None, DynamicCache | None]:
        query_states, key_states, val_states = self.prepare_qkv(
            hidden_states=hidden_states,
            position_ids=within_seq_position_ids
            if self.layer_type == AttentionLayerType.WITHIN_SEQ
            else global_position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        q_sequence_ids, k_sequence_ids = self._build_qk_sequence_ids(
            sequence_ids=sequence_ids,
            q_len=query_states.shape[1],
            kv_len=key_states.shape[1],
        )

        attn_output, attn_weights, s_max = self._attn(
            query_states=query_states,
            key_states=key_states,
            val_states=val_states,
            q_sequence_ids=q_sequence_ids,
            k_sequence_ids=k_sequence_ids,
            attention_args=attention_args,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
        )

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, s_max, past_key_value

    def _build_qk_sequence_ids(
        self,
        sequence_ids: torch.Tensor,
        q_len: int,
        kv_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_sequence_ids = torch.where(
            sequence_ids != -1,
            torch.zeros_like(sequence_ids, dtype=torch.long),
            torch.full_like(sequence_ids, -1, dtype=torch.long),
        )
        if q_len < kv_len:
            first_token_id = q_sequence_ids[:, 0].unsqueeze(1)
            pad_prefix = first_token_id.expand(q_sequence_ids.shape[0], kv_len - q_len)
            k_sequence_ids = torch.cat([pad_prefix, q_sequence_ids], dim=-1)
        else:
            k_sequence_ids = q_sequence_ids
        return q_sequence_ids, k_sequence_ids

    def _resolve_attention_backend(self) -> AttentionBackend:
        requested_backend = self.config.attn_backend
        _validate_attention_backend(requested_backend)
        if requested_backend == AttentionBackend.AUTO.value:
            if is_kernels_flash_attention_available():
                return AttentionBackend.KERNELS_FLASH
            if is_flex_attention_available():
                return AttentionBackend.FLEX
            return AttentionBackend.SDPA
        if requested_backend == AttentionBackend.KERNELS_FLASH.value:
            if is_kernels_flash_attention_available():
                return AttentionBackend.KERNELS_FLASH
            if is_flex_attention_available():
                logger.warning_once("Falling back from kernels_flash to flex because kernels flash attention is unavailable.")
                return AttentionBackend.FLEX
            logger.warning_once("Falling back from kernels_flash to sdpa because kernels flash and flex attention are unavailable.")
            return AttentionBackend.SDPA
        if requested_backend == AttentionBackend.FLASH_ATTN.value:
            if is_flash_attention_available():
                return AttentionBackend.FLASH_ATTN
            if is_flex_attention_available():
                logger.warning_once("Falling back from flash_attn to flex because flash-attn package is unavailable.")
                return AttentionBackend.FLEX
            logger.warning_once("Falling back from flash_attn to sdpa because flash-attn and flex attention are unavailable.")
            return AttentionBackend.SDPA
        if requested_backend == AttentionBackend.FLEX.value:
            if is_flex_attention_available():
                return AttentionBackend.FLEX
            logger.warning_once("Falling back from flex to sdpa because flex attention is unavailable.")
            return AttentionBackend.SDPA
        return AttentionBackend.SDPA

    def _attn(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        val_states: torch.Tensor,
        q_sequence_ids: torch.Tensor,
        k_sequence_ids: torch.Tensor,
        attention_args: AttentionArgs | None = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        if output_attentions:
            return self._manual_attn(
                query_states=query_states,
                key_states=key_states,
                val_states=val_states,
                q_sequence_ids=q_sequence_ids,
                k_sequence_ids=k_sequence_ids,
                output_s_max=output_s_max,
            )

        backend = self._resolve_attention_backend()
        if backend == AttentionBackend.KERNELS_FLASH:
            attn_output, attn_weights = self._kernels_flash_attn(
                query_states=query_states,
                key_states=key_states,
                val_states=val_states,
                q_sequence_ids=q_sequence_ids,
                k_sequence_ids=k_sequence_ids,
            )
        elif backend == AttentionBackend.FLASH_ATTN:
            attn_output, attn_weights = self._flash_attn_package(
                query_states=query_states,
                key_states=key_states,
                val_states=val_states,
                q_sequence_ids=q_sequence_ids,
                k_sequence_ids=k_sequence_ids,
            )
        elif backend == AttentionBackend.FLEX:
            attn_output, attn_weights = self._flex_attn(
                query_states=query_states,
                key_states=key_states,
                val_states=val_states,
                q_sequence_ids=q_sequence_ids,
                k_sequence_ids=k_sequence_ids,
                attention_args=attention_args,
            )
        elif backend == AttentionBackend.SDPA:
            attn_output, attn_weights = self._sdpa_attn(
                query_states=query_states,
                key_states=key_states,
                val_states=val_states,
                q_sequence_ids=q_sequence_ids,
                k_sequence_ids=k_sequence_ids,
            )
        else:
            raise AssertionError(f"Unsupported resolved backend: {backend}")

        s_max = self._compute_s_max(query_states, key_states) if output_s_max else None
        return attn_output, attn_weights, s_max

    def _compute_s_max(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
    ) -> list[torch.Tensor]:
        with torch.no_grad():
            query_BHLD = query_states.transpose(1, 2).contiguous()
            key_BHLD = key_states.transpose(1, 2).contiguous()
            key_BHLD = repeat_kv(key_BHLD, self.num_key_value_groups)
            q_norm = torch.linalg.vector_norm(query_BHLD, dim=-1)
            k_norm = torch.linalg.vector_norm(key_BHLD, dim=-1)
            s_max_bound = (q_norm.max(dim=-1).values * k_norm.max(dim=-1).values).max(dim=0).values * self.scale
            return [s_max_bound[h] for h in range(self.num_heads)]

    def _manual_attn(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        val_states: torch.Tensor,
        q_sequence_ids: torch.Tensor,
        k_sequence_ids: torch.Tensor,
        output_s_max: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        query_BHLD = query_states.transpose(1, 2).contiguous()
        key_BHLD = key_states.transpose(1, 2).contiguous()
        val_BHLD = val_states.transpose(1, 2).contiguous()
        key_BHLD = repeat_kv(key_BHLD, self.num_key_value_groups)
        val_BHLD = repeat_kv(val_BHLD, self.num_key_value_groups)
        attention_mask = _build_nonpad_attention_mask(q_sequence_ids, k_sequence_ids)
        attn_weights = torch.matmul(query_BHLD, key_BHLD.transpose(-2, -1)) * self.scale
        attn_weights = attn_weights.masked_fill(attention_mask.logical_not(), float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        context_BHLD = torch.matmul(attn_weights, val_BHLD)
        attn_output = context_BHLD.transpose(1, 2).reshape(query_states.shape[0], query_states.shape[1], self.hidden_size).contiguous()
        s_max = self._compute_s_max(query_states, key_states) if output_s_max else None
        return attn_output, attn_weights, s_max

    def _kernels_flash_attn(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        val_states: torch.Tensor,
        q_sequence_ids: torch.Tensor,
        k_sequence_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, q_len = query_states.shape[0], query_states.shape[1]
        attn_output = kernels_flash_attention_func(
            query_states=query_states,
            key_states=key_states,
            val_states=val_states,
            q_sequence_ids=q_sequence_ids,
            k_sequence_ids=k_sequence_ids,
            causal=False,
        )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        return attn_output, None

    def _flash_attn_package(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        val_states: torch.Tensor,
        q_sequence_ids: torch.Tensor,
        k_sequence_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, q_len = query_states.shape[0], query_states.shape[1]
        attn_output = flash_attn_package_attention_func(
            query_states=query_states,
            key_states=key_states,
            value_states=val_states,
            q_sequence_ids=q_sequence_ids,
            k_sequence_ids=k_sequence_ids,
            causal=False,
        )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        return attn_output, None

    def _flex_attn(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        val_states: torch.Tensor,
        q_sequence_ids: torch.Tensor,
        k_sequence_ids: torch.Tensor,
        attention_args: AttentionArgs | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, q_len = query_states.shape[0], query_states.shape[1]
        block_mask = None
        if attention_args is not None and "block_mask" in attention_args:
            block_mask = attention_args["block_mask"]
        if block_mask is None:
            block_mask = create_nonpad_block_mask(q_sequence_ids, k_sequence_ids)
        outputs = flex_attention_func(query_states, key_states, val_states, block_mask=block_mask)
        outputs = outputs.reshape(bsz, q_len, self.hidden_size).contiguous()
        return outputs, None

    def _sdpa_attn(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        val_states: torch.Tensor,
        q_sequence_ids: torch.Tensor,
        k_sequence_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, q_len = query_states.shape[0], query_states.shape[1]
        attn_output = sdpa_attention_func(
            query_states=query_states,
            key_states=key_states,
            value_states=val_states,
            q_sequence_ids=q_sequence_ids,
            k_sequence_ids=k_sequence_ids,
            num_key_value_groups=self.num_key_value_groups,
        )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        return attn_output, None


class MLP(nn.Module):
    def __init__(self, config: E1Config):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act_fn(self.w1(hidden_states)))


class GLUMLP(nn.Module):
    def __init__(self, config: E1Config):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        hidden_states = self.w2(hidden_states)
        return hidden_states


class FFN(nn.Module):
    def __init__(self, config: E1Config):
        super().__init__()
        mlp_cls = GLUMLP if config.gated_mlp else MLP
        self.mlp = mlp_cls(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)


@dataclass
class E1ModelOutputWithPast(ModelOutput):
    """Base class for model's outputs, with potential hidden states and attentions.

    Attributes:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: DynamicCache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    s_max: tuple[list[torch.Tensor], ...] | None = None


@dataclass
class E1MaskedLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = None
    mlm_loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: DynamicCache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    s_max: tuple[list[torch.Tensor], ...] | None = None


@dataclass
class E1ClassificationOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: DynamicCache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    s_max: tuple[list[torch.Tensor], ...] | None = None


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        if layer_norm is None:
            return torch.nn.functional.rms_norm(
                hidden_states, (self.hidden_size,), self.weight, self.variance_epsilon
            ).to(input_dtype)
        else:
            return layer_norm.rms_norm_fn(
                x=hidden_states,
                weight=self.weight,
                bias=None,  # no bias
                residual=None,
                eps=self.variance_epsilon,
                dropout_p=0.0,  # no dropout by default
                prenorm=False,
                residual_in_fp32=False,
            ).to(input_dtype)


class NormAttentionNorm(nn.Module):
    def __init__(self, config: E1Config, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        within_seq_position_ids: torch.LongTensor,
        global_position_ids: torch.LongTensor,
        sequence_ids: torch.LongTensor,
        attention_args: AttentionArgs | None = None,
        past_key_value: DynamicCache | None = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None, DynamicCache | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, s_max, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            within_seq_position_ids=within_seq_position_ids,
            global_position_ids=global_position_ids,
            sequence_ids=sequence_ids,
            attention_args=attention_args,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return hidden_states, residual, self_attn_weights, s_max, present_key_value


class DecoderLayer(nn.Module):
    def __init__(self, config: E1Config, layer_idx: int):
        super().__init__()
        self.initializer_range = config.initializer_range
        self.hidden_size = config.hidden_size
        self.norm_attn_norm = NormAttentionNorm(config, layer_idx)
        self.ffn = FFN(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        within_seq_position_ids: torch.LongTensor,
        global_position_ids: torch.LongTensor,
        sequence_ids: torch.LongTensor,
        attention_args: AttentionArgs | None = None,
        past_key_value: DynamicCache | None = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None, DynamicCache | None]:
        hidden_states, residual, self_attn_weights, s_max, present_key_value = self.norm_attn_norm(
            hidden_states=hidden_states,
            within_seq_position_ids=within_seq_position_ids,
            global_position_ids=global_position_ids,
            sequence_ids=sequence_ids,
            attention_args=attention_args,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
            use_cache=use_cache,
        )

        # Fully Connected
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, self_attn_weights, s_max, present_key_value


class E1PreTrainedModel(PreTrainedModel):
    config_class = E1Config
    config: E1Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DecoderLayer"]
    _transformer_layer_cls = [DecoderLayer]
    _skip_keys_device_placement = "past_key_values"
    all_tied_weights_keys = {}

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    def _backward_compatibility_gradient_checkpointing(self) -> None:
        if self.supports_gradient_checkpointing and getattr(self.config, "gradient_checkpointing", False):
            self.gradient_checkpointing_enable(dict(use_reentrant=False))

    @property
    def _device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def attn_backend(self) -> str:
        return self.config.attn_backend

    @attn_backend.setter
    def attn_backend(self, backend: str) -> None:
        _validate_attention_backend(backend)
        self.config.attn_backend = backend

    @classmethod
    def is_remote_code(cls) -> bool:
        # Prevent post-load reinitialization of tensors already loaded from checkpoints.
        return True


class FAST_E1_ENCODER(E1PreTrainedModel):
    config: E1Config
    config_class = E1Config
    def __init__(self, config: E1Config, **kwargs):
        E1PreTrainedModel.__init__(self, config, **kwargs)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_seq_id = nn.Embedding(config.max_num_sequences, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = config.gradient_checkpointing
        self.prep_tokens = E1BatchPreparer()
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self.prep_tokens.tokenizer,
            bos_token="<bos>",
            eos_token="<eos>",
            pad_token="<pad>",
            mask_token="?",
        )
        #self.init_weights()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    @torch.inference_mode()
    def _embed(self, sequences: List[str], return_attention_mask: bool = False, **kwargs) -> torch.Tensor:
        batch = self.tokenize_batch(sequences=sequences, device=self._device)
        last_hidden_state = self.forward(**batch, output_hidden_states=False, output_attentions=False).last_hidden_state
        if return_attention_mask:
            attention_mask = (batch['sequence_ids'] != -1).long()
            return last_hidden_state, attention_mask
        else:
            return last_hidden_state

    def tokenize_batch(
        self,
        sequences: list[str | tuple[str, str] | list[str]],
        device: torch.device | None = None,
        non_blocking: bool = False,
    ) -> dict[str, torch.Tensor]:
        target_device = self._device if device is None else device
        non_blocking = non_blocking and target_device.type == "cuda"
        encoded_items: list[dict[str, torch.Tensor]] = []

        for item in sequences:
            parts: list[str]
            if isinstance(item, str):
                split_parts = item.split(",")
                if len(split_parts) == 1:
                    parts = split_parts
                else:
                    assert len(split_parts) == 2, (
                        f"Expected at most two sequences per batch item, got {len(split_parts)} in '{item}'."
                    )
                    parts = split_parts
            else:
                assert len(item) > 0, "Each non-string batch item must contain at least one sequence."
                assert len(item) <= 2, f"Expected at most two sequences per batch item, got {len(item)}."
                parts = list(item)

            single_encodings = [self.prep_tokens.prepare_singleseq(sequence) for sequence in parts]
            input_ids = torch.cat([encoding["input_ids"] for encoding in single_encodings], dim=0).long()
            within_seq_position_ids = torch.cat([encoding["position_ids"] for encoding in single_encodings], dim=0).long()
            global_position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)

            seq_ids: list[torch.Tensor] = []
            for sequence_index in range(len(single_encodings)):
                token_count = single_encodings[sequence_index]["input_ids"].shape[0]
                seq_ids.append(torch.full((token_count,), sequence_index, dtype=torch.long))
            sequence_ids = torch.cat(seq_ids, dim=0)
            token_type_ids = sequence_ids.clone()
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            labels = input_ids.clone()

            encoded_items.append(
                {
                    "input_ids": input_ids,
                    "within_seq_position_ids": within_seq_position_ids,
                    "global_position_ids": global_position_ids,
                    "sequence_ids": sequence_ids,
                    "token_type_ids": token_type_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

        padded_batch: dict[str, torch.Tensor] = {}
        pad_map = {
            "input_ids": self.padding_idx,
            "within_seq_position_ids": -1,
            "global_position_ids": -1,
            "sequence_ids": -1,
            "token_type_ids": 0,
            "attention_mask": 0,
            "labels": self.padding_idx,
        }
        for key in pad_map:
            padded_batch[key] = pad_sequence(
                [encoding[key] for encoding in encoded_items],
                batch_first=True,
                padding_value=pad_map[key],
            ).to(device=target_device, dtype=torch.long, non_blocking=non_blocking)

        return padded_batch

    def _build_position_ids_from_sequence_ids(
        self,
        sequence_ids: torch.LongTensor,
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        batch_size, seq_length = sequence_ids.shape
        within_seq_position_ids = torch.full_like(sequence_ids, -1, dtype=torch.long)
        global_position_ids = torch.full_like(sequence_ids, -1, dtype=torch.long)
        for batch_idx in range(batch_size):
            sequence_counters: dict[int, int] = {}
            global_counter = 0
            for token_idx in range(seq_length):
                sequence_id = int(sequence_ids[batch_idx, token_idx].item())
                if sequence_id == -1:
                    continue
                if sequence_id not in sequence_counters:
                    sequence_counters[sequence_id] = 0
                within_seq_position_ids[batch_idx, token_idx] = sequence_counters[sequence_id]
                sequence_counters[sequence_id] = sequence_counters[sequence_id] + 1
                global_position_ids[batch_idx, token_idx] = global_counter
                global_counter = global_counter + 1
        return within_seq_position_ids, global_position_ids

    def _prepare_forward_ids(
        self,
        input_ids: torch.LongTensor,
        within_seq_position_ids: torch.LongTensor | None,
        global_position_ids: torch.LongTensor | None,
        sequence_ids: torch.LongTensor | None,
        attention_mask: torch.LongTensor | None,
        token_type_ids: torch.LongTensor | None,
    ) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        if sequence_ids is None:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
            sequence_ids = torch.where(
                attention_mask.bool(),
                token_type_ids.long(),
                torch.full_like(token_type_ids, -1, dtype=torch.long),
            )
        else:
            sequence_ids = sequence_ids.long()

        assert sequence_ids.max().item() < self.config.max_num_sequences, (
            f"Found sequence id {sequence_ids.max().item()} but max_num_sequences={self.config.max_num_sequences}."
        )
        assert sequence_ids.min().item() >= -1, "sequence_ids must be >= -1."

        if within_seq_position_ids is None or global_position_ids is None:
            derived_within_seq_position_ids, derived_global_position_ids = self._build_position_ids_from_sequence_ids(sequence_ids)
            if within_seq_position_ids is None:
                within_seq_position_ids = derived_within_seq_position_ids
            if global_position_ids is None:
                global_position_ids = derived_global_position_ids
        else:
            within_seq_position_ids = within_seq_position_ids.long()
            global_position_ids = global_position_ids.long()

        return within_seq_position_ids.long(), global_position_ids.long(), sequence_ids.long()

    # Ignore copy
    def forward(
        self,
        input_ids: torch.LongTensor,
        within_seq_position_ids: torch.LongTensor | None = None,
        global_position_ids: torch.LongTensor | None = None,
        sequence_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        past_key_values: DynamicCache | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_s_max: bool = False,
        **kwargs
    ) -> E1ModelOutputWithPast:
        """
        Args:
            input_ids: (batch_size, seq_length)
            within_seq_position_ids: (batch_size, seq_length)
                This tensor contains the position of each residue within the sequence itself.
                For example, if the input is ["<bos>1ABC2<eos><bos>1DEF2<eos>", "<bos>1GH2<eos><bos>1JKL2<eos><pad>"],
                the tensor would be [[0,1,2,3,4,5,6,0,1,2,3,4,5,6], [0,1,2,3,4,5,0,1,2,3,4,5,6,-1]]
            global_position_ids: (batch_size, seq_length)
                This tensor contains the position of each residue within the global sequence.
                For example, if the input is ["<bos>1ABC2<eos><bos>1DEF2<eos>", "<bos>1GH2<eos><bos>1JKL2<eos>"],
                the tensor would be [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1]]
            sequence_ids: (batch_size, seq_length)
                This tensor contains the sequence id of each residue.
                For example, if the input is ["<bos>1ABC2<eos><bos>1DEF2<eos>", "<bos>1GH2<eos><bos>1JKL2<eos>"],
                the tensor would be [[0,0,0,0,0,0,0,1,1,1,1,1,1,1], [0,0,0,0,0,0,1,1,1,1,1,1,1,-1]]
            past_key_values: DynamicCache
            use_cache: bool
            output_attentions: bool
            output_hidden_states: bool

        Returns:
            E1ModelOutputWithPast: Model Outputs
        """
        batch_size, seq_length = input_ids.shape

        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        elif not use_cache:
            # To avoid weirdness with gradient checkpointing: https://github.com/huggingface/transformers/issues/28499
            past_key_values = None

        within_seq_position_ids, global_position_ids, sequence_ids = self._prepare_forward_ids(
            input_ids=input_ids,
            within_seq_position_ids=within_seq_position_ids,
            global_position_ids=global_position_ids,
            sequence_ids=sequence_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        global_position_ids = global_position_ids.view(-1, seq_length).long()
        within_seq_position_ids = within_seq_position_ids.view(-1, seq_length).long()
        sequence_ids = sequence_ids.view(-1, seq_length).long()

        max_position_id = torch.max(within_seq_position_ids).item()
        min_position_id = torch.min(within_seq_position_ids).item()
        assert max_position_id < self.config.max_num_positions_within_seq and min_position_id >= -1, (
            f"Position ids must be in the range [-1, {self.config.max_num_positions_within_seq}); got max {max_position_id} and min {min_position_id}"
        )

        inputs_embeds = self.embed_tokens(input_ids)
        # -1 is used to indicate padding tokens, so we need to clamp the sequence ids to 0
        inputs_embeds = inputs_embeds + self.embed_seq_id(sequence_ids.clamp(min=0))

        # In case we need to do any manual typecasting
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        else:
            target_dtype = self.layers[0].norm_attn_norm.self_attn.q_proj.weight.dtype
        hidden_states = inputs_embeds.to(target_dtype)

        attention_args: AttentionArgs | None = None

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_s_max = () if output_s_max else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)  # type: ignore[operator]

            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    within_seq_position_ids,
                    global_position_ids,
                    sequence_ids,
                    attention_args,
                    past_key_values,
                    output_attentions,
                    output_s_max,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    within_seq_position_ids=within_seq_position_ids,
                    global_position_ids=global_position_ids,
                    sequence_ids=sequence_ids,
                    attention_args=attention_args,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_s_max=output_s_max,
                    use_cache=use_cache,
                )

            hidden_states, self_attn_weights, layer_s_max, present_key_value = layer_outputs

            if use_cache:
                # NOTE: it's necessary to re-assign past_key_values because FSDP2
                # passes certain arguments by value, not by reference.
                # See https://github.com/huggingface/transformers/issues/38190#issuecomment-2914016168
                next_decoder_cache = past_key_values = present_key_value

            if output_attentions:
                all_self_attns += (self_attn_weights,)  # type: ignore[operator]
            if output_s_max:
                all_s_max += (layer_s_max,)  # type: ignore[operator]

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)  # type: ignore[operator]

        next_cache = next_decoder_cache if use_cache else None

        return E1ModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            s_max=all_s_max,
        )


class E1Model(E1PreTrainedModel):
    config: E1Config
    config_class = E1Config

    def __init__(self, config: E1Config, **kwargs):
        E1PreTrainedModel.__init__(self, config, **kwargs)
        self.model: FAST_E1_ENCODER = FAST_E1_ENCODER(config, **kwargs)
        #self.init_weights()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.set_input_embeddings(value)

    @torch.inference_mode()
    def _embed(self, sequences: List[str], return_attention_mask: bool = False, **kwargs) -> torch.Tensor:
        return self.model._embed(sequences, return_attention_mask=return_attention_mask, **kwargs)

    def tokenize_batch(
        self,
        sequences: list[str | tuple[str, str] | list[str]],
        device: torch.device | None = None,
        non_blocking: bool = False,
    ) -> dict[str, torch.Tensor]:
        return self.model.tokenize_batch(sequences=sequences, device=device, non_blocking=non_blocking)

    def forward(
        self,
        input_ids: torch.LongTensor,
        within_seq_position_ids: torch.LongTensor | None = None,
        global_position_ids: torch.LongTensor | None = None,
        sequence_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        past_key_values: DynamicCache | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_s_max: bool = False,
        **kwargs,
    ) -> E1ModelOutputWithPast:
        return self.model(
            input_ids=input_ids,
            within_seq_position_ids=within_seq_position_ids,
            global_position_ids=global_position_ids,
            sequence_ids=sequence_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
            **kwargs,
        )


class E1ForMaskedLM(E1PreTrainedModel):
    config: E1Config
    config_class = E1Config
    def __init__(self, config: E1Config, **kwargs):
        E1PreTrainedModel.__init__(self, config, **kwargs)
        self.model: FAST_E1_ENCODER = FAST_E1_ENCODER(config, **kwargs)
        self.vocab_size = config.vocab_size
        self.mlm_head = torch.nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size, bias=True),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps),
            nn.Linear(config.hidden_size, config.vocab_size, bias=True),
        )
        self.gradient_checkpointing = config.gradient_checkpointing
        self.prep_tokens = self.model.prep_tokens
        self.tokenizer = self.model.tokenizer
        #self.init_weights()

    @property
    def device_mesh(self) -> torch.distributed.device_mesh.DeviceMesh:
        return self.model.device_mesh

    @torch.inference_mode()
    def _embed(self, sequences: List[str], return_attention_mask: bool = False, **kwargs) -> torch.Tensor:
        batch = self.model.tokenize_batch(sequences=sequences, device=self._device)
        last_hidden_state = self.model(**batch, output_hidden_states=False, output_attentions=False).last_hidden_state
        if return_attention_mask:
            attention_mask = (batch['sequence_ids'] != -1).long()
            return last_hidden_state, attention_mask
        else:
            return last_hidden_state

    def tokenize_batch(
        self,
        sequences: list[str | tuple[str, str] | list[str]],
        device: torch.device | None = None,
        non_blocking: bool = False,
    ) -> dict[str, torch.Tensor]:
        return self.model.tokenize_batch(sequences=sequences, device=device, non_blocking=non_blocking)

    def forward(
        self,
        input_ids: torch.LongTensor,
        within_seq_position_ids: torch.LongTensor | None = None,
        global_position_ids: torch.LongTensor | None = None,
        sequence_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        past_key_values: DynamicCache | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_s_max: bool = False,
        **kwargs,
    ) -> E1MaskedLMOutputWithPast:
        """
        Args:
            input_ids: (batch_size, seq_length)
            within_seq_position_ids: (batch_size, seq_length)
                This tensor contains the position of each residue within the sequence itself.
                For example, if the input is ["<bos>1ABC2<eos><bos>1DEF2<eos>", "<bos>1GH2<eos><bos>1JKL2<eos><pad>"],
                the tensor would be [[0,1,2,3,4,5,6,0,1,2,3,4,5,6], [0,1,2,3,4,5,0,1,2,3,4,5,6,-1]]
            global_position_ids: (batch_size, seq_length)
                This tensor contains the position of each residue within the global sequence.
                For example, if the input is ["<bos>1ABC2<eos><bos>1DEF2<eos>", "<bos>1GH2<eos><bos>1JKL2<eos>"],
                the tensor would be [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1]]
            sequence_ids: (batch_size, seq_length)
                This tensor contains the sequence id of each residue.
                For example, if the input is ["<bos>1ABC2<eos><bos>1DEF2<eos>", "<bos>1GH2<eos><bos>1JKL2<eos>"],
                the tensor would be [[0,0,0,0,0,0,0,1,1,1,1,1,1,1], [0,0,0,0,0,0,1,1,1,1,1,1,1,-1]]
            labels: (batch_size, seq_length)
            past_key_values: DynamicCache
            use_cache: bool
            output_attentions: bool
            output_hidden_states: bool

        Returns:
            E1MaskedLMOutputWithPast: Model Outputs
        """
        outputs: E1ModelOutputWithPast = self.model(
            input_ids=input_ids,
            within_seq_position_ids=within_seq_position_ids,
            global_position_ids=global_position_ids,
            sequence_ids=sequence_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
        )

        last_hidden_state = outputs.last_hidden_state
        loss = None

        # Compute masked language modeling loss
        mlm_logits = self.mlm_head(last_hidden_state).float()
        mlm_loss = 0.0
        if labels is not None:
            mlm_logits_flat = mlm_logits.contiguous().view(-1, self.config.vocab_size)
            mlm_labels_flat = labels.to(mlm_logits_flat.device).contiguous().view(-1)
            mlm_loss = F.cross_entropy(mlm_logits_flat, mlm_labels_flat, reduction="none")
            mask = mlm_labels_flat != self.model.padding_idx
            n_mlm = mask.sum()
            mlm_loss = (mlm_loss * mask.to(mlm_loss)).sum() / (1 if n_mlm == 0 else n_mlm)
            loss = 0.0
            loss += mlm_loss

        return E1MaskedLMOutputWithPast(
            loss=loss,
            mlm_loss=mlm_loss,
            logits=mlm_logits,
            last_hidden_state=last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            s_max=outputs.s_max,
        )


class E1ForSequenceClassification(E1PreTrainedModel):
    config: E1Config
    config_class = E1Config
    def __init__(self, config: E1Config, **kwargs):
        E1PreTrainedModel.__init__(self, config, **kwargs)
        self.model: FAST_E1_ENCODER = FAST_E1_ENCODER(config, **kwargs)
        self.vocab_size = config.vocab_size
        self.num_labels = config.num_labels
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size * 4),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size * 4),
            nn.Linear(config.hidden_size * 4, config.num_labels),
        )
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.gradient_checkpointing = config.gradient_checkpointing
        self.prep_tokens = E1BatchPreparer()

        if 'pooling_types' in kwargs and isinstance(kwargs['pooling_types'], List[str]) and len(kwargs['pooling_types']) > 0:
            pooling_types = kwargs['pooling_types']
        else:
            pooling_types = ['mean', 'var']
        self.pooler = Pooler(pooling_types)
        #self.init_weights()

    @property
    def device_mesh(self) -> torch.distributed.device_mesh.DeviceMesh:
        return self.model.device_mesh

    @torch.inference_mode()
    def _embed(self, sequences: List[str], return_attention_mask: bool = False, **kwargs) -> torch.Tensor:
        batch = self.model.tokenize_batch(sequences=sequences, device=self._device)
        last_hidden_state = self.model(**batch, output_hidden_states=False, output_attentions=False).last_hidden_state
        if return_attention_mask:
            attention_mask = (batch['sequence_ids'] != -1).long()
            return last_hidden_state, attention_mask
        else:
            return last_hidden_state

    def forward(
        self,
        input_ids: torch.LongTensor,
        within_seq_position_ids: torch.LongTensor | None = None,
        global_position_ids: torch.LongTensor | None = None,
        sequence_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        past_key_values: DynamicCache | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_s_max: bool = False,
        **kwargs,
    ) -> E1ClassificationOutputWithPast:
        outputs: E1ModelOutputWithPast = self.model(
            input_ids=input_ids,
            within_seq_position_ids=within_seq_position_ids,
            global_position_ids=global_position_ids,
            sequence_ids=sequence_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
        )

        if sequence_ids is None:
            assert attention_mask is not None, "attention_mask is required when sequence_ids is not provided."
            seq_attention_mask = attention_mask.long()
        else:
            seq_attention_mask = (sequence_ids != -1).long()
        x = outputs.last_hidden_state
        features = self.pooler(x, seq_attention_mask)
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

        return E1ClassificationOutputWithPast(
            loss=loss,
            logits=logits,
            last_hidden_state=x,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            s_max=outputs.s_max,
        )


class E1ForTokenClassification(E1PreTrainedModel):
    config: E1Config
    config_class = E1Config
    def __init__(self, config: E1Config, **kwargs):
        E1PreTrainedModel.__init__(self, config, **kwargs)
        self.model: FAST_E1_ENCODER = FAST_E1_ENCODER(config, **kwargs)
        self.vocab_size = config.vocab_size
        self.num_labels = config.num_labels
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size * 4),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size * 4),
            nn.Linear(config.hidden_size * 4, config.num_labels),
        )
        self.loss_fct = nn.CrossEntropyLoss()
        self.gradient_checkpointing = config.gradient_checkpointing
        self.prep_tokens = E1BatchPreparer()
        #self.init_weights()

    @property
    def device_mesh(self) -> torch.distributed.device_mesh.DeviceMesh:
        return self.model.device_mesh

    @torch.inference_mode()
    def _embed(self, sequences: List[str], return_attention_mask: bool = False, **kwargs) -> torch.Tensor:
        batch = self.model.tokenize_batch(sequences=sequences, device=self._device)
        last_hidden_state = self.model(**batch, output_hidden_states=False, output_attentions=False).last_hidden_state
        if return_attention_mask:
            attention_mask = (batch['sequence_ids'] != -1).long()
            return last_hidden_state, attention_mask
        else:
            return last_hidden_state

    def forward(
        self,
        input_ids: torch.LongTensor,
        within_seq_position_ids: torch.LongTensor | None = None,
        global_position_ids: torch.LongTensor | None = None,
        sequence_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        past_key_values: DynamicCache | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_s_max: bool = False,
        **kwargs,
    ) -> E1ClassificationOutputWithPast:
        outputs: E1ModelOutputWithPast = self.model(
            input_ids=input_ids,
            within_seq_position_ids=within_seq_position_ids,
            global_position_ids=global_position_ids,
            sequence_ids=sequence_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
        )

        x = outputs.last_hidden_state
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return E1ClassificationOutputWithPast(
            loss=loss,
            logits=logits,
            last_hidden_state=x,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            s_max=outputs.s_max,
        )


