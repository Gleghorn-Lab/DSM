#! /usr/bin/env python3
import argparse
import torch


def reference_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, causal: bool = False) -> torch.Tensor:
    query, key, value = (tensor.transpose(1, 2).contiguous() for tensor in (query, key, value))
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=causal)
    return out.transpose(1, 2).contiguous()


def varlen_reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    batch_size = int(cu_seqlens_q.shape[0] - 1)
    out = torch.zeros((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=q.dtype)
    for batch_index in range(batch_size):
        q_start = int(cu_seqlens_q[batch_index].item())
        q_end = int(cu_seqlens_q[batch_index + 1].item())
        k_start = int(cu_seqlens_k[batch_index].item())
        k_end = int(cu_seqlens_k[batch_index + 1].item())
        q_slice = q[q_start:q_end].unsqueeze(0)
        k_slice = k[k_start:k_end].unsqueeze(0)
        v_slice = v[k_start:k_end].unsqueeze(0)
        out[q_start:q_end] = reference_attention(q_slice, k_slice, v_slice, causal=causal).squeeze(0)
    return out


def call_flash_attn_func(kernel, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    if hasattr(kernel, "fwd"):
        return kernel.fwd(q=q, k=k, v=v, is_causal=causal)[0]
    if hasattr(kernel, "flash_attn_func"):
        try:
            result = kernel.flash_attn_func(q=q, k=k, v=v, causal=causal)
        except TypeError:
            result = kernel.flash_attn_func(q, k, v, 0.0, None, causal)
        if isinstance(result, tuple):
            return result[0]
        return result
    raise AssertionError("Kernel missing both fwd and flash_attn_func.")


def call_flash_attn_varlen_func(
    kernel,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_q: int,
    max_k: int,
) -> torch.Tensor:
    if hasattr(kernel, "varlen_fwd"):
        return kernel.varlen_fwd(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            is_causal=False,
        )[0]
    if hasattr(kernel, "flash_attn_varlen_func"):
        try:
            result = kernel.flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=max_q,
                max_seqlen_k=max_k,
                causal=False,
            )
        except TypeError:
            result = kernel.flash_attn_varlen_func(
                q,
                k,
                v,
                cu_q,
                cu_k,
                max_q,
                max_k,
                0.0,
                None,
                False,
            )
        if isinstance(result, tuple):
            return result[0]
        return result
    raise AssertionError("Kernel missing both varlen_fwd and flash_attn_varlen_func.")


def run_smoke_test(kernel_id: str, atol: float, rtol: float) -> bool:
    from kernels import get_kernel

    print(f"Loading kernel: {kernel_id}")
    kernel = get_kernel(kernel_id)
    print(f"Loaded kernel object: {kernel}")

    device = torch.device("cuda")
    dtype = torch.float16

    torch.manual_seed(42)
    batch_size, seq_len, num_heads, head_dim = 2, 64, 4, 32
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)

    out_ref = reference_attention(q, k, v, causal=False)
    out_kernel = call_flash_attn_func(kernel, q, k, v, causal=False)
    standard_ok = torch.allclose(out_kernel, out_ref, atol=atol, rtol=rtol)
    print(f"Standard attention close: {standard_ok}")

    out_ref_causal = reference_attention(q, k, v, causal=True)
    out_kernel_causal = call_flash_attn_func(kernel, q, k, v, causal=True)
    causal_ok = torch.allclose(out_kernel_causal, out_ref_causal, atol=atol, rtol=rtol)
    print(f"Causal attention close: {causal_ok}")

    q_var = torch.randn(10, num_heads, head_dim, device=device, dtype=dtype)
    k_var = torch.randn(12, num_heads, head_dim, device=device, dtype=dtype)
    v_var = torch.randn(12, num_heads, head_dim, device=device, dtype=dtype)
    cu_q = torch.tensor([0, 3, 7, 10], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, 4, 9, 12], device=device, dtype=torch.int32)

    out_var_ref = varlen_reference_attention(q_var, k_var, v_var, cu_q, cu_k, causal=False)
    out_var_kernel = call_flash_attn_varlen_func(kernel, q_var, k_var, v_var, cu_q, cu_k, max_q=4, max_k=5)
    varlen_ok = torch.allclose(out_var_kernel, out_var_ref, atol=atol, rtol=rtol)
    print(f"Varlen attention close: {varlen_ok}")

    return bool(standard_ok and causal_ok and varlen_ok)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for kernels flash-attn3/2 against SDPA.")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for output comparison.")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for output comparison.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA must be available to run flash-attention smoke tests."

    kernel_ids = ["kernels-community/flash-attn3", "kernels-community/flash-attn2"]
    last_error = None
    for kernel_id in kernel_ids:
        try:
            test_ok = run_smoke_test(kernel_id=kernel_id, atol=args.atol, rtol=args.rtol)
            if test_ok:
                print(f"Flash attention smoke test PASSED with {kernel_id}")
                return
            print(f"Flash attention smoke test FAILED numerical checks for {kernel_id}")
        except Exception as exc:
            last_error = exc
            print(f"Flash attention smoke test failed for {kernel_id}: {exc}")

    if last_error is not None:
        raise RuntimeError(f"All kernels flash-attention tests failed. Last error: {last_error}") from last_error
    raise RuntimeError("All kernels flash-attention tests failed.")


if __name__ == "__main__":
    main()
