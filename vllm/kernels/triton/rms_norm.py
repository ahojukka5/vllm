# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton rms_norm / fused_add_rms_norm IR implementations.

For platforms (e.g. ROCm builds without the _C layernorm kernels and without
aiter) where the only alternative is the multi-kernel PyTorch-native path.
Single kernel launch per call, fp32 accumulation, matching the native
implementation's rounding sequence (variance in fp32, x rounded to the
weight dtype before the weight multiply).
"""
import torch
import triton
import triton.language as tl
from torch import Tensor

from vllm import ir
from vllm.platforms import current_platform

TRITON_SUPPORTED = current_platform.is_cuda_alike()

_triton_no_var_size = lambda x, weight, epsilon, variance_size=None: (
    variance_size is None
    and x.is_cuda
    and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    and (weight is None or weight.dtype == x.dtype)
    and x.is_contiguous()
)

_triton_add_no_var_size = (
    lambda x, x_residual, weight, epsilon, variance_size=None: (
        variance_size is None
        and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and (weight is None or weight.dtype == x.dtype)
        and x.is_contiguous()
        and x_residual.is_contiguous()
    )
)


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    eps,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x) / N
    r = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Match native rounding: x*rsqrt rounded to the weight dtype, then
    # multiplied by the weight (product rounds once more to the out dtype).
    y = (x * r).to(w_ptr.dtype.element_ty).to(tl.float32) * w
    tl.store(y_ptr + row * N + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _fused_add_rms_norm_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    y_ptr,
    eps,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # In-place: y aliases x, res_ptr holds the residual in and out.
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(res_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    s = x + r
    tl.store(res_ptr + row * N + cols, s.to(res_ptr.dtype.element_ty), mask=mask)
    var = tl.sum(s * s) / N
    rr = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (s * rr).to(w_ptr.dtype.element_ty).to(tl.float32) * w
    tl.store(y_ptr + row * N + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


_ones_cache: dict[tuple[int, torch.dtype, torch.device], Tensor] = {}


def _ones(n: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    key = (n, dtype, device)
    w = _ones_cache.get(key)
    if w is None:
        w = torch.ones(n, dtype=dtype, device=device)
        _ones_cache[key] = w
    return w


def _num_warps(block: int) -> int:
    return max(1, min(8, block // 1024))


@ir.ops.rms_norm.register_impl(
    "triton", supports_args=_triton_no_var_size, supported=TRITON_SUPPORTED
)
def rms_norm(
    x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None
) -> Tensor:
    assert variance_size is None
    shape = x.shape
    n = shape[-1]
    x2 = x.reshape(-1, n)
    if weight is None:
        weight = _ones(n, x.dtype, x.device)
    y = torch.empty_like(x2)
    block = triton.next_power_of_2(n)
    _rms_norm_kernel[(x2.shape[0],)](
        x2, weight, y, epsilon, n, BLOCK=block, num_warps=_num_warps(block)
    )
    return y.reshape(shape)


@ir.ops.fused_add_rms_norm.register_impl(
    "triton",
    supports_args=_triton_add_no_var_size,
    supported=TRITON_SUPPORTED,
    inplace=True,
)
def fused_add_rms_norm(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    assert variance_size is None
    shape = x.shape
    n = shape[-1]
    x2 = x.reshape(-1, n)
    r2 = x_residual.reshape(-1, n)
    if weight is None:
        weight = _ones(n, x.dtype, x.device)
    block = triton.next_power_of_2(n)
    _fused_add_rms_norm_kernel[(x2.shape[0],)](
        x2, r2, weight, x2, epsilon, n, BLOCK=block, num_warps=_num_warps(block)
    )
    return x, x_residual
