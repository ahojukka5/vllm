# SPDX-License-Identifier: Apache-2.0
"""Fused Triton SituGLU activation for ROCm builds without torch.ops._C.

Semantics match SituAndMul.forward_native exactly (fp32 compute, bf16 i/o):
    out = beta*tanh(gate/beta) * sigmoid(gate) * up_clip
    up_clip = linear_beta*tanh(up/linear_beta) when linear_beta > 0 else up

One kernel launch, one read of `input`, one write of `output` (the
PyTorch fallback does 6-7 launches with scratch round-trips).
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _situ_and_mul_kernel(
    out_ptr,
    in_ptr,
    beta,
    linear_beta,
    M,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    cmask = col < D

    gate = tl.load(in_ptr + row * 2 * D + col, mask=cmask, other=0.0).to(tl.float32)
    up = tl.load(in_ptr + row * 2 * D + D + col, mask=cmask, other=0.0).to(tl.float32)

    g = beta * libdevice.tanh(gate / beta) * tl.sigmoid(gate)
    if linear_beta > 0:
        up = linear_beta * libdevice.tanh(up / linear_beta)

    tl.store(out_ptr + row * D + col, (g * up).to(out_ptr.dtype.element_ty),
             mask=cmask)


def situ_and_mul_triton(
    output: torch.Tensor,
    input: torch.Tensor,
    beta: float,
    linear_beta: float,
) -> None:
    D = output.shape[-1]
    in2d = input.reshape(-1, 2 * D)
    out2d = output.reshape(-1, D)
    assert in2d.is_contiguous() and out2d.is_contiguous()
    M = in2d.shape[0]
    BLOCK = 1024
    grid = (M, triton.cdiv(D, BLOCK))
    _situ_and_mul_kernel[grid](
        out2d, in2d, float(beta), float(linear_beta), M, D,
        BLOCK=BLOCK, num_warps=4,
    )
