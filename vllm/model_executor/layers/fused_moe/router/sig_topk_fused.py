# SPDX-License-Identifier: Apache-2.0
"""Fused sigmoid+bias+topk+renorm router for Kimi K3 on ROCm.

The stock fast path (vllm#26779) uses torch.topk, which on ROCm lowers to
sbtopk::gatherTopK + radixSortKVInPlace (~72 us/layer at E=1024, ~7 ms/step
at 92 layers). This kernel does the whole routing decision in registers
instead: one program per token row, K=16 over E<=1024 experts.

Top-K runs on int64 keys packing (orderable_score, inverted_index) so ties
resolve to the lower expert id like torch.topk. The orderable-int trick
(u ^ (arith_sign | 0x80000000)) is monotone only as UNSIGNED, so the int32
must be ZERO-extended into the int64 key (o & 0xffffffff), not sign-extended.
Masked pad lanes hold -inf, which packs to the lowest key and can never win.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _sig_topk_kernel(
    gating_ptr,          # [M, E] float32 raw logits
    bias_ptr,            # [E] float32 score correction bias
    out_w_ptr,           # [M, K] float32
    out_i_ptr,           # [M, K] int32
    E: tl.constexpr,
    E_PAD: tl.constexpr,
    K: tl.constexpr,
    RENORM: tl.constexpr,
    scale,               # routed_scaling_factor
):
    row = tl.program_id(0)
    offs = tl.arange(0, E_PAD)
    mask = offs < E
    logits = tl.load(gating_ptr + row * E + offs, mask=mask,
                     other=float("-inf"))
    scores = tl.sigmoid(logits)
    bias = tl.load(bias_ptr + offs, mask=mask, other=0.0)
    biased = scores + bias

    # orderable map (unsigned-monotone): zero-extend into int64, then flip
    # the int64 sign bit so signed comparison matches the unsigned order
    u = biased.to(tl.int32, bitcast=True)
    o = u ^ ((u >> 31) | (-2147483648))
    key = (((o.to(tl.int64) & 0xffffffff) << 32)
           | (E_PAD - 1 - offs).to(tl.int64)) ^ (-9223372036854775808)
    top = tl.topk(key, K)                      # [K] int64, descending
    top_ids = (E_PAD - 1) - (top & 0xffffffff).to(tl.int32)
    top_w = tl.gather(scores, top_ids, 0)

    if RENORM:
        top_w = top_w / tl.sum(top_w, axis=0)
    if scale != 1.0:
        top_w = top_w * scale

    tl.store(out_w_ptr + row * K + tl.arange(0, K), top_w)
    tl.store(out_i_ptr + row * K + tl.arange(0, K), top_ids)


def _pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def sig_topk_fused_impl(
    gating_output: torch.Tensor,      # [M, E] float32
    e_score_correction_bias: torch.Tensor,  # [E] float32
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, E = gating_output.shape
    E_PAD = _pow2(E)
    out_w = torch.empty(M, topk, dtype=torch.float32,
                        device=gating_output.device)
    out_i = torch.empty(M, topk, dtype=torch.int32,
                        device=gating_output.device)
    _sig_topk_kernel[(M,)](
        gating_output, e_score_correction_bias, out_w, out_i,
        E=E, E_PAD=E_PAD, K=topk,
        RENORM=renormalize, scale=routed_scaling_factor,
        num_warps=4,
    )
    return out_w, out_i


def sig_topk_fused_fake(
    gating_output: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    M = gating_output.shape[0]
    return (
        torch.empty(M, topk, dtype=torch.float32,
                    device=gating_output.device),
        torch.empty(M, topk, dtype=torch.int32,
                    device=gating_output.device),
    )


direct_register_custom_op(
    op_name="sig_topk_fused",
    op_func=sig_topk_fused_impl,
    fake_impl=sig_topk_fused_fake,
)
