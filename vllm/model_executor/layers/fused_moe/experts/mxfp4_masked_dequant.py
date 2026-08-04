# SPDX-License-Identifier: Apache-2.0
"""Cudagraph-safe masked MXFP4 dequantization for MoE experts.

The emulation MoE path dequantizes expert weights on every forward pass.
At low batch sizes only a few local experts are touched per step, but the
touched set is data-dependent — and D2H syncs or data-dependent launch
counts are illegal during cudagraph capture. This module implements the
selective dequant as a single Triton kernel launch whose grid covers all
experts: each program checks a GPU-resident per-expert mask and exits
immediately for untouched experts. The launch shape is static, so the
whole thing is capture-safe, and untouched weight slices are never read
by the downstream grouped GEMM (it launches tiles only for experts that
appear in `expert_ids`).

Numerics match the OCP MXFP4 spec (and quark's dq_mxfp4, bit-exact):
packed uint8 -> two e2m1 nibbles; value = fp4(nib) * 2**(e8m0 - 127).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _mxfp4_dequant_masked_kernel(
    w_ptr,  # uint8 [E, N, K//2] packed nibbles
    s_ptr,  # uint8 [E, N, K//32] e8m0 scales
    mask_ptr,  # int32 [E] 1 = dequantize this expert, 0 = skip
    out_ptr,  # bf16/fp16 [E, N, K]
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    e = tl.program_id(2)
    m = tl.load(mask_ptr + e)
    if m == 0:
        return

    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    cols = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    rmask = rows < N
    cmask = cols < K
    mask2d = rmask[:, None] & cmask[None, :]

    KB2: tl.constexpr = K // 2
    KS: tl.constexpr = K // 32

    byte_cols = cols // 2
    nib_hi = (cols % 2) == 1
    w_off = e * N * KB2 + rows[:, None] * KB2 + byte_cols[None, :]
    byte = tl.load(w_ptr + w_off, mask=mask2d, other=0).to(tl.int32)
    nib = tl.where(nib_hi[None, :], (byte >> 4) & 0xF, byte & 0xF)

    # e2m1: sign | 2 exp bits (bias 1) | 1 mantissa bit
    sign = (nib >> 3) & 1
    exp = (nib >> 1) & 0x3
    mant = (nib & 1).to(tl.float32)
    mag = tl.where(exp == 0, 0.5 * mant,
                   (1.0 + 0.5 * mant) * tl.exp2((exp - 1).to(tl.float32)))
    fp4 = tl.where(sign == 1, -mag, mag)

    s_off = e * N * KS + rows[:, None] * KS + (cols // 32)[None, :]
    scale = tl.load(s_ptr + s_off, mask=mask2d, other=0).to(tl.int32)
    factor = tl.exp2((scale - 127).to(tl.float32))

    val = fp4 * factor
    o_off = e * N * K + rows[:, None] * K + cols[None, :]
    tl.store(out_ptr + o_off, val.to(out_ptr.dtype.element_ty), mask=mask2d)


def mxfp4_dequant_masked(
    w: torch.Tensor,
    scale: torch.Tensor,
    expert_mask: torch.Tensor,
    dtype: torch.dtype,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequantize packed MXFP4 experts, skipping experts with mask == 0.

    w:     uint8 [E, N, K//2] (contiguous)
    scale: uint8 [E, N, K//32] e8m0 (contiguous)
    expert_mask: int32 [E] on the same device (1 = materialize, 0 = skip)
    out:   optional preallocated [E, N, K] in `dtype`. Untouched experts'
           contents are left as-is (garbage/stale) by contract.
    """
    E, N, KB2 = w.shape
    K = KB2 * 2
    assert w.dtype == torch.uint8 and scale.dtype == torch.uint8
    assert w.is_contiguous() and scale.is_contiguous()
    assert scale.shape == (E, N, K // 32)
    assert expert_mask.shape == (E,)

    if out is None:
        out = torch.empty((E, N, K), device=w.device, dtype=dtype)
    else:
        assert out.shape == (E, N, K) and out.dtype == dtype
        assert out.is_contiguous()

    BLOCK_N, BLOCK_K = 64, 256
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K), E)
    _mxfp4_dequant_masked_kernel[grid](
        w, scale, expert_mask, out, N, K,
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return out


def moe_touched_expert_mask(
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
    num_local_experts: int,
) -> torch.Tensor:
    """GPU-resident per-local-expert "receives >=1 token" mask (int32 [E]).

    Capture-safe: no D2H sync, no dynamic shapes, and no torch.bincount
    (its HIP implementation is illegal during stream capture). topk_ids
    [T, k] global ids; expert_map [G] maps global -> local or -1.
    """
    local = expert_map[topk_ids.long()]
    valid = (local >= 0).reshape(-1, 1)
    clamped = torch.where(local >= 0, local, torch.zeros_like(local))
    onehot = torch.nn.functional.one_hot(
        clamped.reshape(-1).long(), num_local_experts
    ).bool()
    onehot &= valid
    return onehot.any(0).to(torch.int32)
