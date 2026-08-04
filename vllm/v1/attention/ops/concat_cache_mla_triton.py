# SPDX-License-Identifier: Apache-2.0
"""Single-launch Triton concat_and_cache_mla for ROCm.

Replaces the PyTorch-native fallback, which per call runs torch.nonzero
(a hidden GPU sync and capture-illegal dynamic allocation) plus ~6 more
kernels, on every MLA layer every step. One program per slot-mapping
entry: copies kv_c[t] and k_pe[t] into the paged cache, skipping
negative (padding) slots. Static grid over slot_mapping.numel(), so it
is cudagraph-capture safe.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _concat_cache_mla_kernel(
    kv_c_ptr,
    k_pe_ptr,
    cache_ptr,
    slot_ptr,
    block_size,
    LORA: tl.constexpr,
    PE: tl.constexpr,
    BLOCK_LORA: tl.constexpr,
    BLOCK_PE: tl.constexpr,
):
    t = tl.program_id(0)
    slot = tl.load(slot_ptr + t)
    if slot < 0:
        return
    blk = slot // block_size
    off = slot % block_size
    entry = (blk * block_size + off) * (LORA + PE)

    cols = tl.arange(0, BLOCK_LORA)
    lmask = cols < LORA
    v = tl.load(kv_c_ptr + t * LORA + cols, mask=lmask)
    tl.store(cache_ptr + entry + cols, v, mask=lmask)

    pcols = tl.arange(0, BLOCK_PE)
    pmask = pcols < PE
    p = tl.load(k_pe_ptr + t * PE + pcols, mask=pmask)
    tl.store(cache_ptr + entry + LORA + pcols, p, mask=pmask)


def concat_and_cache_mla_triton(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    num_tokens = slot_mapping.numel()
    kv_lora_rank = kv_c.size(1)
    pe_dim = k_pe.size(1)
    block_size = kv_cache.size(1)
    assert kv_cache.size(2) == kv_lora_rank + pe_dim
    assert kv_c.stride(-1) == 1 and k_pe.stride(-1) == 1
    assert kv_cache.is_contiguous()
    assert kv_c.dtype == kv_cache.dtype and k_pe.dtype == kv_cache.dtype

    grid = (num_tokens,)
    _concat_cache_mla_kernel[grid](
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        block_size,
        LORA=kv_lora_rank,
        PE=pe_dim,
        BLOCK_LORA=triton.next_power_of_2(kv_lora_rank),
        BLOCK_PE=triton.next_power_of_2(pe_dim),
        num_warps=4,
    )
