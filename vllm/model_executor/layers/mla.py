# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from dataclasses import dataclass

import torch

from vllm.config import CacheConfig
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.platforms import current_platform

_MLA_DEBUG_STATS = os.environ.get("KDA_DEBUG_STATS") == "1"
_mla_debug_call_counts: dict[str, int] = {}


def _mla_debug_stats(tag: str, x: torch.Tensor, max_calls: int = 40) -> None:
    """Temporary diagnostic: log tensor stats for the Gated-MLA output-gate
    investigation (docs/investigation.md). No-op unless KDA_DEBUG_STATS=1;
    only ever called from layers that construct a g_proj (Kimi-K3 MLA), so
    this is a no-op for every other model regardless of the env var."""
    if not _MLA_DEBUG_STATS:
        return
    try:
        from vllm.distributed import get_dp_group, get_tensor_model_parallel_rank

        if get_tensor_model_parallel_rank() != 0:
            return
        dp_rank = get_dp_group().rank_in_group
    except Exception:
        dp_rank = -1
    count = _mla_debug_call_counts.get(tag, 0)
    if count >= max_calls:
        return
    _mla_debug_call_counts[tag] = count + 1
    from vllm.logger import init_logger

    logger = init_logger(__name__)
    xf = x.detach().float()
    finite = xf[torch.isfinite(xf)]
    if finite.numel() == 0:
        logger.warning(
            "MLA_GATE_DEBUG dp=%d %s call=%d shape=%s ALL NON-FINITE",
            dp_rank, tag, count, tuple(x.shape),
        )
        return
    logger.warning(
        "MLA_GATE_DEBUG dp=%d %s call=%d shape=%s mean=%.6f std=%.6f min=%.6f "
        "max=%.6f absmax=%.6f last_row_mean=%.6f last_row_absmax=%.6f",
        dp_rank, tag, count, tuple(x.shape),
        finite.mean().item(), finite.std().item(),
        finite.min().item(), finite.max().item(), finite.abs().max().item(),
        x[-1].detach().float().mean().item() if x.dim() >= 1 and x.shape[0] > 0 else -1.0,
        x[-1].detach().float().abs().max().item() if x.dim() >= 1 and x.shape[0] > 0 else -1.0,
    )


def _mla_debug_argmax_row(
    tag: str, pre_norm: torch.Tensor, post_norm: torch.Tensor, max_calls: int = 40
) -> None:
    """Temporary diagnostic: find the token row with the largest |post_norm|
    value and report that row's pre-norm RMS (to test the hypothesis that
    kv_a_layernorm blows up when its input has near-zero variance at a
    specific prefill position) plus which row index it is."""
    if not _MLA_DEBUG_STATS:
        return
    if post_norm.dim() < 2 or post_norm.shape[0] <= 1:
        return
    try:
        from vllm.distributed import get_dp_group, get_tensor_model_parallel_rank

        if get_tensor_model_parallel_rank() != 0:
            return
        dp_rank = get_dp_group().rank_in_group
    except Exception:
        dp_rank = -1
    count = _mla_debug_call_counts.get(tag, 0)
    if count >= max_calls:
        return
    _mla_debug_call_counts[tag] = count + 1
    from vllm.logger import init_logger

    logger = init_logger(__name__)
    post_f = post_norm.detach().float()
    pre_f = pre_norm.detach().float()
    row_absmax = post_f.abs().amax(dim=tuple(range(1, post_f.dim())))
    row = int(torch.argmax(row_absmax).item())
    pre_row = pre_f[row]
    pre_row_rms = pre_row.pow(2).mean().sqrt().item()
    pre_row_absmax = pre_row.abs().max().item()
    post_row_absmax = row_absmax[row].item()
    num_rows = post_f.shape[0]
    logger.warning(
        "MLA_GATE_DEBUG dp=%d %s call=%d ARGMAX row=%d/%d pre_norm_rms=%.8f "
        "pre_norm_absmax=%.6f post_norm_absmax=%.6f",
        dp_rank, tag, count, row, num_rows, pre_row_rms, pre_row_absmax,
        post_row_absmax,
    )


@dataclass
class MLAModules:
    """Modules used in MLA."""

    kv_a_layernorm: torch.nn.Module
    kv_b_proj: torch.nn.Module
    rotary_emb: torch.nn.Module
    o_proj: torch.nn.Module
    fused_qkv_a_proj: torch.nn.Module | None
    kv_a_proj_with_mqa: torch.nn.Module | None
    q_a_layernorm: torch.nn.Module | None
    q_b_proj: torch.nn.Module | None
    q_proj: torch.nn.Module | None
    indexer: torch.nn.Module | None
    is_sparse: bool
    topk_indices_buffer: torch.Tensor | None
    indexer_rotary_emb: torch.nn.Module | None = None
    g_proj: torch.nn.Module | None = None


# --8<-- [start:multi_head_latent_attention]
@PluggableLayer.register("multi_head_latent_attention")
class MultiHeadLatentAttentionWrapper(PluggableLayer):
    """Pluggable MLA layer which allows OOT backends to add
    custom implementations of the outer MLA layer (including rope & o_proj).
    Note that currently oot platforms can still use CustomOp.register_oot to
    replace MLA layer entirely, although we use PluggableLayer to register
    this layer now.

    This class takes positions and hidden_states as input.
    The input tensors can either contain prefill tokens or decode tokens.
    The class does the following:

    1. MLA Preprocess.
    2. Perform multi-head attention to prefill tokens and
       multi-query attention to decode tokens separately.
    3. Return the output tensor.
    """

    # --8<-- [end:multi_head_latent_attention]

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        mla_modules: MLAModules,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        skip_topk: bool = False,
        non_causal_multi_token_decode: bool = False,
        allow_short_prefill_indexer_scoring_skip: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.fused_qkv_a_proj = mla_modules.fused_qkv_a_proj
        self.kv_a_proj_with_mqa = mla_modules.kv_a_proj_with_mqa
        self.q_a_layernorm = mla_modules.q_a_layernorm
        self.q_b_proj = mla_modules.q_b_proj
        self.q_proj = mla_modules.q_proj
        self.kv_a_layernorm = mla_modules.kv_a_layernorm
        self.kv_b_proj = mla_modules.kv_b_proj
        self.rotary_emb = mla_modules.rotary_emb
        self.o_proj = mla_modules.o_proj
        self.indexer = mla_modules.indexer
        self.indexer_rope_emb = mla_modules.indexer_rotary_emb
        self.is_sparse = mla_modules.is_sparse
        self.g_proj = mla_modules.g_proj

        # Whether to skip top-k token selection computation in this layer.
        # When True, the indexer will not be called, and the layer will reuse
        # the topk_tokens buffer written by a previous layer in the same pass.
        # Refer: https://arxiv.org/abs/2603.12201 for more details.
        self.skip_topk = skip_topk
        # qrep is active when the query projection is a DCP-group-sharded layer
        # that materializes the full group head set locally.
        q_proj_layer = self.q_b_proj if self.q_lora_rank is not None else self.q_proj
        self.dcp_q_replicate = getattr(q_proj_layer, "qrep_active", False)
        if self.indexer is not None:
            assert hasattr(self.indexer, "topk_tokens")
            self.topk_tokens = self.indexer.topk_tokens
            self.topk_indices_buffer = mla_modules.topk_indices_buffer

        self.mla_attn = MLAAttention(
            num_heads=self.num_heads,
            scale=scale,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            dcp_q_replicate=self.dcp_q_replicate,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
            topk_indices_buffer=mla_modules.topk_indices_buffer,
            non_causal_multi_token_decode=non_causal_multi_token_decode,
        )
        indexer_op = getattr(self.indexer, "indexer_op", None)
        if indexer_op is not None and hasattr(
            indexer_op, "dense_mha_metadata_layer_name"
        ):
            enable_short_prefill_scoring_skip = (
                allow_short_prefill_indexer_scoring_skip
                and not self.skip_topk
                and not getattr(indexer_op, "use_pcp", False)
                and current_platform.is_cuda()
            )
            # The indexer and main MLA use independent decode thresholds and
            # may classify the same short extend differently. Bind the main
            # MLA layer name so the eager indexer op can check whether the
            # batch's top-k indices will be consumed.
            # PCP is excluded because indexer cache/scoring ownership differs
            # across ranks and the no-consumer invariant has not been
            # established there.
            indexer_op.dense_mha_metadata_layer_name = (
                self.mla_attn.layer_name if enable_short_prefill_scoring_skip else ""
            )
        self.prefix = prefix

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_c = None
        kv_lora = None

        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None, (
                "fused_qkv_a_proj is required when q_lora_rank is not None"
            )
            assert self.q_a_layernorm is not None, (
                "q_a_layernorm is required when q_lora_rank is not None"
            )
            assert self.q_b_proj is not None, (
                "q_b_proj is required when q_lora_rank is not None"
            )

            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)
            q_proj_layer = self.q_b_proj
            q_proj_input = q_c
        else:
            assert self.kv_a_proj_with_mqa is not None, (
                "kv_a_proj_with_mqa is required when q_lora_rank is None"
            )
            assert self.q_proj is not None, (
                "q_proj is required when q_lora_rank is None"
            )
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q_proj_layer = self.q_proj
            q_proj_input = hidden_states

        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)
        if self.g_proj is not None:
            _mla_debug_argmax_row(f"{self.prefix}_kv_c_layernorm", kv_c, kv_c_normed)
        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        q = q_proj_layer(q_proj_input)[0]
        heads = self.num_heads
        if self.dcp_q_replicate:
            heads *= q_proj_layer.group_size
        q = q.view(-1, heads, self.qk_head_dim)

        if self.rotary_emb is not None:
            q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                positions, q[..., self.qk_nope_head_dim :], k_pe
            )

        if self.indexer and self.is_sparse and not self.skip_topk:
            self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        q_dcp_replicated = None
        if self.dcp_q_replicate:
            q_dcp_replicated, q = q, q_proj_layer._local_view(q)

        if self.g_proj is not None:
            _mla_debug_stats(f"{self.prefix}_hidden_states_in", hidden_states)
            _mla_debug_stats(f"{self.prefix}_q_pre_attn", q)
            _mla_debug_stats(f"{self.prefix}_kv_c_normed_pre_attn", kv_c_normed)
            _mla_debug_stats(f"{self.prefix}_k_pe_pre_attn", k_pe)

        attn_out = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
            q_dcp_replicated=q_dcp_replicated,
        )

        if self.g_proj is not None:
            _mla_debug_stats(f"{self.prefix}_attn_out_pre_gate", attn_out)
            gate = self.g_proj(hidden_states)[0].sigmoid()
            _mla_debug_stats(f"{self.prefix}_gate_value", gate)
            attn_out = attn_out * gate
            _mla_debug_stats(f"{self.prefix}_attn_out_post_gate", attn_out)

        out = self.o_proj(attn_out)[0]
        if self.g_proj is not None:
            _mla_debug_stats(f"{self.prefix}_o_proj_out", out)
        return out
