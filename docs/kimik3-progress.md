
## Phase 2 continuation 35 (2026-08-05 16:45–17:15) — LL dead end confirmed; tuner queue

- **O4 (NCCL_PROTO=LL): WASH** (xnpt2, job 20738145): C=1 2.70 vs baseline
  2.65 tok/s; C=5 11.42 vs 11.59. Null in the 64-rank cross-node regime too
  (GLM's intra-node -12% does not transfer, but no win either). Dead end;
  default RCCL stays.
- **bf16 split-K GEMV: 1.00x overall** (gemvtune 20737946) — hipBLAS is fine
  in the eager stack. Dead end (GLM's win was vs a compile-frozen config).
- Router fast path (VLLM_GROUPED_TOPK_FAST) staged as arm xgf; MoE packed
  GEMM tuner (208us/call vs ~25us roofline) fixed and requeued behind
  dev-g slots (GLM campaign currently holds both).
- FULL-capture deadlock forensics (benchcg15): hang at capture of batch size
  2/3 — RCCL collective divergence inside capture. Matches GLM open problem
  ("capture-safe fast all-reduce on gfx90a — nobody cracked it"). Candidate
  unlock: one-shot RCCL warmup on the capture stream before capture begins
  (RCCL lazy connection setup is capture-illegal; pre-warmed communicators
  are capturable). Staged as the next big structural experiment after the
  cheap arms.

