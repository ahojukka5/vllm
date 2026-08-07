
## Bisection continues: custom-kernel toggles mis-fire, zmq flake (17:00–17:55)

- **xbndg12 (20819097)** FULL+GROUP+fixed-CustomAR: still 0x1000 at
  first decode replay → either EP-RCCL-in-graph (still present there)
  or a model kernel.
- **xbndg13 (20819485)** SKIP_EP+fixed-CustomAR: still 0x1000 → if the
  CustomAR fix is effective (it is; registration-skip log + copy path),
  the faulter is a MODEL kernel (KDA/MLA/attn_res/mxfp4/sampler), not
  collectives at all.
- **xbndg14 (20819858)** all-custom-kernels-off: capture died instantly
  with `hipErrorStreamCaptureUnsupported` — the stock fallback router
  uses capture-illegal torch.bincount. Lesson: VLLM_GROUPED_TOPK_FAST=1
  is *required* for capture on this stack; toggle-bisection of custom
  kernels is not viable this way.
- **xbndg15 (20820402, 20820892)**: SKIP_EP + fixed-CustomAR + GROUP +
  HSA_COREDUMP/HSA_ENABLE_DEBUG — first attempt: capture 100% OK, then
  killed by a zmq "Address already in use" collision at API startup
  (second occurrence today; vLLM get_open_port TOCTOU with 8 API procs
  on the head node). Retry pending on dev-g resources.
- CustomAR fix confirmed correct-by-reading AND the "warmup mimics
  allocation" branch means eager post-capture ARs were always
  copy-path; the captured registered=True calls were the bug.
