# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any


def tensors_str_no_data(arg: Any):
    # Metadata-only formatting: str(tensor) — even with
    # printoptions(threshold=1) — calls torch.masked_select while
    # summarizing values, which is illegal during CUDA graph capture
    # (hipErrorStreamCaptureUnsupported). IR-op debug logging evaluates
    # this lazily inside captured regions, so it must never read tensor
    # data. Shapes/dtypes/devices are sufficient for op tracing.
    import torch

    def _fmt(a):
        if isinstance(a, torch.Tensor):
            return (f"Tensor(shape={tuple(a.shape)}, dtype={a.dtype}, "
                    f"device={a.device})")
        if isinstance(a, (list, tuple)):
            return type(a)(_fmt(x) for x in a)
        if isinstance(a, dict):
            return {k: _fmt(v) for k, v in a.items()}
        return a

    return str(_fmt(arg))
