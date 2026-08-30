"""Builder for the historical diffusion LinearAttention reference graph."""

from __future__ import annotations

from typing import Any


def build(*, resolution: int, batch: int, dtype: Any) -> tuple[Any, tuple[Any, ...]]:
    import torch

    from ...workload import PublicDiffusionLinearAttention
    from .._runtime import prepare_module

    module = prepare_module(PublicDiffusionLinearAttention(policy="direct"), dtype)
    sample = torch.randn(batch, 64, resolution, resolution, dtype=dtype)
    return module, (sample,)
