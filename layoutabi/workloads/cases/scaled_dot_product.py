"""Public negative graph: scaled dot-product attention (Vaswani et al. 2017)."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .._runtime import prepare_module


class PublicScaledDotProductAttention(nn.Module):
    """Standard QK^T softmax V attention used as a public no-op control."""

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        scale = query.shape[-1] ** -0.5
        scores = (query @ key.transpose(-2, -1)) * scale
        return scores.softmax(dim=-1) @ value


def build(*, resolution: int, batch: int, dtype: Any) -> tuple[Any, tuple[Any, ...]]:
    module = prepare_module(PublicScaledDotProductAttention(), dtype)
    qkv = tuple(torch.randn(batch, 4, resolution, 32, dtype=dtype) for _ in range(3))
    return module, qkv
