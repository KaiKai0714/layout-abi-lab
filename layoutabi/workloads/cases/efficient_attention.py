"""Source-equivalent Efficient Attention (Shen et al., WACV 2021).

Provenance:
  Repository: https://github.com/cmsflash/efficient-attention
  Commit: 46a5f9eaf09470affb0ab30932b7748cc3c871ef
  Source: efficient_attention.py
  License: MIT
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .._runtime import prepare_module


class PublicEfficientAttention(nn.Module):
    """Public Efficient Attention equations without a layout policy switch."""

    def __init__(
        self,
        in_channels: int = 64,
        key_channels: int = 64,
        head_count: int = 4,
        value_channels: int = 64,
    ) -> None:
        super().__init__()
        if key_channels % head_count != 0 or value_channels % head_count != 0:
            raise ValueError("key and value channels must divide head_count")
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        self.head_key_channels = key_channels // head_count
        self.head_value_channels = value_channels // head_count
        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_: Tensor) -> Tensor:
        batch, _channels, height, width = input_.shape
        spatial = height * width
        keys = self.keys(input_).reshape(
            batch, self.head_count, self.head_key_channels, spatial
        )
        queries = self.queries(input_).reshape(
            batch, self.head_count, self.head_key_channels, spatial
        )
        values = self.values(input_).reshape(
            batch, self.head_count, self.head_value_channels, spatial
        )
        key = keys.softmax(dim=-1)
        query = queries.softmax(dim=-2)
        context = key @ values.transpose(-2, -1)
        attended = context.transpose(-2, -1) @ query
        aggregated = attended.reshape(batch, self.value_channels, height, width)
        return self.reprojection(aggregated) + input_


def published_loop_forward(module: PublicEfficientAttention, input_: Tensor) -> Tensor:
    """Reference implementation of the published per-head loop, for tests."""

    batch, _, height, width = input_.shape
    keys = module.keys(input_).reshape((batch, module.key_channels, height * width))
    queries = module.queries(input_).reshape((batch, module.key_channels, height * width))
    values = module.values(input_).reshape((batch, module.value_channels, height * width))
    attended_values = []
    for index in range(module.head_count):
        key = F.softmax(
            keys[
                :,
                index * module.head_key_channels : (index + 1) * module.head_key_channels,
                :,
            ],
            dim=2,
        )
        query = F.softmax(
            queries[
                :,
                index * module.head_key_channels : (index + 1) * module.head_key_channels,
                :,
            ],
            dim=1,
        )
        value = values[
            :,
            index * module.head_value_channels : (index + 1) * module.head_value_channels,
            :,
        ]
        context = key @ value.transpose(1, 2)
        attended_value = (context.transpose(1, 2) @ query).reshape(
            batch, module.head_value_channels, height, width
        )
        attended_values.append(attended_value)
    aggregated = torch.cat(attended_values, dim=1)
    return module.reprojection(aggregated) + input_


def build(*, resolution: int, batch: int, dtype: Any) -> tuple[Any, tuple[Any, ...]]:
    module = prepare_module(PublicEfficientAttention(), dtype)
    sample = torch.randn(batch, 64, resolution, resolution, dtype=dtype)
    return module, (sample,)


def reference_outputs(module: Any, inputs: tuple[Any, ...]) -> Any:
    return published_loop_forward(module, inputs[0])
