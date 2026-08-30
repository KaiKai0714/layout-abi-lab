"""Semantic, shape, dtype, and device guards for accepted matches."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.fx import GraphModule

from .capture import normalize_inputs
from .matcher import PatternMatch
from .pattern import REQUIRED_DTYPE_NAME


def _tensor_inputs(example_inputs: tuple[Any, ...]) -> list[torch.Tensor]:
    return [value for value in example_inputs if isinstance(value, torch.Tensor)]


def input_guard_problems(
    model: nn.Module, example_inputs: tuple[Any, ...]
) -> list[str]:
    problems: list[str] = []
    if model.training:
        problems.append("inference only: model is in training mode")
    tensors = _tensor_inputs(example_inputs)
    if not tensors:
        problems.append("example_inputs contain no tensors")
        return problems
    dtypes = {tensor.dtype for tensor in tensors}
    if dtypes != {torch.float16}:
        names = sorted(str(dtype) for dtype in dtypes)
        problems.append(
            f"supported dtype is {REQUIRED_DTYPE_NAME}; got {', '.join(names)}"
        )
    devices = {str(tensor.device.type) for tensor in tensors}
    if len(devices) != 1:
        problems.append(f"example_inputs span multiple device types: {sorted(devices)}")
    for tensor in tensors:
        if any(size <= 0 for size in tensor.shape):
            problems.append("dynamic or empty input shape is not supported")
    return problems


def match_guard_problems(
    graph_module: GraphModule,
    match: PatternMatch,
    example_inputs: tuple[Any, ...],
) -> list[str]:
    problems: list[str] = []
    nodes = {node.name: node for node in graph_module.graph.nodes}
    softmax = nodes.get(match.k_softmax)
    if softmax is None:
        return ["matched softmax node is missing from the captured graph"]
    rank = None
    meta = softmax.meta.get("tensor_meta") or softmax.meta.get("val")
    shape = getattr(meta, "shape", None)
    if shape is not None:
        rank = len(tuple(shape))
        if rank != 4:
            problems.append(f"K producer rank must be 4; got {rank}")
    tensors = _tensor_inputs(example_inputs)
    if tensors and tensors[0].device.type != "cuda":
        # Rewrite is allowed on CPU for tests; autotune enforces CUDA separately.
        pass
    return problems


def require_cuda(example_inputs: tuple[Any, ...]) -> list[str]:
    tensors = _tensor_inputs(normalize_inputs(example_inputs))
    if not tensors:
        return ["autotune requires CUDA example inputs"]
    if any(tensor.device.type != "cuda" for tensor in tensors):
        return ["autotune requires CUDA example inputs"]
    if not torch.cuda.is_available():
        return ["autotune requires a CUDA-enabled PyTorch build"]
    return []
