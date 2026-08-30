"""Shared helpers for case builders. Torch is imported only when building."""

from __future__ import annotations

from typing import Any


def torch_dtype(name: str) -> Any:
    import torch

    mapping = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dtype {name!r}") from exc


def prepare_module(module: Any, dtype: Any) -> Any:
    import torch

    module = module.eval()
    if dtype == torch.float16:
        return module.half()
    if dtype != torch.float32:
        return module.to(dtype=dtype)
    return module


def place(module: Any, tensors: tuple[Any, ...], device: str) -> tuple[Any, tuple[Any, ...]]:
    if device == "cuda":
        module = module.cuda()
        tensors = tuple(tensor.cuda() for tensor in tensors)
    return module, tensors
