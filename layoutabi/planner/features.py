"""Interpretable features for layout-repair decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DecisionFeatures:
    n: int | None
    n_mod_8: int | None
    dtype: str
    dtype_bytes: int
    batch: int
    heads: int | None
    dim: int | None
    scope: str
    cuda: bool
    device: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_DTYPE_BYTES = {
    "fp16": 2,
    "float16": 2,
    "torch.float16": 2,
    "bf16": 2,
    "bfloat16": 2,
    "torch.bfloat16": 2,
    "fp32": 4,
    "float32": 4,
    "torch.float32": 4,
}


def dtype_bytes(dtype: str) -> int:
    return _DTYPE_BYTES.get(str(dtype).lower(), 0)


def features_from_sizes(
    *,
    n: int | None,
    dtype: str = "fp16",
    batch: int = 1,
    heads: int | None = None,
    dim: int | None = None,
    scope: str = "live",
    cuda: bool = False,
    device: str = "unknown",
) -> DecisionFeatures:
    n_mod = None if n is None else int(n) % 8
    return DecisionFeatures(
        n=n,
        n_mod_8=n_mod,
        dtype=str(dtype),
        dtype_bytes=dtype_bytes(str(dtype)),
        batch=int(batch),
        heads=heads,
        dim=dim,
        scope=scope,
        cuda=cuda,
        device=device,
    )


def features_from_index_row(
    row: dict[str, Any],
    *,
    environment: dict[str, Any] | None = None,
    scope: str,
) -> DecisionFeatures:
    env = environment or {}
    consumer = row.get("consumer_n")
    n = int(consumer) if isinstance(consumer, int) else None
    dtype = str(row.get("dtype") or "fp16")
    batch = int(row["batch"]) if isinstance(row.get("batch"), int) else 1
    device = str(env.get("device") or "unknown")
    return features_from_sizes(
        n=n,
        dtype=dtype,
        batch=batch,
        scope=scope,
        cuda=True,
        device=device,
    )


def features_from_live(
    example_inputs: tuple[Any, ...],
    *,
    matches: list[Any] | None = None,
    graph_module: Any | None = None,
) -> DecisionFeatures:
    import torch

    tensors = [value for value in example_inputs if isinstance(value, torch.Tensor)]
    first = tensors[0]
    dtype = str(first.dtype).replace("torch.", "")
    batch = int(first.shape[0]) if first.ndim >= 1 else 1
    n = None
    heads = None
    dim = None
    if graph_module is not None and matches:
        nodes = {node.name: node for node in graph_module.graph.nodes}
        softmax = nodes.get(getattr(matches[0], "k_softmax", ""))
        meta = getattr(softmax, "meta", {}) if softmax is not None else {}
        tensor_meta = meta.get("tensor_meta") or meta.get("val")
        shape = getattr(tensor_meta, "shape", None)
        if shape is not None and len(tuple(shape)) >= 1:
            dims = tuple(int(item) for item in shape)
            n = dims[-1]
            if len(dims) >= 4:
                batch = dims[0]
                heads = dims[1]
                dim = dims[2]
    if n is None and first.ndim == 4 and first.shape[1] in {3, 64}:
        # NCHW activation: K last dim is H*W (+ memory tokens are graph-specific).
        n = int(first.shape[-2] * first.shape[-1])
    elif n is None and first.ndim >= 2:
        n = int(first.shape[-1])
    return features_from_sizes(
        n=n,
        dtype=dtype,
        batch=batch,
        heads=heads,
        dim=dim,
        scope="live",
        cuda=first.device.type == "cuda",
        device=str(first.device),
    )
