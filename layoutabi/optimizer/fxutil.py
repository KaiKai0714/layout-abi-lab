"""Helpers for inspecting public FX / aten graphs without private inductor IR."""

from __future__ import annotations

from typing import Any

from torch.fx import Node


def describe_target(target: Any) -> str:
    if isinstance(target, str):
        return target
    text = str(target)
    name = getattr(target, "__name__", "")
    if name and name not in {"default", "int", "Tensor"}:
        return name
    return text


def target_key(target: Any) -> str:
    return describe_target(target).lower()


def is_softmax(node: Node) -> bool:
    if node.op == "call_method" and node.target == "softmax":
        return True
    if node.op != "call_function":
        return False
    key = target_key(node.target)
    return "softmax" in key


def is_matmul(node: Node) -> bool:
    if node.op == "call_method" and node.target in {"matmul", "mm", "bmm"}:
        return True
    if node.op != "call_function":
        return False
    name = getattr(node.target, "__name__", "")
    if name in {"matmul", "mm", "bmm"}:
        return True
    key = target_key(node.target)
    if "addmm" in key:
        return False
    return "matmul" in key or key.endswith(".mm") or key.endswith(".bmm")


def is_transpose(node: Node) -> bool:
    if node.op == "call_method" and node.target in {"transpose", "mT", "t"}:
        return True
    if node.op != "call_function":
        return False
    return "transpose" in target_key(node.target)


def softmax_dim(node: Node) -> int | None:
    if "dim" in node.kwargs and isinstance(node.kwargs["dim"], int):
        return int(node.kwargs["dim"])
    ints = [value for value in node.args[1:] if isinstance(value, int)]
    if ints:
        return ints[0]
    return None


def transpose_dims(node: Node) -> tuple[int, int] | None:
    if node.op == "call_method" and node.target in {"mT", "t"}:
        return (-2, -1)
    ints = [value for value in node.args[1:] if isinstance(value, int)]
    if len(ints) >= 2:
        return int(ints[0]), int(ints[1])
    dim0 = node.kwargs.get("dim0")
    dim1 = node.kwargs.get("dim1")
    if isinstance(dim0, int) and isinstance(dim1, int):
        return dim0, dim1
    return None


def swaps_last_two(dims: tuple[int, int] | None, rank: int | None) -> bool:
    if dims is None:
        return False
    dim0, dim1 = dims
    if rank is None:
        return {dim0, dim1} == {-2, -1}
    return {dim0 % rank, dim1 % rank} == {rank - 2, rank - 1}


def is_last_dim(dim: int | None, rank: int | None) -> bool:
    if dim is None:
        return False
    if rank is None:
        return dim == -1
    return dim % rank == rank - 1


def matmul_args(node: Node) -> tuple[Node | None, Node | None]:
    tensors = [value for value in node.args if isinstance(value, Node)]
    if len(tensors) >= 2:
        return tensors[0], tensors[1]
    return None, None


def node_rank(node: Node) -> int | None:
    meta = node.meta.get("tensor_meta")
    if meta is None:
        meta = node.meta.get("val")
    shape = getattr(meta, "shape", None)
    if shape is None and hasattr(meta, "size"):
        try:
            shape = meta.size()
        except Exception:
            shape = None
    if shape is None:
        return None
    try:
        return len(tuple(shape))
    except TypeError:
        return None


def graph_fingerprint(graph_module: Any) -> str:
    import hashlib
    import json

    rows = []
    for node in graph_module.graph.nodes:
        if node.op in {"placeholder", "output", "get_attr"}:
            rows.append([node.op, str(node.target)])
            continue
        literals = []
        for value in list(node.args) + list(node.kwargs.values()):
            if isinstance(value, (int, float, bool, str)) or value is None:
                literals.append(value)
        rows.append([node.op, target_key(node.target), literals])
    payload = json.dumps(rows, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
