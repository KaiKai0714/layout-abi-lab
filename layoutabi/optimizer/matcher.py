"""Bounded matcher for the LinearAttention K-softmax → KTV GEMM pattern."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch.fx import GraphModule, Node

from .fxutil import (
    is_last_dim,
    is_matmul,
    is_softmax,
    is_transpose,
    matmul_args,
    node_rank,
    softmax_dim,
    swaps_last_two,
    transpose_dims,
)
from .pattern import PATTERN_ID


@dataclass
class PatternMatch:
    pattern_id: str
    k_softmax: str
    k_name: str
    v_name: str
    v_transpose: str
    ktv_matmul: str
    context_transpose: str | None
    q_matmul: str | None
    softmax_dim: int
    location: str


@dataclass
class MatchResult:
    matches: list[PatternMatch] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)


def _reject(result: MatchResult, location: str, reason: str) -> None:
    result.rejections.append({"location": location, "reason": reason})


def match_graph(graph_module: GraphModule) -> MatchResult:
    """Find supported KTV sites. Does not inspect module class names or paths."""

    result = MatchResult()
    for node in graph_module.graph.nodes:
        if not is_softmax(node):
            continue
        dim = softmax_dim(node)
        rank = node_rank(node)
        if not is_last_dim(dim, rank):
            # Q-softmax uses dim=-2 in this pattern; that is not a rejection.
            continue
        matmul_users = [user for user in node.users if is_matmul(user)]
        if not matmul_users:
            continue
        for user in matmul_users:
            _match_ktv(result, node, user, dim if dim is not None else -1, rank)
    return result


def _match_ktv(
    result: MatchResult,
    softmax_node: Node,
    matmul_node: Node,
    dim: int,
    rank: int | None,
) -> None:
    lhs, rhs = matmul_args(matmul_node)
    if lhs is None or rhs is None:
        _reject(result, matmul_node.name, "matmul does not have two tensor operands")
        return
    if lhs is not softmax_node:
        _reject(
            result,
            matmul_node.name,
            "K-softmax is not the left operand of the consumer GEMM",
        )
        return
    if not is_transpose(rhs):
        _reject(
            result,
            matmul_node.name,
            "right operand is not a transpose of V",
        )
        return
    rhs_rank = node_rank(rhs) or rank
    if not swaps_last_two(transpose_dims(rhs), rhs_rank):
        _reject(
            result,
            rhs.name,
            "V transpose does not swap the last two dimensions",
        )
        return
    v_node = rhs.args[0] if rhs.args and isinstance(rhs.args[0], Node) else None
    if v_node is None:
        _reject(result, rhs.name, "transpose has no tensor source")
        return

    context_transpose = None
    q_matmul = None
    for consumer in matmul_node.users:
        if not is_transpose(consumer):
            continue
        if not swaps_last_two(transpose_dims(consumer), node_rank(consumer) or rank):
            continue
        gemm2 = [user for user in consumer.users if is_matmul(user)]
        if gemm2:
            context_transpose = consumer.name
            q_matmul = gemm2[0].name
            break
    if context_transpose is None:
        _reject(
            result,
            matmul_node.name,
            "KTV GEMM is not followed by context.transpose @ Q",
        )
        return

    result.matches.append(
        PatternMatch(
            pattern_id=PATTERN_ID,
            k_softmax=softmax_node.name,
            k_name=softmax_node.name,
            v_name=v_node.name,
            v_transpose=rhs.name,
            ktv_matmul=matmul_node.name,
            context_transpose=context_transpose,
            q_matmul=q_matmul,
            softmax_dim=dim,
            location=matmul_node.name,
        )
    )
