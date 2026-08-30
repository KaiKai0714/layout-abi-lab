"""Insert BHND-backed materialization before a matched KTV GEMM."""

from __future__ import annotations

import copy

from torch.fx import GraphModule, Node

from .matcher import PatternMatch


def _nodes(graph_module: GraphModule) -> dict[str, Node]:
    return {node.name: node for node in graph_module.graph.nodes}


def _insert_bhnd(graph_module: GraphModule, before: Node, source: Node) -> Node:
    with graph_module.graph.inserting_before(before):
        transposed = graph_module.graph.call_method("transpose", args=(source, -2, -1))
        contiguous = graph_module.graph.call_method("contiguous", args=(transposed,))
        repaired = graph_module.graph.call_method("transpose", args=(contiguous, -2, -1))
    return repaired


def apply_repair(
    graph_module: GraphModule,
    matches: list[PatternMatch],
    *,
    repair_k: bool,
    repair_v: bool,
) -> GraphModule:
    """Return a copied graph with layout repair inserted at every match."""

    rewritten = copy.deepcopy(graph_module)
    if not repair_k and not repair_v:
        rewritten.graph.lint()
        rewritten.recompile()
        return rewritten
    for match in matches:
        nodes = _nodes(rewritten)
        matmul_node = nodes[match.ktv_matmul]
        k_node = nodes[match.k_name]
        v_node = nodes[match.v_name]
        lhs, rhs = list(matmul_node.args[:2])
        if repair_k:
            lhs = _insert_bhnd(rewritten, matmul_node, k_node)
        if repair_v:
            repaired_v = _insert_bhnd(rewritten, matmul_node, v_node)
            with rewritten.graph.inserting_before(matmul_node):
                rhs = rewritten.graph.call_method("transpose", args=(repaired_v, -2, -1))
        matmul_node.args = (lhs, rhs, *matmul_node.args[2:])
    rewritten.graph.lint()
    rewritten.recompile()
    return rewritten
