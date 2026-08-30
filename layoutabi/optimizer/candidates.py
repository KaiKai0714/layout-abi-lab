"""Build direct and repair graph candidates from accepted matches."""

from __future__ import annotations

from torch.fx import GraphModule

from .matcher import PatternMatch
from .pattern import REWRITE_POLICIES
from .rewrite import apply_repair


def policies_for(policy: str) -> tuple[str, ...]:
    if policy == "autotune":
        return REWRITE_POLICIES
    if policy in REWRITE_POLICIES:
        return (policy,)
    return ()


def build_candidates(
    graph_module: GraphModule,
    matches: list[PatternMatch],
    policy: str,
) -> dict[str, GraphModule]:
    candidates: dict[str, GraphModule] = {}
    for name in policies_for(policy):
        candidates[name] = apply_repair(
            graph_module,
            matches,
            repair_k=name in {"repair_k", "repair_kv"},
            repair_v=name == "repair_kv",
        )
    return candidates
