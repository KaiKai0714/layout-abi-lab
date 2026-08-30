"""Oracle comparison metrics. Gates were frozen before scoring held-out rows."""

from __future__ import annotations

import math
from typing import Any

# Frozen v0.6 gates. Do not retune after inspecting held-out community rows.
GEOMEAN_REGRET_GATE = 1.05
MAX_REGRET_GATE = 1.15


def oracle_action(outcome: str) -> str | None:
    if outcome == "repair_win":
        return "repair_kv"
    if outcome == "direct_win":
        return "direct"
    if outcome == "parity":
        return "direct"
    return None


def cell_regret(action: str, oracle: str, direct_ms: float, repair_ms: float) -> float:
    chosen = repair_ms if action == "repair_kv" else direct_ms
    best = repair_ms if oracle == "repair_kv" else direct_ms
    if best <= 0:
        return 1.0
    return max(chosen / best, 1.0)


def geometric_mean(values: list[float]) -> float | None:
    if not values:
        return None
    logs = [math.log(max(value, 1e-12)) for value in values]
    return math.exp(sum(logs) / len(logs))


def score_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score planner actions against an oracle. Autotune is treated as oracle."""

    n = len(rows)
    matches = 0
    false_repair = 0
    false_direct = 0
    regrets: list[float] = []
    for row in rows:
        oracle = row["oracle"]
        action = row["action"]
        if action == "autotune":
            action = oracle
        if action == oracle:
            matches += 1
        elif action == "repair_kv" and oracle == "direct":
            false_repair += 1
        elif action == "direct" and oracle == "repair_kv":
            false_direct += 1
        regrets.append(
            cell_regret(action, oracle, row["direct_ms"], row["repair_ms"])
        )
    geomean = geometric_mean(regrets)
    max_regret = max(regrets) if regrets else None
    return {
        "n": n,
        "oracle_match_rate": (matches / n) if n else None,
        "false_repair_rate": (false_repair / n) if n else None,
        "false_direct_rate": (false_direct / n) if n else None,
        "false_repair_count": false_repair,
        "false_direct_count": false_direct,
        "geomean_regret": geomean,
        "max_regret": max_regret,
        "geomean_regret_gate": GEOMEAN_REGRET_GATE,
        "max_regret_gate": MAX_REGRET_GATE,
        "pass_geomean_gate": geomean is not None and geomean <= GEOMEAN_REGRET_GATE,
        "pass_max_regret_gate": max_regret is not None and max_regret <= MAX_REGRET_GATE,
        "pass_false_repair_zero": false_repair == 0,
    }
