"""Score planner baselines on published result rows. CPU-only, no PyTorch."""

from __future__ import annotations

from typing import Any

from .features import DecisionFeatures, features_from_index_row
from .metrics import oracle_action, score_predictions
from .policies import EVAL_POLICY_NAMES, decide

SCOPES = ("eager_module", "compiled_module")


def _latency_pair(scope_cell: dict[str, Any]) -> tuple[float, float] | None:
    direct = scope_cell.get("direct_ms")
    repair = scope_cell.get("repair_kv_ms")
    if not isinstance(direct, (int, float)) or not isinstance(repair, (int, float)):
        return None
    if isinstance(direct, bool) or isinstance(repair, bool):
        return None
    return float(direct), float(repair)


def collect_oracle_rows(
    index: dict[str, Any],
    *,
    scopes: tuple[str, ...] = SCOPES,
    primary_only: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bundle in index.get("bundles", []):
        if primary_only and bundle.get("role") == "replicate":
            continue
        environment = bundle.get("environment") or {}
        for row in bundle.get("rows", []):
            for scope in scopes:
                cell = row.get(scope)
                if not isinstance(cell, dict):
                    continue
                outcome = cell.get("outcome")
                oracle = oracle_action(str(outcome))
                latencies = _latency_pair(cell)
                if oracle is None or latencies is None:
                    continue
                features = features_from_index_row(row, environment=environment, scope=scope)
                rows.append(
                    {
                        "bundle": bundle.get("id"),
                        "device": environment.get("device"),
                        "torch": environment.get("torch"),
                        "resolution": row.get("resolution"),
                        "consumer_n": row.get("consumer_n"),
                        "n_mod_8": features.n_mod_8,
                        "dtype": features.dtype,
                        "batch": features.batch,
                        "scope": scope,
                        "oracle": oracle,
                        "direct_ms": latencies[0],
                        "repair_ms": latencies[1],
                        "features": features.as_dict(),
                    }
                )
    return rows


def evaluate_rows(
    oracle_rows: list[dict[str, Any]],
    *,
    policies: tuple[str, ...] = EVAL_POLICY_NAMES,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for name in policies:
        predicted = []
        for row in oracle_rows:
            features = DecisionFeatures(**row["features"])
            decision = decide(features, name)
            predicted.append(
                {
                    **{key: row[key] for key in (
                        "bundle",
                        "resolution",
                        "scope",
                        "oracle",
                        "direct_ms",
                        "repair_ms",
                        "n_mod_8",
                    )},
                    "action": decision["action"],
                    "reason": decision["reason"],
                }
            )
        reports[name] = {
            "metrics": score_predictions(predicted),
            "predictions": predicted,
        }
    return reports


def n_mod_8_separates_oracle(oracle_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Whether N % 8 actually splits direct vs repair in the scored set."""

    pairs = sorted({(row.get("n_mod_8"), row["oracle"]) for row in oracle_rows})
    mods = {row.get("n_mod_8") for row in oracle_rows}
    oracles = {row["oracle"] for row in oracle_rows}
    return {
        "unique_n_mod_8": sorted(mods, key=lambda item: (item is None, item)),
        "unique_oracles": sorted(oracles),
        "pairs": [{"n_mod_8": a, "oracle": b} for a, b in pairs],
        "separates_oracle": len(mods) > 1 and len(oracles) > 1,
    }


def evaluate_index(
    index: dict[str, Any],
    *,
    scopes: tuple[str, ...] = SCOPES,
    held_out_resolutions: tuple[int, ...] = (128,),
) -> dict[str, Any]:
    all_rows = collect_oracle_rows(index, scopes=scopes)
    held_out = [
        row for row in all_rows if row.get("resolution") in held_out_resolutions
    ]
    calibration = [
        row for row in all_rows if row.get("resolution") not in held_out_resolutions
    ]
    return {
        "scope": list(scopes),
        "held_out_resolutions": list(held_out_resolutions),
        "n_mod_8_diagnostic": n_mod_8_separates_oracle(all_rows),
        "all": {"n": len(all_rows), "policies": _metrics_only(evaluate_rows(all_rows))},
        "calibration": {
            "n": len(calibration),
            "policies": _metrics_only(evaluate_rows(calibration)),
        },
        "held_out": {
            "n": len(held_out),
            "policies": _metrics_only(evaluate_rows(held_out)),
        },
        "note": (
            "N % 8 is scored as a community-testable hypothesis, not a shipping law. "
            "cost_model falls back to autotune when N % 8 != 0 so it does not bake "
            "false-repair into a static rule. Autotune is scored as matching the oracle."
        ),
    }


def _metrics_only(reports: dict[str, Any]) -> dict[str, Any]:
    return {name: payload["metrics"] for name, payload in reports.items()}


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Planner evaluation",
        "",
        report.get("note", ""),
        "",
        f"Held-out resolutions: {', '.join(str(item) for item in report.get('held_out_resolutions', []))}",
        "",
    ]
    diagnostic = report.get("n_mod_8_diagnostic") or {}
    if diagnostic:
        separates = diagnostic.get("separates_oracle")
        lines.extend(
            [
                f"N % 8 values: {diagnostic.get('unique_n_mod_8')}.",
                f"Splits oracle labels: {separates}.",
                "",
            ]
        )
    for split in ("calibration", "held_out", "all"):
        block = report.get(split) or {}
        lines.extend([f"## {split} (n={block.get('n', 0)})", ""])
        lines.append(
            "| Policy | Match | False-repair | False-direct | Geomean regret | Max regret | Gates |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for name, metrics in (block.get("policies") or {}).items():
            match = metrics.get("oracle_match_rate")
            fr = metrics.get("false_repair_rate")
            fd = metrics.get("false_direct_rate")
            geo = metrics.get("geomean_regret")
            mx = metrics.get("max_regret")
            gates = []
            if name == "cost_model":
                gates.append("FR0" if metrics.get("pass_false_repair_zero") else "FR")
                gates.append("geo" if metrics.get("pass_geomean_gate") else "geo-fail")
                gates.append("max" if metrics.get("pass_max_regret_gate") else "max-fail")
            lines.append(
                f"| {name} | {_pct(match)} | {_pct(fr)} | {_pct(fd)} | "
                f"{_num(geo)} | {_num(mx)} | {', '.join(gates) or '—'} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.1f}%"


def _num(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"
