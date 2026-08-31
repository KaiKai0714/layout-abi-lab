"""Public optimizer API: inspect and optimize with conservative fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import nn

from ..errors import InvalidArgumentError, MissingPyTorchError
from .autotune import autotune, run_canary
from .cache import cache_identity, load_entry, make_cache_key, store_entry
from .candidates import build_candidates
from .capture import capture_graph, normalize_inputs
from .diagnostics import empty_diagnostics, framework_fingerprint
from .fxutil import graph_fingerprint
from .guards import input_guard_problems, match_guard_problems, require_cuda
from .matcher import match_graph
from .pattern import PATTERN_ID, REWRITE_POLICIES, SUPPORTED_POLICIES
from ..planner.policies import LIVE_PLANNER_POLICIES


@dataclass
class OptimizeResult:
    module: nn.Module
    decision: str
    diagnostics: dict[str, Any]


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise MissingPyTorchError("layoutabi.optimize") from exc
    return torch


def _public_diagnostics(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if not key.startswith("_")}


def _inspect_impl(model: nn.Module, example_inputs: Any) -> dict[str, Any]:
    diagnostics = empty_diagnostics(framework=framework_fingerprint())
    inputs = normalize_inputs(example_inputs)
    diagnostics["guard_problems"] = input_guard_problems(model, inputs)
    captured = capture_graph(model, inputs)
    diagnostics["capture_method"] = captured.method
    diagnostics["original_fingerprint"] = captured.fingerprint
    diagnostics["capture_notes"] = captured.notes
    matched = match_graph(captured.graph_module)
    diagnostics["rejections"] = list(matched.rejections)
    accepted = []
    for item in matched.matches:
        problems = match_guard_problems(captured.graph_module, item, inputs)
        if problems:
            diagnostics["rejections"].append(
                {"location": item.location, "reason": "; ".join(problems)}
            )
            continue
        accepted.append(item)
    diagnostics["matches"] = [item.__dict__ for item in accepted]
    if diagnostics["guard_problems"]:
        diagnostics["reason"] = "guard_failed"
    elif not accepted:
        diagnostics["reason"] = "no_supported_pattern"
    else:
        diagnostics["decision"] = "inspect"
        diagnostics["reason"] = f"matched {PATTERN_ID}"
    diagnostics["_accepted"] = accepted
    diagnostics["_captured"] = captured
    diagnostics["_inputs"] = inputs
    return diagnostics


def inspect(model: nn.Module, example_inputs: Any) -> dict[str, Any]:
    """Capture and match without rewriting.

    Returns a diagnostics dict with schema ``layoutabi_optimizer_diagnostics_v1``.
    Unsupported graphs stay inspectable; capture failures are recorded rather
    than raised.
    """

    _require_torch()
    try:
        return _public_diagnostics(_inspect_impl(model, example_inputs))
    except Exception as exc:
        return empty_diagnostics(
            framework=framework_fingerprint(),
            reason="exception",
            error=f"{type(exc).__name__}: {exc}",
        )


def optimize(
    model: nn.Module,
    example_inputs: Any,
    policy: str = "autotune",
    compile: bool = False,
    cache_dir: str | Path | None = None,
    shape_mode: str = "exact",
    unseen_shape: str = "direct",
    allow_sync_autotune: bool = True,
) -> OptimizeResult:
    """Rewrite a supported inference graph or return the original module.

    Unknown ``policy``, ``shape_mode``, or ``unseen_shape`` values raise
    ``InvalidArgumentError``. Every other failure keeps ``model`` unchanged.
    """

    _require_torch()
    if policy not in SUPPORTED_POLICIES:
        raise InvalidArgumentError(
            f"Unknown policy {policy!r}; expected one of {SUPPORTED_POLICIES}"
        )
    if shape_mode not in {"exact", "bucket"}:
        raise InvalidArgumentError("shape_mode must be 'exact' or 'bucket'")
    if unseen_shape not in {"direct", "noop", "autotune"}:
        raise InvalidArgumentError("unseen_shape must be 'direct', 'noop', or 'autotune'")
    diagnostics = empty_diagnostics(framework=framework_fingerprint(), policy=policy)
    try:
        return _optimize_inner(
            model,
            example_inputs,
            policy=policy,
            compile=compile,
            cache_dir=Path(cache_dir) if cache_dir is not None else None,
            diagnostics=diagnostics,
            shape_mode=shape_mode,
            unseen_shape=unseen_shape,
            allow_sync_autotune=allow_sync_autotune,
        )
    except Exception as exc:
        diagnostics["decision"] = "noop"
        diagnostics["reason"] = "exception"
        diagnostics["error"] = f"{type(exc).__name__}: {exc}"
        return OptimizeResult(module=model, decision="noop", diagnostics=diagnostics)


def _optimize_inner(
    original: nn.Module,
    example_inputs: Any,
    *,
    policy: str,
    compile: bool,
    cache_dir: Path | None,
    diagnostics: dict[str, Any],
    shape_mode: str = "exact",
    unseen_shape: str = "direct",
    allow_sync_autotune: bool = True,
) -> OptimizeResult:
    import time

    started = time.perf_counter()
    inspected = _inspect_impl(original, example_inputs)
    diagnostics.update(_public_diagnostics(inspected))
    diagnostics["timings_ms"] = {
        "capture": (time.perf_counter() - started) * 1000.0,
        "autotune": None,
        "compile": None,
        "steady_state": None,
    }
    if policy == "off":
        diagnostics["decision"] = "off"
        diagnostics["reason"] = "user_policy"
        return _finish(original, "off", diagnostics, compile=compile, inputs=inspected["_inputs"])

    if inspected["guard_problems"]:
        diagnostics["decision"] = "noop"
        diagnostics["reason"] = "guard_failed"
        return OptimizeResult(module=original, decision="noop", diagnostics=diagnostics)

    accepted = inspected["_accepted"]
    captured = inspected["_captured"]
    inputs = inspected["_inputs"]
    if not accepted:
        diagnostics["decision"] = "noop"
        diagnostics["reason"] = inspected.get("reason", "no_supported_pattern")
        return OptimizeResult(module=original, decision="noop", diagnostics=diagnostics)

    identity = cache_identity(
        graph_fingerprint=captured.fingerprint,
        example_inputs=inputs,
        torch_info=diagnostics.get("framework", {}),
        shape_mode=shape_mode,
    )
    cache_key = make_cache_key(identity)
    diagnostics["cache"] = {
        "key": cache_key,
        "hit": False,
        "miss": True,
        "identity": identity,
        "shape_mode": shape_mode,
    }
    if shape_mode == "bucket" and not identity.get("shape_in_range"):
        diagnostics["decision"] = "noop" if unseen_shape == "noop" else "direct"
        diagnostics["reason"] = "unseen_shape"
        if unseen_shape == "noop":
            return OptimizeResult(module=original, decision="noop", diagnostics=diagnostics)
        if unseen_shape != "autotune":
            return _finish(
                original, "direct", diagnostics, compile=compile, inputs=inputs
            )

    selected_policy = policy
    diagnostics["planner"] = None
    if policy in LIVE_PLANNER_POLICIES:
        from ..planner.features import features_from_live
        from ..planner.policies import decide as planner_decide

        features = features_from_live(
            inputs, matches=accepted, graph_module=captured.graph_module
        )
        planned = planner_decide(features, policy)
        diagnostics["planner"] = {"features": features.as_dict(), **planned}
        selected_policy = planned["action"]
        diagnostics["reason"] = planned["reason"]

    cache_hit = False
    if selected_policy == "autotune":
        cached = load_entry(cache_dir, cache_key)
        cached_decision = cached.get("decision") if cached else None
        if cached_decision in REWRITE_POLICIES:
            selected_policy = cached_decision
            cache_hit = True
            diagnostics["cache"]["hit"] = True
            diagnostics["cache"]["miss"] = False
            diagnostics["reason"] = "cache_hit"

    if selected_policy == "autotune" and not allow_sync_autotune:
        fallback = "noop" if unseen_shape == "noop" else "direct"
        diagnostics["reason"] = "sync_autotune_disabled"
        diagnostics["decision"] = fallback
        if fallback == "noop":
            return OptimizeResult(module=original, decision="noop", diagnostics=diagnostics)
        selected_policy = "direct"

    if selected_policy == "autotune":
        cuda_problems = require_cuda(inputs)
        if cuda_problems:
            diagnostics["decision"] = "noop"
            diagnostics["reason"] = cuda_problems[0]
            diagnostics["guard_problems"] = (
                list(diagnostics["guard_problems"]) + cuda_problems
            )
            return OptimizeResult(module=original, decision="noop", diagnostics=diagnostics)
        build_policy = "autotune"
    else:
        build_policy = selected_policy

    candidates = build_candidates(captured.graph_module, accepted, build_policy)
    canary = run_canary(original, candidates, inputs)
    diagnostics["candidate_correctness"] = canary
    passing = {
        name: module
        for name, module in candidates.items()
        if canary.get(name, {}).get("pass")
    }
    if not passing:
        diagnostics["decision"] = "noop"
        diagnostics["reason"] = "correctness_failed"
        return OptimizeResult(module=original, decision="noop", diagnostics=diagnostics)

    if selected_policy == "autotune":
        tune_started = time.perf_counter()
        tune = autotune(passing, inputs)
        diagnostics["timings_ms"]["autotune"] = (time.perf_counter() - tune_started) * 1000.0
        diagnostics["autotune"] = {key: value for key, value in tune.items() if key != "selected"}
        selected = tune["selected"]
        if selected is None or selected not in passing:
            diagnostics["decision"] = "noop"
            diagnostics["reason"] = "autotune_failed"
            return OptimizeResult(module=original, decision="noop", diagnostics=diagnostics)
        diagnostics["reason"] = "autotune_fastest"
        diagnostics["break_even_invocations"] = _break_even_invocations(
            diagnostics["timings_ms"].get("autotune"),
            diagnostics["autotune"].get("latencies") or {},
        )
        store_entry(
            cache_dir,
            cache_key,
            {
                "decision": selected,
                "autotune": diagnostics["autotune"],
                "identity": identity,
            },
        )
    else:
        if selected_policy not in passing:
            diagnostics["decision"] = "noop"
            diagnostics["reason"] = (
                "cache_hit_failed_canary" if cache_hit else "correctness_failed"
            )
            return OptimizeResult(module=original, decision="noop", diagnostics=diagnostics)
        selected = selected_policy
        if not cache_hit and policy in REWRITE_POLICIES:
            diagnostics["reason"] = "user_policy"

    chosen = original if selected == "direct" else passing[selected]
    diagnostics["decision"] = selected
    diagnostics["rewrite_fingerprint"] = (
        captured.fingerprint if selected == "direct" else graph_fingerprint(chosen)
    )
    return _finish(chosen, selected, diagnostics, compile=compile, inputs=inputs)


def _break_even_invocations(autotune_ms: float | None, latencies: dict[str, Any]) -> int | None:
    if autotune_ms is None:
        return None
    medians = {
        name: cell.get("median_ms")
        for name, cell in latencies.items()
        if isinstance(cell, dict) and isinstance(cell.get("median_ms"), (int, float))
    }
    direct = medians.get("direct")
    if direct is None or not medians:
        return None
    best = min(medians.values())
    savings = direct - best
    if savings <= 0:
        return None
    import math

    return int(math.ceil(autotune_ms / savings))


def _finish(
    module: nn.Module,
    decision: str,
    diagnostics: dict[str, Any],
    *,
    compile: bool,
    inputs: tuple[Any, ...],
) -> OptimizeResult:
    import time

    if not compile:
        return OptimizeResult(module=module, decision=decision, diagnostics=diagnostics)
    try:
        import torch

        if not hasattr(torch, "compile"):
            diagnostics["compile"] = {
                "applied": False,
                "reason": "torch.compile unavailable",
            }
            return OptimizeResult(module=module, decision=decision, diagnostics=diagnostics)
        compile_started = time.perf_counter()
        compiled = torch.compile(module)
        compiled.eval()
        with torch.no_grad():
            compiled(*inputs)
        if diagnostics.get("timings_ms") is not None:
            diagnostics["timings_ms"]["compile"] = (
                time.perf_counter() - compile_started
            ) * 1000.0
        diagnostics["compile"] = {"applied": True}
        return OptimizeResult(module=compiled, decision=decision, diagnostics=diagnostics)
    except Exception as exc:
        diagnostics["compile"] = {
            "applied": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
        return OptimizeResult(module=module, decision=decision, diagnostics=diagnostics)
