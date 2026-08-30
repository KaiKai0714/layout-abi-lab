"""Public optimizer API: inspect and optimize with conservative fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import nn

from .autotune import autotune, run_canary
from .cache import cache_identity, load_entry, make_cache_key, store_entry
from .candidates import build_candidates
from .capture import capture_graph, normalize_inputs
from .diagnostics import empty_diagnostics, framework_fingerprint
from .fxutil import graph_fingerprint
from .guards import input_guard_problems, match_guard_problems, require_cuda
from .matcher import match_graph
from .pattern import PATTERN_ID, REWRITE_POLICIES, SUPPORTED_POLICIES


@dataclass
class OptimizeResult:
    module: nn.Module
    decision: str
    diagnostics: dict[str, Any]


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "layoutabi.optimize requires PyTorch. Install a CUDA-enabled build "
            "appropriate for this GPU; this package does not install PyTorch."
        ) from exc
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
    """Capture and match without rewriting. Unsupported graphs stay inspectable."""

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
) -> OptimizeResult:
    """Rewrite a supported inference graph or return the original module."""

    _require_torch()
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(
            f"Unknown policy {policy!r}; expected one of {SUPPORTED_POLICIES}"
        )
    diagnostics = empty_diagnostics(framework=framework_fingerprint(), policy=policy)
    try:
        return _optimize_inner(
            model,
            example_inputs,
            policy=policy,
            compile=compile,
            cache_dir=Path(cache_dir) if cache_dir is not None else None,
            diagnostics=diagnostics,
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
) -> OptimizeResult:
    inspected = _inspect_impl(original, example_inputs)
    diagnostics.update(_public_diagnostics(inspected))
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
    )
    cache_key = make_cache_key(identity)
    diagnostics["cache"] = {
        "key": cache_key,
        "hit": False,
        "miss": True,
        "identity": identity,
    }

    selected_policy = policy
    cache_hit = False
    if policy == "autotune":
        cached = load_entry(cache_dir, cache_key)
        cached_decision = cached.get("decision") if cached else None
        if cached_decision in REWRITE_POLICIES:
            selected_policy = cached_decision
            cache_hit = True
            diagnostics["cache"]["hit"] = True
            diagnostics["cache"]["miss"] = False
            diagnostics["reason"] = "cache_hit"

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
        tune = autotune(passing, inputs)
        diagnostics["autotune"] = {key: value for key, value in tune.items() if key != "selected"}
        selected = tune["selected"]
        if selected is None or selected not in passing:
            diagnostics["decision"] = "noop"
            diagnostics["reason"] = "autotune_failed"
            return OptimizeResult(module=original, decision="noop", diagnostics=diagnostics)
        diagnostics["reason"] = "autotune_fastest"
        store_entry(
            cache_dir,
            cache_key,
            {"decision": selected, "autotune": diagnostics["autotune"]},
        )
    else:
        if selected_policy not in passing:
            diagnostics["decision"] = "noop"
            diagnostics["reason"] = (
                "cache_hit_failed_canary" if cache_hit else "correctness_failed"
            )
            return OptimizeResult(module=original, decision="noop", diagnostics=diagnostics)
        selected = selected_policy
        if not cache_hit:
            diagnostics["reason"] = "user_policy"

    chosen = original if selected == "direct" else passing[selected]
    diagnostics["decision"] = selected
    diagnostics["rewrite_fingerprint"] = (
        captured.fingerprint if selected == "direct" else graph_fingerprint(chosen)
    )
    return _finish(chosen, selected, diagnostics, compile=compile, inputs=inputs)


def _finish(
    module: nn.Module,
    decision: str,
    diagnostics: dict[str, Any],
    *,
    compile: bool,
    inputs: tuple[Any, ...],
) -> OptimizeResult:
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
        compiled = torch.compile(module)
        compiled.eval()
        with torch.no_grad():
            compiled(*inputs)
        diagnostics["compile"] = {"applied": True}
        return OptimizeResult(module=compiled, decision=decision, diagnostics=diagnostics)
    except Exception as exc:
        diagnostics["compile"] = {
            "applied": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
        return OptimizeResult(module=module, decision=decision, diagnostics=diagnostics)
