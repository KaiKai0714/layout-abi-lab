"""Correctness canary and full-module CUDA-event candidate selection."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn

from .pattern import CORRECTNESS_TOLERANCE


def correctness_report(
    value: torch.Tensor,
    reference: torch.Tensor,
    tolerance: float = CORRECTNESS_TOLERANCE,
) -> dict[str, Any]:
    value_float = value.detach().float()
    reference_float = reference.detach().float()
    max_abs = float((value_float - reference_float).abs().max())
    denominator = max(float(reference_float.abs().max()), 1e-12)
    relative_inf = max_abs / denominator
    return {
        "max_abs": max_abs,
        "relative_inf": relative_inf,
        "tolerance": tolerance,
        "pass": bool(max_abs <= tolerance and relative_inf <= tolerance),
    }


def _measure_cuda(fn: Callable[[], Any], iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / iterations


def run_canary(
    original: nn.Module,
    candidates: dict[str, nn.Module],
    example_inputs: tuple[Any, ...],
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    original.eval()
    with torch.no_grad():
        reference = original(*example_inputs)
        for name, module in candidates.items():
            module.eval()
            try:
                value = module(*example_inputs)
                reports[name] = correctness_report(value, reference)
            except Exception as exc:
                reports[name] = {
                    "pass": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
    return reports


def autotune(
    candidates: dict[str, nn.Module],
    example_inputs: tuple[Any, ...],
    *,
    warmup: int = 8,
    iterations: int = 20,
    cycles: int = 3,
) -> dict[str, Any]:
    """Return medians and the fastest correctness-passing candidate."""

    names = list(candidates)
    order_log: list[list[str]] = []
    samples: dict[str, list[float]] = {name: [] for name in names}
    with torch.no_grad():
        for module in candidates.values():
            module.eval()
            for _ in range(warmup):
                module(*example_inputs)
        torch.cuda.synchronize()
        for cycle in range(cycles):
            rotated = names[cycle % len(names) :] + names[: cycle % len(names)]
            order_log.append(rotated)
            for name in rotated:
                module = candidates[name]
                samples[name].append(
                    _measure_cuda(lambda selected=module: selected(*example_inputs), iterations)
                )
    latencies = {
        name: {
            "samples_ms": values,
            "median_ms": sorted(values)[len(values) // 2] if values else None,
        }
        for name, values in samples.items()
    }
    ranked = sorted(
        (name for name in names if latencies[name]["median_ms"] is not None),
        key=lambda name: latencies[name]["median_ms"],
    )
    return {
        "warmup": warmup,
        "iterations": iterations,
        "cycles": cycles,
        "order": order_log,
        "latencies": latencies,
        "selected": ranked[0] if ranked else None,
    }
