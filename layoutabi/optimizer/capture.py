"""Capture a fixed-shape inference graph with public torch.export or FX APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.fx import GraphModule

from .fxutil import graph_fingerprint


class CaptureError(RuntimeError):
    """Raised when no public capture path can produce a runnable graph."""


@dataclass
class CaptureResult:
    graph_module: GraphModule
    method: str
    fingerprint: str
    notes: list[str] = field(default_factory=list)


def normalize_inputs(example_inputs: Any) -> tuple[Any, ...]:
    if isinstance(example_inputs, torch.Tensor):
        return (example_inputs,)
    if isinstance(example_inputs, tuple):
        return example_inputs
    if isinstance(example_inputs, list):
        return tuple(example_inputs)
    raise TypeError("example_inputs must be a tensor, tuple, or list")


def _run_shape_prop(graph_module: GraphModule, example_inputs: tuple[Any, ...]) -> None:
    try:
        from torch.fx.passes.shape_prop import ShapeProp

        ShapeProp(graph_module).propagate(*example_inputs)
    except Exception:
        return


def _symbolic_trace(model: nn.Module, example_inputs: tuple[Any, ...]) -> GraphModule:
    graph_module = torch.fx.symbolic_trace(model)
    graph_module.eval()
    with torch.no_grad():
        graph_module(*example_inputs)
    return graph_module


def _export(model: nn.Module, example_inputs: tuple[Any, ...]) -> GraphModule:
    exported = torch.export.export(model, example_inputs)
    graph_module = exported.module()
    graph_module.eval()
    with torch.no_grad():
        graph_module(*example_inputs)
    return graph_module


def capture_graph(model: nn.Module, example_inputs: tuple[Any, ...]) -> CaptureResult:
    """Return a runnable GraphModule. Prefer FX trace for rewrite stability."""

    notes: list[str] = []
    try:
        graph_module = _symbolic_trace(model, example_inputs)
        _run_shape_prop(graph_module, example_inputs)
        return CaptureResult(
            graph_module=graph_module,
            method="symbolic_trace",
            fingerprint=graph_fingerprint(graph_module),
            notes=notes,
        )
    except Exception as exc:
        notes.append(f"symbolic_trace: {type(exc).__name__}: {exc}")

    if not hasattr(torch, "export"):
        raise CaptureError("; ".join(notes) if notes else "No public capture API succeeded")
    try:
        graph_module = _export(model, example_inputs)
        _run_shape_prop(graph_module, example_inputs)
        notes.append("fell back to torch.export")
        return CaptureResult(
            graph_module=graph_module,
            method="export",
            fingerprint=graph_fingerprint(graph_module),
            notes=notes,
        )
    except Exception as exc:
        notes.append(f"torch.export: {type(exc).__name__}: {exc}")
        raise CaptureError("; ".join(notes)) from exc
