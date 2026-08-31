"""Public FX/export capture adapters.

Framework-private failures stay inside these helpers so they cannot break
`import layoutabi`.
"""

from __future__ import annotations

from typing import Any


def try_symbolic_trace(model: Any, example_inputs: tuple[Any, ...]) -> Any:
    from .fx_trace import symbolic_trace_module

    return symbolic_trace_module(model, example_inputs)


def try_export(model: Any, example_inputs: tuple[Any, ...]) -> Any:
    from .export_graph import export_module

    return export_module(model, example_inputs)
