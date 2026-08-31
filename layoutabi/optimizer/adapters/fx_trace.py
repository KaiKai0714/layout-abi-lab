"""torch.fx.symbolic_trace adapter.

Importing this module does not require PyTorch. Capture failures stay here.
"""

from __future__ import annotations

from typing import Any


def symbolic_trace_module(model: Any, example_inputs: tuple[Any, ...]) -> Any:
    import torch

    graph_module = torch.fx.symbolic_trace(model)
    graph_module.eval()
    with torch.no_grad():
        graph_module(*example_inputs)
    return graph_module
