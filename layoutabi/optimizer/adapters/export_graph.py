"""torch.export adapter.

Importing this module does not require PyTorch. Missing or failing export stays
local so it cannot break ``import layoutabi``.
"""

from __future__ import annotations

from typing import Any


def export_module(model: Any, example_inputs: tuple[Any, ...]) -> Any:
    import torch

    if not hasattr(torch, "export"):
        raise RuntimeError("This PyTorch build does not provide torch.export")
    exported = torch.export.export(model, example_inputs)
    graph_module = exported.module()
    graph_module.eval()
    with torch.no_grad():
        graph_module(*example_inputs)
    return graph_module
