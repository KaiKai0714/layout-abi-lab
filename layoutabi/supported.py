"""Declared support matrix. Importing this module does not require PyTorch."""

from __future__ import annotations

from typing import Any

from .optimizer.pattern import PATTERN_ID, PATTERN_SUMMARY, SUPPORTED_POLICIES
from .optimizer.shapes import SHAPE_BUCKETS, UNSEEN_SHAPE_ACTIONS
from .workloads import list_workloads

# Reference software stacks that completed the public L40S protocol.
REFERENCE_STACKS = (
    {"torch": "2.11.0+cu128", "cuda_build": "12.8", "role": "reference"},
    {"torch": "2.11.0+cu126", "cuda_build": "12.6", "role": "reference"},
    {"torch": "2.10.0+cu128", "cuda_build": "12.8", "role": "reference"},
)

CPU_TOOLS = (
    "validate",
    "validate-tree",
    "aggregate",
    "prepare-submission",
    "migrate-schema",
    "list-workloads",
    "evaluate-planner",
    "cache-info",
    "cache-clear",
    "supported",
    "rc-status",
    "scan-release",
)

OPTIMIZER_APIS = ("optimize", "inspect", "inspect-model", "optimize-model")


def supported(
    *,
    dtype: str | None = None,
    workload: str | None = None,
) -> dict[str, Any]:
    """Return what this release supports. Does not load PyTorch."""

    workloads = list_workloads()
    if workload is not None:
        workloads = [item for item in workloads if item.get("id") == workload]
        if not workloads:
            from .errors import InvalidArgumentError

            raise InvalidArgumentError(f"Unknown workload {workload!r}")
    dtype_ok = dtype is None or str(dtype).lower() in {"fp16", "float16", "torch.float16"}
    payload = {
        "pattern_id": PATTERN_ID,
        "pattern": PATTERN_SUMMARY,
        "policies": list(SUPPORTED_POLICIES),
        "dtype": "float16",
        "inference_only": True,
        "fixed_shapes": True,
        "shape_buckets": list(SHAPE_BUCKETS),
        "unseen_shape_actions": list(UNSEEN_SHAPE_ACTIONS),
        "workloads": workloads,
        "software_stacks": [dict(item) for item in REFERENCE_STACKS],
        "installs_pytorch": False,
        "cpu_tools": list(CPU_TOOLS),
        "optimizer_requires_pytorch": list(OPTIMIZER_APIS),
        "cuda_autotune": True,
        "dtype_requested_supported": dtype_ok,
        "release_status": "released",
        "feature_freeze": True,
        "allow_new_patterns": False,
    }
    if dtype is not None and not dtype_ok:
        payload["notes"] = [
            f"{dtype} is a documented boundary, not a supported optimizer dtype."
        ]
    return payload
