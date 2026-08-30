"""Discover optimizer cases from JSON specs so new graphs can be dropped in."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from .synthetic import synthetic_cells

_CASES_DIR = Path(__file__).resolve().parent / "cases"
_REQUIRED = (
    "id",
    "role",
    "title",
    "license",
    "graph_fingerprint",
    "expected_optimizer",
)
_ROLES = {"positive_reference", "public", "negative", "experimental"}
_EXPECT = {"match", "noop"}


def _validate_spec(spec: dict[str, Any], path: Path) -> None:
    missing = [key for key in _REQUIRED if key not in spec]
    if missing:
        raise ValueError(f"{path.name} missing fields: {', '.join(missing)}")
    if spec["id"] != path.stem:
        raise ValueError(f"{path.name} id {spec['id']!r} must match the filename stem")
    if spec["role"] not in _ROLES:
        raise ValueError(f"{path.name} has unknown role {spec['role']!r}")
    if spec["expected_optimizer"] not in _EXPECT:
        raise ValueError(
            f"{path.name} has unknown expected_optimizer {spec['expected_optimizer']!r}"
        )
    builder = path.with_suffix(".py")
    if not builder.is_file():
        raise ValueError(f"{path.name} has no builder {builder.name}")


def load_catalog() -> dict[str, dict[str, Any]]:
    """Load every case spec. Does not import PyTorch."""

    catalog: dict[str, dict[str, Any]] = {}
    for path in sorted(_CASES_DIR.glob("*.json")):
        spec = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(spec, dict):
            raise ValueError(f"{path.name} must contain a JSON object")
        _validate_spec(spec, path)
        if spec["id"] in catalog:
            raise ValueError(f"Duplicate workload id {spec['id']!r}")
        catalog[spec["id"]] = dict(spec)
    return catalog


def list_workloads() -> list[dict[str, Any]]:
    return list(load_catalog().values())


def get_workload(workload_id: str) -> dict[str, Any]:
    catalog = load_catalog()
    try:
        return dict(catalog[workload_id])
    except KeyError as exc:
        known = ", ".join(sorted(catalog))
        raise ValueError(f"Unknown workload {workload_id!r}; expected one of {known}") from exc


def _load_builder(workload_id: str) -> Any:
    module = importlib.import_module(f"{__package__}.cases.{workload_id}")
    builder = getattr(module, "build", None)
    if builder is None:
        raise ValueError(f"Workload {workload_id!r} has no build() function")
    return module, builder


def make_workload(
    workload_id: str,
    *,
    resolution: int = 128,
    batch: int = 1,
    dtype: str = "fp16",
    device: str = "cpu",
) -> tuple[Any, tuple[Any, ...]]:
    """Build an eval module and example inputs for a named workload."""

    spec = get_workload(workload_id)
    if resolution <= 0 or batch <= 0:
        raise ValueError("resolution and batch must be positive")
    from ._runtime import place, torch_dtype

    torch_dtype_value = torch_dtype(dtype)
    _module, builder = _load_builder(spec["id"])
    module, tensors = builder(
        resolution=resolution, batch=batch, dtype=torch_dtype_value
    )
    return place(module, tensors, device)


# Backward-compatible alias used by earlier v0.5 tests.
CATALOG = None


def __getattr__(name: str) -> Any:
    if name == "CATALOG":
        return load_catalog()
    raise AttributeError(name)
