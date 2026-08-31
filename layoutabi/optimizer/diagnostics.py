"""Structured optimizer diagnostics for inspect and optimize results."""

from __future__ import annotations

from typing import Any

from ..schema import DIAGNOSTICS_SCHEMA, current_version
from .pattern import CANDIDATE_IMPL_HASH, PATTERN_ID

DIAGNOSTICS_SCHEMA_VERSION = current_version(DIAGNOSTICS_SCHEMA)


def empty_diagnostics(**fields: Any) -> dict[str, Any]:
    payload = {
        "schema": DIAGNOSTICS_SCHEMA,
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "pattern_id": PATTERN_ID,
        "candidate_impl": CANDIDATE_IMPL_HASH,
        "capture_method": None,
        "original_fingerprint": None,
        "rewrite_fingerprint": None,
        "matches": [],
        "rejections": [],
        "guard_problems": [],
        "candidate_correctness": {},
        "autotune": None,
        "cache": {"key": None, "hit": False, "miss": True},
        "decision": "noop",
        "reason": "unsupported",
        "framework": {},
    }
    payload.update(fields)
    payload["schema"] = DIAGNOSTICS_SCHEMA
    payload["schema_version"] = DIAGNOSTICS_SCHEMA_VERSION
    return payload


def validate_diagnostics(payload: dict[str, Any]) -> list[str]:
    """JSON-Schema-validate a public diagnostics object."""

    from ..schema import normalize_document

    _migrated, problems = normalize_document(payload, DIAGNOSTICS_SCHEMA)
    return problems


def framework_fingerprint() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"torch_available": False}
    info: dict[str, Any] = {
        "torch_available": True,
        "version": torch.__version__,
        "cuda_build": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "compile_available": hasattr(torch, "compile"),
    }
    if torch.cuda.is_available():
        info["device"] = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        info["compute_capability"] = f"{capability[0]}.{capability[1]}"
    return info
