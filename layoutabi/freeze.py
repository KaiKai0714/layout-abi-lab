"""Release-candidate freeze snapshot. Importing this module does not require PyTorch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import CURRENT_VERSIONS

FREEZE_SCHEMA = "layoutabi_rc_freeze_v1"
FREEZE_PATH = Path(__file__).resolve().parent / "schemas" / "rc_freeze.json"

# Honest v1.0 gates. Open items are not claimed complete.
GATES = (
    {
        "id": "cpu_ci",
        "status": "verified_by_ci",
        "detail": "Package import, schema, index, and unit tests on CPU.",
    },
    {
        "id": "l40s_reference",
        "status": "published",
        "detail": "Checksum-validated L40S reference bundles remain in results/reference_l40s.",
    },
    {
        "id": "public_workloads",
        "status": "verified_by_ci",
        "detail": "Two independent public match graphs and one public SDPA no-op.",
    },
    {
        "id": "software_stacks",
        "status": "published",
        "detail": "Three L40S PyTorch/CUDA stacks. Stacks are not extra devices.",
    },
    {
        "id": "cache_concurrency",
        "status": "verified_by_ci",
        "detail": "Locked cache, corruption recovery, and unseen-shape tests.",
    },
    {
        "id": "license_privacy_scan",
        "status": "verified_by_ci",
        "detail": "layoutabi scan-release on git-tracked files.",
    },
    {
        "id": "orin",
        "status": "published",
        "detail": "Eager 128 FP16 community bundle; compiled unavailable; 256 not measured.",
    },
    {
        "id": "second_architecture",
        "status": "published",
        "detail": "Orin is a second architecture: eager-128 direct-win, not an L40S speedup replica.",
    },
    {
        "id": "three_level_residue_contrast",
        "status": "published",
        "detail": (
            "L40S PyTorch 2.11/CUDA 12.8 bundle spans N%8==0, even non-multiple, "
            "and odd with isolated KTV profiler names."
        ),
    },
    {
        "id": "operand_pointer_contrast",
        "status": "open",
        "detail": (
            "L40S 100-cell K-pointer by V-pointer grid is published; Orin remains open."
        ),
    },
)


def _public_surface() -> tuple[list[str], list[str]]:
    import layoutabi

    names = [name for name in layoutabi.__all__ if name != "__version__"]
    exceptions = [
        name for name in names if isinstance(getattr(layoutabi, name), type)
    ]
    functions = [name for name in names if name not in exceptions]
    return sorted(functions), sorted(exceptions)


def live_freeze() -> dict[str, Any]:
    """Return the freeze snapshot taken from this install."""

    from .optimizer.cache import CACHE_PROTOCOL, CACHE_SCHEMA
    from .optimizer.pattern import (
        CANDIDATE_IMPL_HASH,
        CORRECTNESS_TOLERANCE,
        PATTERN_ID,
        SUPPORTED_POLICIES,
    )
    from .optimizer.shapes import SHAPE_BUCKETS, UNSEEN_SHAPE_ACTIONS

    public_api, exceptions = _public_surface()
    return {
        "schema": FREEZE_SCHEMA,
        "schema_version": 1,
        "allow_new_patterns": False,
        "cache_protocol": CACHE_PROTOCOL,
        "cache_schema": CACHE_SCHEMA,
        "candidate_impl": CANDIDATE_IMPL_HASH,
        "correctness_tolerance": CORRECTNESS_TOLERANCE,
        "document_schemas": dict(sorted(CURRENT_VERSIONS.items())),
        "exceptions": exceptions,
        "pattern_id": PATTERN_ID,
        "policies": list(SUPPORTED_POLICIES),
        "public_api": public_api,
        "shape_buckets": list(SHAPE_BUCKETS),
        "unseen_shape_actions": list(UNSEEN_SHAPE_ACTIONS),
    }


def load_frozen() -> dict[str, Any]:
    payload = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{FREEZE_PATH} must contain a JSON object")
    return payload


def freeze_problems(live: dict[str, Any] | None = None) -> list[str]:
    """Return mismatches between the live snapshot and the frozen file."""

    frozen = load_frozen()
    current = live if live is not None else live_freeze()
    problems: list[str] = []
    frozen_keys = set(frozen)
    live_keys = set(current)
    for key in sorted(frozen_keys - live_keys):
        problems.append(f"frozen key missing from live snapshot: {key}")
    for key in sorted(live_keys - frozen_keys):
        problems.append(f"live snapshot has unfrozen key: {key}")
    for key in sorted(frozen_keys & live_keys):
        if frozen[key] != current[key]:
            problems.append(f"{key}: frozen {frozen[key]!r} != live {current[key]!r}")
    if current.get("allow_new_patterns") is not False:
        problems.append("allow_new_patterns must stay false during the RC freeze")
    return problems


def rc_status() -> dict[str, Any]:
    from . import __version__

    problems = freeze_problems()
    open_gates = [item for item in GATES if item["status"] == "open"]
    return {
        "release_status": "candidate",
        "version": __version__,
        "freeze_ok": not problems,
        "freeze_problems": problems,
        "gates": [dict(item) for item in GATES],
        "v1_open_gates": [item["id"] for item in open_gates],
        "policy": "Bug fixes only. New matcher patterns require a new RC.",
    }
