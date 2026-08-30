"""Validate result completeness, correctness gates, schemas, and integrity hashes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import (
    COMPILE_SCHEMA,
    EAGER_SCHEMA,
    ENVIRONMENT_SCHEMA,
    MANIFEST_SCHEMA,
    load_json_object,
    normalize_document,
    sha256_file,
)


def _label(kind: str, problem: str) -> str:
    if problem.startswith("Unsupported schema"):
        return f"Unsupported {kind} schema"
    return f"{kind}: {problem}"


def _normalize_file(
    path: Path, expected_schema: str, kind: str
) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        payload = load_json_object(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, [f"Could not parse {path.name}: {exc}"]
    migrated, problems = normalize_document(payload, expected_schema)
    return migrated, [_label(kind, problem) for problem in problems]


def validate_result(directory: Path, strict: bool = False) -> list[str]:
    """Return human-readable validation problems; an empty list means valid."""

    problems: list[str] = []
    required = ("environment.json", "eager_results.json", "SUMMARY.md", "manifest.json")
    for name in required:
        if not (directory / name).is_file():
            problems.append(f"Missing required file: {name}")
    if problems:
        return problems

    environment, env_problems = _normalize_file(
        directory / "environment.json", ENVIRONMENT_SCHEMA, "environment"
    )
    eager, eager_problems = _normalize_file(
        directory / "eager_results.json", EAGER_SCHEMA, "eager result"
    )
    manifest, manifest_problems = _normalize_file(
        directory / "manifest.json", MANIFEST_SCHEMA, "manifest"
    )
    problems.extend(env_problems)
    problems.extend(eager_problems)
    problems.extend(manifest_problems)

    if eager is not None:
        points = eager.get("points")
        if not isinstance(points, list) or not points:
            problems.append("Eager results contain no measurement points")
        else:
            for point in points:
                if not isinstance(point, dict):
                    problems.append("Eager results contain a non-object measurement point")
                    continue
                resolution = point.get("resolution", "unknown")
                for seed in point.get("seeds", []):
                    if not isinstance(seed, dict):
                        continue
                    for scope in ("chain", "module"):
                        cells = seed.get("correctness", {}).get(scope, {})
                        if not isinstance(cells, dict):
                            continue
                        for policy, cell in cells.items():
                            if isinstance(cell, dict) and not cell.get("pass", False):
                                problems.append(
                                    f"Correctness failed at resolution={resolution}, "
                                    f"seed={seed.get('seed')}, scope={scope}, policy={policy}"
                                )

    if manifest is not None:
        files = manifest.get("files", {})
        if not isinstance(files, dict):
            problems.append("Manifest files field is not an object")
        else:
            for name, expected in files.items():
                path = directory / str(name)
                if not path.is_file():
                    problems.append(f"Manifest references a missing file: {name}")
                elif sha256_file(path) != expected:
                    problems.append(f"Checksum mismatch: {name}")

    compile_path = directory / "compile_results.json"
    if compile_path.is_file():
        compiled, compile_problems = _normalize_file(compile_path, COMPILE_SCHEMA, "compiled result")
        problems.extend(compile_problems)
        if compiled is not None:
            for point in compiled.get("points", []):
                if not isinstance(point, dict):
                    continue
                policies = point.get("policies", {})
                if not isinstance(policies, dict):
                    continue
                for policy, cell in policies.items():
                    if not isinstance(cell, dict):
                        continue
                    if strict and not cell.get("available", False):
                        problems.append(
                            f"Compiled cell unavailable at resolution={point.get('resolution')}, "
                            f"policy={policy}"
                        )
                    if cell.get("available") and not cell.get("correctness", {}).get("pass", False):
                        problems.append(
                            f"Compiled correctness failed at resolution={point.get('resolution')}, "
                            f"policy={policy}"
                        )
    elif strict:
        problems.append("Strict validation requires compile_results.json")

    return problems


def discover_result_bundles(root: Path) -> list[Path]:
    """Discover validator-compatible bundles without treating legacy raw JSON as one."""

    if not root.is_dir():
        return []
    return sorted(path.parent for path in root.rglob("manifest.json") if path.is_file())
