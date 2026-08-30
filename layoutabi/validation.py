"""Validate result completeness, correctness gates, and integrity hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_result(directory: Path, strict: bool = False) -> list[str]:
    """Return human-readable validation problems; an empty list means valid."""

    problems: list[str] = []
    required = ("environment.json", "eager_results.json", "SUMMARY.md", "manifest.json")
    for name in required:
        if not (directory / name).is_file():
            problems.append(f"Missing required file: {name}")
    if problems:
        return problems

    try:
        environment = _load(directory / "environment.json")
        eager = _load(directory / "eager_results.json")
        manifest = _load(directory / "manifest.json")
    except Exception as exc:
        return [f"Could not parse result bundle: {exc}"]

    if environment.get("schema") != "layoutabi_environment_v1":
        problems.append("Unsupported environment schema")
    if eager.get("schema") != "layoutabi_eager_v1":
        problems.append("Unsupported eager result schema")
    if manifest.get("schema") != "layoutabi_manifest_v1":
        problems.append("Unsupported manifest schema")

    points = eager.get("points")
    if not isinstance(points, list) or not points:
        problems.append("Eager results contain no measurement points")
    else:
        for point in points:
            resolution = point.get("resolution", "unknown")
            for seed in point.get("seeds", []):
                for scope in ("chain", "module"):
                    for policy, cell in seed.get("correctness", {}).get(scope, {}).items():
                        if not cell.get("pass", False):
                            problems.append(
                                f"Correctness failed at resolution={resolution}, "
                                f"seed={seed.get('seed')}, scope={scope}, policy={policy}"
                            )

    files = manifest.get("files", {})
    if not isinstance(files, dict):
        problems.append("Manifest files field is not an object")
    else:
        for name, expected in files.items():
            path = directory / name
            if not path.is_file():
                problems.append(f"Manifest references a missing file: {name}")
            elif _sha256(path) != expected:
                problems.append(f"Checksum mismatch: {name}")

    compile_path = directory / "compile_results.json"
    if compile_path.is_file():
        try:
            compiled = _load(compile_path)
            if compiled.get("schema") != "layoutabi_compile_v1":
                problems.append("Unsupported compiled result schema")
            for point in compiled.get("points", []):
                for policy, cell in point.get("policies", {}).items():
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
        except Exception as exc:
            problems.append(f"Could not parse compile_results.json: {exc}")
    elif strict:
        problems.append("Strict validation requires compile_results.json")

    return problems

