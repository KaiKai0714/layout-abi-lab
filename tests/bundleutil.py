"""Shared CPU-only result-bundle fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_bundle(
    bundle: Path,
    *,
    device: str = "Test GPU",
    compute_capability: str = "9.9",
    torch_version: str = "test-torch",
    cuda_build: str = "test-cuda",
    cudnn: int = 1,
    resolution: int = 64,
    dtype: str = "fp16",
    batch: int = 1,
    direct_ms: float = 2.0,
    repair_kv_ms: float = 1.0,
    include_compile: bool = False,
    extra_environment: dict[str, Any] | None = None,
    measurement: dict[str, Any] | None = None,
) -> Path:
    bundle.mkdir(parents=True, exist_ok=True)
    environment: dict[str, Any] = {
        "schema": "layoutabi_environment_v1",
        "python": "test-python",
        "nvidia_smi": {"stdout": f"{device}, test-driver"},
        "torch": {
            "version": torch_version,
            "cuda_build": cuda_build,
            "cudnn": cudnn,
            "devices": [{"name": device, "compute_capability": compute_capability}],
        },
    }
    if extra_environment:
        environment.update(extra_environment)
    write_json(bundle / "environment.json", environment)

    policy_stats = {
        "direct": {"median_ms": direct_ms},
        "repair_k": {"median_ms": (direct_ms + repair_kv_ms) / 2.0},
        "repair_kv": {"median_ms": repair_kv_ms},
        "direct_over_repair_kv": direct_ms / repair_kv_ms,
        "repair_kv_wins_all_seeds": repair_kv_ms < direct_ms,
    }
    correctness = {policy: {"pass": True} for policy in ("direct", "repair_k", "repair_kv")}
    eager: dict[str, Any] = {
        "schema": "layoutabi_eager_v1",
        "points": [
            {
                "resolution": resolution,
                "consumer_n": resolution * resolution + 4,
                "dtype": dtype,
                "batch": batch,
                "seeds": [
                    {
                        "seed": 1,
                        "correctness": {"chain": correctness, "module": correctness},
                    }
                ],
                "aggregate": {"chain": policy_stats, "module": policy_stats},
            }
        ],
    }
    if measurement is not None:
        eager["measurement"] = measurement
    write_json(bundle / "eager_results.json", eager)
    (bundle / "SUMMARY.md").write_text("# Synthetic result\n", encoding="utf-8")
    measured = ["environment.json", "eager_results.json", "SUMMARY.md"]
    if include_compile:
        write_json(
            bundle / "compile_results.json",
            {
                "schema": "layoutabi_compile_v1",
                "points": [
                    {
                        "resolution": resolution,
                        "policies": {
                            "direct": {
                                "available": True,
                                "steady_state": {"median_ms": direct_ms},
                                "correctness": {"pass": True},
                            },
                            "repair_kv": {
                                "available": True,
                                "steady_state": {"median_ms": repair_kv_ms},
                                "correctness": {"pass": True},
                            },
                        },
                        "direct_over_repair_kv": direct_ms / repair_kv_ms,
                    }
                ],
            },
        )
        measured.append("compile_results.json")
    write_json(
        bundle / "manifest.json",
        {
            "schema": "layoutabi_manifest_v1",
            "files": {name: sha256(bundle / name) for name in measured},
        },
    )
    return bundle
