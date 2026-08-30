"""Stable identity keys for result-bundle duplicate and replicate detection."""

from __future__ import annotations

import hashlib
import json
from typing import Any

# All v0.1/v0.2 public bundles reconstruct this audited LinearAttention graph.
DEFAULT_GRAPH_FINGERPRINT = (
    "public_diffusion_linear_attention:"
    "lucidrains/denoising-diffusion-pytorch@"
    "faed4db28e724735323fa91c70aa9b28a6e1cbac"
)


def canonical_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def graph_fingerprint(eager: dict[str, Any]) -> str:
    explicit = eager.get("graph_fingerprint")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    return DEFAULT_GRAPH_FINGERPRINT


def protocol_fingerprint(eager: dict[str, Any]) -> str:
    raw_measurement = eager.get("measurement")
    measurement = raw_measurement if isinstance(raw_measurement, dict) else {}
    raw_points = eager.get("points")
    points = raw_points if isinstance(raw_points, list) else []
    payload = {
        "seeds": measurement.get("seeds"),
        "cycles": measurement.get("cycles"),
        "iterations_per_measurement": measurement.get("iterations_per_measurement"),
        "order": measurement.get("order"),
        "points": [
            {
                "resolution": point.get("resolution"),
                "dtype": point.get("dtype"),
                "batch": point.get("batch"),
                "consumer_n": point.get("consumer_n"),
            }
            for point in points
            if isinstance(point, dict)
        ],
    }
    return sha256_text(canonical_dumps(payload))


def environment_summary(environment: dict[str, Any]) -> dict[str, Any]:
    torch_info = environment.get("torch", {})
    devices = torch_info.get("devices", []) if isinstance(torch_info, dict) else []
    device = devices[0] if devices else {}
    if not isinstance(device, dict):
        device = {}
    torch_ok = isinstance(torch_info, dict)
    nvidia_smi = environment.get("nvidia_smi")
    driver = nvidia_smi.get("stdout", "") if isinstance(nvidia_smi, dict) else ""
    return {
        "device": device.get("name", "unknown"),
        "compute_capability": device.get("compute_capability", "unknown"),
        "torch": torch_info.get("version", "unknown") if torch_ok else "unknown",
        "cuda_build": torch_info.get("cuda_build", "unknown") if torch_ok else "unknown",
        "cudnn": torch_info.get("cudnn") if torch_ok else None,
        "driver_record": driver,
        "python": environment.get("python", "unknown"),
    }


def identity_key(environment: dict[str, Any], eager: dict[str, Any]) -> str:
    summary = environment_summary(environment)
    payload = {
        "graph": graph_fingerprint(eager),
        "device": summary["device"],
        "compute_capability": summary["compute_capability"],
        "torch": summary["torch"],
        "cuda_build": summary["cuda_build"],
        "cudnn": summary["cudnn"],
        "protocol": protocol_fingerprint(eager),
    }
    return sha256_text(canonical_dumps(payload))


def measurement_file_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    files = manifest.get("files", {})
    if not isinstance(files, dict):
        return {}
    interesting = ("environment.json", "eager_results.json", "compile_results.json")
    return {name: str(files[name]) for name in interesting if name in files}
