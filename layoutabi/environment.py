"""Collect a portable hardware and software fingerprint for result provenance."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__


def _command(args: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=15, check=False)
        return {
            "available": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {"available": False, "reason": repr(exc)}


def collect_environment() -> dict[str, Any]:
    """Return non-secret runtime metadata required to interpret a benchmark result."""

    payload: dict[str, Any] = {
        "schema": "layoutabi_environment_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "layoutabi_version": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "container": {
            "likely_container": Path("/.dockerenv").exists(),
            "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "nvidia_smi": _command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,compute_cap,memory.total,power.limit",
                "--format=csv,noheader,nounits",
            ]
        ),
    }
    try:
        import torch

        torch_info: dict[str, Any] = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "git_version": getattr(torch.version, "git_version", None),
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "compile_available": hasattr(torch, "compile"),
        }
        if torch.cuda.is_available():
            devices = []
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                devices.append(
                    {
                        "index": index,
                        "name": props.name,
                        "compute_capability": f"{props.major}.{props.minor}",
                        "sm_count": int(props.multi_processor_count),
                        "total_memory_bytes": int(props.total_memory),
                    }
                )
            torch_info["devices"] = devices
        payload["torch"] = torch_info
    except Exception as exc:
        payload["torch"] = {"available": False, "reason": repr(exc)}
    return payload


def write_environment(path: Path) -> dict[str, Any]:
    payload = collect_environment()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
