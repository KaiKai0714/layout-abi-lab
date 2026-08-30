"""Simple versioned decision cache with atomic replacement."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import __version__
from .pattern import CANDIDATE_IMPL_HASH, PATTERN_ID


CACHE_SCHEMA = "layoutabi_optimizer_cache_v1"
CACHE_FILENAME = "decisions.json"


def make_cache_key(payload: dict[str, Any]) -> str:
    import hashlib

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def cache_identity(
    *,
    graph_fingerprint: str,
    example_inputs: tuple[Any, ...],
    torch_info: dict[str, Any],
) -> dict[str, Any]:
    import torch

    tensors = [value for value in example_inputs if isinstance(value, torch.Tensor)]
    device = tensors[0].device if tensors else None
    return {
        "graph_fingerprint": graph_fingerprint,
        "pattern_id": PATTERN_ID,
        "candidate_impl": CANDIDATE_IMPL_HASH,
        "optimizer_version": __version__,
        "shapes": [list(tensor.shape) for tensor in tensors],
        "strides": [list(tensor.stride()) for tensor in tensors],
        "dtypes": [str(tensor.dtype) for tensor in tensors],
        "devices": [str(tensor.device) for tensor in tensors],
        "torch": torch_info.get("version"),
        "cuda_build": torch_info.get("cuda_build"),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device is not None and device.type == "cuda"
            else None
        ),
    }


def _cache_path(cache_dir: Path) -> Path:
    return cache_dir / CACHE_FILENAME


def load_entry(cache_dir: Path | None, key: str) -> dict[str, Any] | None:
    if cache_dir is None:
        return None
    path = _cache_path(cache_dir)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema") != CACHE_SCHEMA:
        return None
    entry = payload.get("entries", {}).get(key)
    return entry if isinstance(entry, dict) else None


def store_entry(cache_dir: Path | None, key: str, entry: dict[str, Any]) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir)
    payload = {"schema": CACHE_SCHEMA, "schema_version": 1, "entries": {}}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("entries"), dict):
                payload = loaded
                payload["schema"] = CACHE_SCHEMA
                payload["schema_version"] = 1
        except (OSError, json.JSONDecodeError):
            payload = {"schema": CACHE_SCHEMA, "schema_version": 1, "entries": {}}
    record = dict(entry)
    record["created_utc"] = datetime.now(timezone.utc).isoformat()
    payload.setdefault("entries", {})[key] = record
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
