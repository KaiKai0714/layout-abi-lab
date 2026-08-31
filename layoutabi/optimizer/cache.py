"""Versioned decision cache with process locking and corruption recovery."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import __version__
from .lock import CacheLock
from .pattern import CANDIDATE_IMPL_HASH, PATTERN_ID
from .shapes import bucket_shape, tensor_layout_fields

CACHE_SCHEMA = "layoutabi_optimizer_cache_v2"
CACHE_PROTOCOL = 2
CACHE_FILENAME = "decisions.json"
LOCK_FILENAME = "decisions.lock"
DIAGNOSTICS_FILENAME = "DIAGNOSTICS.md"


def make_cache_key(payload: dict[str, Any]) -> str:
    import hashlib

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _gpu_fields(device: Any) -> dict[str, Any]:
    import torch

    if device is None or getattr(device, "type", None) != "cuda" or not torch.cuda.is_available():
        return {"gpu": None, "compute_capability": None, "gpu_uuid": None}
    index = device.index if getattr(device, "index", None) is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    uuid = getattr(props, "uuid", None)
    return {
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "gpu_uuid": str(uuid) if uuid is not None else None,
    }


def cache_identity(
    *,
    graph_fingerprint: str,
    example_inputs: tuple[Any, ...],
    torch_info: dict[str, Any],
    shape_mode: str = "exact",
) -> dict[str, Any]:
    import torch

    tensors = [value for value in example_inputs if isinstance(value, torch.Tensor)]
    layouts = [tensor_layout_fields(tensor) for tensor in tensors]
    exact_shapes = [item["shape"] for item in layouts]
    bucketed = []
    in_range = True
    for shape in exact_shapes:
        mapped = bucket_shape(shape)
        if mapped is None:
            in_range = False
            bucketed.append(None)
        else:
            bucketed.append(mapped)
    device = tensors[0].device if tensors else None
    gpu = _gpu_fields(device)
    key_shapes = exact_shapes if shape_mode == "exact" else bucketed
    return {
        "cache_protocol": CACHE_PROTOCOL,
        "graph_fingerprint": graph_fingerprint,
        "pattern_id": PATTERN_ID,
        "candidate_impl": CANDIDATE_IMPL_HASH,
        "optimizer_version": __version__,
        "shape_mode": shape_mode,
        "shapes": key_shapes,
        "shapes_exact": exact_shapes,
        "shapes_bucketed": bucketed,
        "shape_in_range": in_range,
        "strides": [item["stride"] for item in layouts],
        "pointer_classes": [item["pointer_class"] for item in layouts],
        "dtypes": [item["dtype"] for item in layouts],
        "devices": [item["device"] for item in layouts],
        "torch": torch_info.get("version"),
        "cuda_build": torch_info.get("cuda_build"),
        "gpu": gpu["gpu"] or torch_info.get("device"),
        "compute_capability": gpu["compute_capability"]
        or torch_info.get("compute_capability"),
        "gpu_uuid": gpu["gpu_uuid"],
    }


def _cache_path(cache_dir: Path) -> Path:
    return cache_dir / CACHE_FILENAME


def _lock_path(cache_dir: Path) -> Path:
    return cache_dir / LOCK_FILENAME


def _empty_payload(*, recovered: bool = False) -> dict[str, Any]:
    payload = {
        "schema": CACHE_SCHEMA,
        "schema_version": CACHE_PROTOCOL,
        "optimizer_version": __version__,
        "entries": {},
    }
    if recovered:
        payload["recovered"] = True
        payload["recovered_utc"] = datetime.now(timezone.utc).isoformat()
    return payload


def _load_payload_unlocked(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return _empty_payload()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        corrupt = path.with_name(path.name + ".corrupt")
        try:
            os.replace(path, corrupt)
        except OSError:
            pass
        return _empty_payload(recovered=True)
    if not isinstance(payload, dict):
        return _empty_payload(recovered=True)
    if payload.get("schema") != CACHE_SCHEMA:
        return _empty_payload(recovered=True)
    if payload.get("schema_version") != CACHE_PROTOCOL:
        return _empty_payload(recovered=True)
    if not isinstance(payload.get("entries"), dict):
        payload["entries"] = {}
    return payload


def load_entry(cache_dir: Path | None, key: str) -> dict[str, Any] | None:
    if cache_dir is None:
        return None
    path = _cache_path(cache_dir)
    with CacheLock(_lock_path(cache_dir)):
        payload = _load_payload_unlocked(path)
    if payload.get("recovered"):
        with CacheLock(_lock_path(cache_dir)):
            _write_payload_unlocked(path, payload)
            _write_diagnostics_unlocked(cache_dir, payload)
    entry = payload.get("entries", {}).get(key)
    return entry if isinstance(entry, dict) else None


def store_entry(cache_dir: Path | None, key: str, entry: dict[str, Any]) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir)
    record = dict(entry)
    record["created_utc"] = datetime.now(timezone.utc).isoformat()
    with CacheLock(_lock_path(cache_dir)):
        payload = _load_payload_unlocked(path)
        payload["schema"] = CACHE_SCHEMA
        payload["schema_version"] = CACHE_PROTOCOL
        payload["optimizer_version"] = __version__
        payload.setdefault("entries", {})[key] = record
        _write_payload_unlocked(path, payload)
        _write_diagnostics_unlocked(cache_dir, payload)


def _write_payload_unlocked(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_diagnostics_unlocked(cache_dir: Path, payload: dict[str, Any]) -> None:
    entries = payload.get("entries") or {}
    lines = [
        "# Layout ABI optimizer cache",
        "",
        f"schema: `{payload.get('schema')}` version {payload.get('schema_version')}",
        f"optimizer: {payload.get('optimizer_version')}",
        f"entries: {len(entries)}",
        "",
        "| Key | Decision | Created |",
        "|---|---|---|",
    ]
    for key, entry in sorted(entries.items()):
        if not isinstance(entry, dict):
            continue
        lines.append(
            f"| `{key[:12]}…` | {entry.get('decision', '—')} | {entry.get('created_utc', '—')} |"
        )
    if payload.get("recovered"):
        lines.extend(
            [
                "",
                f"Recovered from a corrupt or incompatible cache at {payload.get('recovered_utc')}.",
                "Previous file was renamed with a `.corrupt` suffix when possible.",
            ]
        )
    (cache_dir / DIAGNOSTICS_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def cache_info(cache_dir: Path) -> dict[str, Any]:
    path = _cache_path(cache_dir)
    with CacheLock(_lock_path(cache_dir)):
        payload = _load_payload_unlocked(path)
    entries = payload.get("entries") or {}
    return {
        "cache_dir": str(cache_dir),
        "schema": payload.get("schema"),
        "schema_version": payload.get("schema_version"),
        "optimizer_version": payload.get("optimizer_version"),
        "entries": len(entries),
        "recovered": bool(payload.get("recovered")),
        "diagnostics": str(cache_dir / DIAGNOSTICS_FILENAME),
    }


def clear_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with CacheLock(_lock_path(cache_dir)):
        payload = _empty_payload()
        _write_payload_unlocked(_cache_path(cache_dir), payload)
        _write_diagnostics_unlocked(cache_dir, payload)
