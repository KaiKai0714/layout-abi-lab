"""Controlled FP16 operand-pointer alignment audit for the consumer GEMM."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import statistics
from pathlib import Path
from typing import Any, Callable

from .environment import write_environment
from .runtime import prepare_runtime
from .schema import (
    ENVIRONMENT_SCHEMA,
    POINTER_AUDIT_SCHEMA,
    current_version,
    load_json_object,
    normalize_document,
    sha256_file,
)


def pointer_family(names: list[str]) -> str:
    """Return the most specific alignment token observed in profiler names."""

    text = "\n".join(names).lower()
    matches = re.findall(r"align(\d+)(?!\d)", text)
    if matches:
        return f"align{min(int(item) for item in matches)}"
    if "ldg8" in text or "s16816" in text:
        return "ldg8"
    if "cutlass" in text:
        return "cutlass"
    if "gemm" in text or "cublas" in text:
        return "gemm"
    return "unknown"


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _measure(fn: Callable[[], Any], iterations: int) -> float:
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / iterations


def _profile(fn: Callable[[], Any]) -> dict[str, Any]:
    import torch

    try:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn()
        torch.cuda.synchronize()
        names = sorted({str(event.name) for event in prof.events()})
        selected = [
            name
            for name in names
            if any(token in name.lower() for token in ("gemm", "cutlass", "ampere", "wmma"))
        ]
        selected = selected[:80]
        return {
            "available": True,
            "selected_cuda_names": selected,
            "family": pointer_family(selected),
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": repr(exc),
            "selected_cuda_names": [],
            "family": "unknown",
        }


def _numel(shape: tuple[int, ...]) -> int:
    value = 1
    for dimension in shape:
        value *= dimension
    return value


def _allocate_at_mod64(shape: tuple[int, ...], requested_mod64: int) -> tuple[Any, dict[str, int]]:
    """Allocate an FP16 view whose data pointer has the requested byte residue."""

    import torch

    if requested_mod64 < 0 or requested_mod64 >= 64 or requested_mod64 % 2:
        raise ValueError("FP16 pointer offsets must be even byte residues in [0, 62]")
    count = _numel(shape)
    base = torch.empty(count + 64, device="cuda", dtype=torch.float16)
    base_mod64 = int(base.data_ptr() % 64)
    skip_bytes = (requested_mod64 - base_mod64) % 64
    if skip_bytes % 2:
        raise RuntimeError("Allocator returned an address incompatible with an FP16 offset")
    skip_elements = skip_bytes // 2
    view = base[skip_elements : skip_elements + count].reshape(shape)
    actual = int(view.data_ptr() % 64)
    if actual != requested_mod64:
        raise RuntimeError(f"Requested ptr%64={requested_mod64}, observed {actual}")
    return view, {
        "requested_mod64": requested_mod64,
        "actual_mod64": actual,
        "actual_mod16": int(view.data_ptr() % 16),
        "actual_mod8": int(view.data_ptr() % 8),
        "actual_mod4": int(view.data_ptr() % 4),
        "base_mod64": base_mod64,
        "storage_offset_elements": int(view.storage_offset()),
    }


def _correctness(value: Any, reference: Any, tolerance: float = 0.08) -> dict[str, Any]:
    value_float = value.float()
    reference_float = reference.float()
    max_abs = float((value_float - reference_float).abs().max())
    denominator = max(float(reference_float.abs().max()), 1e-12)
    relative_inf = max_abs / denominator
    return {
        "max_abs": max_abs,
        "relative_inf": relative_inf,
        "tolerance": tolerance,
        "pass": bool(max_abs <= tolerance and relative_inf <= tolerance),
    }


def run_pointer_audit(
    *,
    output: Path,
    ns: tuple[int, ...],
    offsets: tuple[int, ...],
    cycles: int,
    iterations: int,
) -> dict[str, Any]:
    """Run the full K-pointer by V-pointer grid at each fixed logical N."""

    prepare_runtime("layoutabi_pointer_audit")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("The pointer audit requires a CUDA-enabled PyTorch build")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    if not ns or any(n <= 0 for n in ns):
        raise ValueError("--ns must contain positive integers")
    if not offsets or len(set(offsets)) != len(offsets):
        raise ValueError("--offsets must contain distinct values")
    if any(offset < 0 or offset >= 64 or offset % 2 for offset in offsets):
        raise ValueError("FP16 offsets must be even byte residues in [0, 62]")
    if cycles < 1 or iterations < 1:
        raise ValueError("--cycles and --iterations must be positive")

    output.mkdir(parents=True, exist_ok=True)
    write_environment(output / "environment.json")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pairs = list(itertools.product(offsets, offsets))
    points: list[dict[str, Any]] = []
    h, d = 4, 8

    with torch.no_grad():
        for n_index, n in enumerate(ns):
            torch.manual_seed(181000 + n)
            source_k = torch.randn(1, h, d, n, device="cuda", dtype=torch.float16)
            source_v = torch.randn(1, h, n, d, device="cuda", dtype=torch.float16)
            reference = source_k @ source_v
            cells: dict[tuple[int, int], dict[str, Any]] = {}
            functions: dict[tuple[int, int], Callable[[], Any]] = {}

            for k_offset, v_offset in pairs:
                k, k_pointer = _allocate_at_mod64(tuple(source_k.shape), k_offset)
                v, v_pointer = _allocate_at_mod64(tuple(source_v.shape), v_offset)
                k.copy_(source_k)
                v.copy_(source_v)
                fn = lambda left=k, right=v: left @ right
                value = fn()
                functions[(k_offset, v_offset)] = fn
                cells[(k_offset, v_offset)] = {
                    "k_offset_bytes": k_offset,
                    "v_offset_bytes": v_offset,
                    "k_pointer": k_pointer,
                    "v_pointer": v_pointer,
                    "k_stride": list(k.stride()),
                    "v_stride": list(v.stride()),
                    "correctness": _correctness(value, reference),
                    "samples_ms": [],
                    # Retain the tensors through closures until this N is complete.
                    "_tensors": (k, v),
                }

            for fn in functions.values():
                for _ in range(5):
                    fn()
            torch.cuda.synchronize()

            for cycle in range(cycles):
                shift = (cycle * 7 + n_index * 3) % len(pairs)
                order = pairs[shift:] + pairs[:shift]
                if cycle % 2:
                    order = list(reversed(order))
                for pair in order:
                    cells[pair]["samples_ms"].append(_measure(functions[pair], iterations))

            rows = []
            for pair in pairs:
                cell = cells[pair]
                cell["timing"] = _summary(cell.pop("samples_ms"))
                cell["profiler"] = _profile(functions[pair])
                cell.pop("_tensors")
                rows.append(cell)

            points.append(
                {
                    "n": n,
                    "n_mod_8": n % 8,
                    "n_class": (
                        "fastest"
                        if n % 8 == 0
                        else "intermediate"
                        if n % 2 == 0
                        else "slowest"
                    ),
                    "rows": rows,
                }
            )
            del source_k, source_v, reference, cells, functions
            torch.cuda.empty_cache()

    props = torch.cuda.get_device_properties(0)
    payload = {
        "schema": POINTER_AUDIT_SCHEMA,
        "schema_version": current_version(POINTER_AUDIT_SCHEMA),
        "device": {
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "sm_count": int(props.multi_processor_count),
        },
        "software": {"torch": torch.__version__, "cuda_build": torch.version.cuda},
        "protocol": {
            "dtype": "fp16",
            "batch": 1,
            "heads": h,
            "head_dim": d,
            "ns": list(ns),
            "offsets_mod64_bytes": list(offsets),
            "operand_grid": "full Cartesian product of K and V pointer residues",
            "cycles": cycles,
            "iterations_per_measurement": iterations,
            "order": "rotating and alternating-direction offset-pair order",
            "setup_copy_measured": False,
        },
        "points": points,
    }
    _write_summary(output / "SUMMARY.md", payload)
    payload["files"] = {
        "environment.json": sha256_file(output / "environment.json"),
        "SUMMARY.md": sha256_file(output / "SUMMARY.md"),
    }
    (output / "pointer_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Operand pointer-alignment audit",
        "",
        f"Device: **{payload['device']['name']}**",
        "",
        "Each matrix cell is `family / median ms`. Rows vary K-pointer residue;",
        "columns vary V-pointer residue. Logical shapes and strides stay fixed",
        "within each N. Setup copies are excluded from timing.",
    ]
    offsets = payload["protocol"]["offsets_mod64_bytes"]
    for point in payload["points"]:
        rows = {(row["k_offset_bytes"], row["v_offset_bytes"]): row for row in point["rows"]}
        lines.extend(
            [
                "",
                f"## N={point['n']} (N%8={point['n_mod_8']}, {point['n_class']})",
                "",
                "| K ptr%64 \\ V ptr%64 | " + " | ".join(str(item) for item in offsets) + " |",
                "|---:|" + "---:|" * len(offsets),
            ]
        )
        for k_offset in offsets:
            cells = []
            for v_offset in offsets:
                row = rows[(k_offset, v_offset)]
                family = row["profiler"]["family"]
                median = row["timing"]["median_ms"]
                cells.append(f"{family} / {median:.6f}")
            lines.append(f"| {k_offset} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Kernel families come from CUDA profiler names, never from latency.",
            "This audit characterizes an isolated FP16 consumer GEMM; it does not",
            "by itself establish full-module repair profitability.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_pointer_audit(directory: Path) -> list[str]:
    """Validate schema, checksums, grid completeness, pointer residues, and correctness."""

    problems: list[str] = []
    result_path = directory / "pointer_audit.json"
    if not result_path.is_file():
        return [f"Missing {result_path}"]
    try:
        payload = load_json_object(result_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"Cannot read {result_path}: {exc}"]
    normalized, schema_problems = normalize_document(payload, POINTER_AUDIT_SCHEMA)
    problems.extend(schema_problems)
    if normalized is None:
        return problems

    for name, expected in normalized.get("files", {}).items():
        path = directory / name
        if not path.is_file():
            problems.append(f"Missing checksummed file: {name}")
        elif sha256_file(path) != expected:
            problems.append(f"Checksum mismatch: {name}")
    environment_path = directory / "environment.json"
    if environment_path.is_file():
        environment, env_problems = normalize_document(
            load_json_object(environment_path), ENVIRONMENT_SCHEMA
        )
        problems.extend(f"environment.json: {item}" for item in env_problems)
        if environment is None:
            problems.append("environment.json: unsupported schema")

    protocol = normalized.get("protocol", {})
    expected_ns = protocol.get("ns", [])
    offsets = protocol.get("offsets_mod64_bytes", [])
    points = normalized.get("points", [])
    observed_ns = [point.get("n") for point in points]
    if observed_ns != expected_ns:
        problems.append(f"N grid mismatch: expected {expected_ns}, observed {observed_ns}")
    expected_pairs = set(itertools.product(offsets, offsets))
    for point in points:
        n = point.get("n")
        rows = point.get("rows", [])
        pairs = {(row.get("k_offset_bytes"), row.get("v_offset_bytes")) for row in rows}
        if pairs != expected_pairs or len(rows) != len(expected_pairs):
            problems.append(f"N={n}: incomplete or duplicate pointer grid")
        for row in rows:
            pair = (row.get("k_offset_bytes"), row.get("v_offset_bytes"))
            if row.get("k_pointer", {}).get("actual_mod64") != pair[0]:
                problems.append(f"N={n} pair={pair}: K pointer residue mismatch")
            if row.get("v_pointer", {}).get("actual_mod64") != pair[1]:
                problems.append(f"N={n} pair={pair}: V pointer residue mismatch")
            if not row.get("correctness", {}).get("pass"):
                problems.append(f"N={n} pair={pair}: correctness failed")
            profiler = row.get("profiler", {})
            if not profiler.get("available"):
                problems.append(f"N={n} pair={pair}: profiler unavailable")
            elif profiler.get("family") == "unknown":
                problems.append(f"N={n} pair={pair}: GEMM family not identified")
    return problems
