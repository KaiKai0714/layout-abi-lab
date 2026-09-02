"""Controlled FP16 operand-pointer alignment audit for the consumer GEMM."""

from __future__ import annotations

import itertools
import json
import re
import statistics
import subprocess
import sys
import tempfile
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


def _selected_kernel_names(names: list[str]) -> list[str]:
    return [
        name
        for name in names
        if any(token in name.lower() for token in ("gemm", "cutlass", "ampere", "wmma"))
    ][:80]


def _event_kernel_names(event: Any) -> list[str]:
    """Collect CUDA kernel names associated with a profiler marker subtree."""

    names: list[str] = []
    stack = [event]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        for kernel in getattr(current, "kernels", ()) or ():
            name = getattr(kernel, "name", None)
            if name:
                names.append(str(name))
        stack.extend(getattr(current, "cpu_children", ()) or ())
    return names


def _profile_grid(
    functions: dict[tuple[int, int], Callable[[], Any]],
    pairs: list[tuple[int, int]],
) -> dict[tuple[int, int], dict[str, Any]]:
    """Profile one whole offset grid in one Kineto session.

    Repeatedly creating one session per cell exhausts profiler collection on some
    embedded PyTorch builds. Named CPU markers retain a one-to-one association
    between each logical cell and its CUDA kernels.
    """

    import torch

    try:
        for pair in pairs:
            functions[pair]()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            for k_offset, v_offset in pairs:
                label = f"layoutabi_pointer_k{k_offset}_v{v_offset}"
                with torch.profiler.record_function(label):
                    functions[(k_offset, v_offset)]()
        torch.cuda.synchronize()
        events = list(prof.events())
        markers = {
            str(event.name): event
            for event in events
            if str(event.name).startswith("layoutabi_pointer_")
        }
        ordered_global = _selected_kernel_names([str(event.name) for event in events])
        result: dict[tuple[int, int], dict[str, Any]] = {}
        for k_offset, v_offset in pairs:
            label = f"layoutabi_pointer_k{k_offset}_v{v_offset}"
            marker = markers.get(label)
            names = [] if marker is None else _event_kernel_names(marker)
            selected = _selected_kernel_names(sorted(set(names)))
            result[(k_offset, v_offset)] = {
                "available": marker is not None,
                "selected_cuda_names": selected,
                "family": pointer_family(selected),
                "collection": "single_session_marker_grid",
            }
            if marker is None:
                result[(k_offset, v_offset)]["reason"] = "profiler marker missing"
        # Some Kineto builds expose CUDA events in global launch order but do not
        # attach them to the record_function subtree. A one-kernel-per-cell grid
        # still has an unambiguous mapping in that case.
        if len(ordered_global) == len(pairs):
            for pair, name in zip(pairs, ordered_global):
                if result[pair]["family"] == "unknown":
                    result[pair]["selected_cuda_names"] = [name]
                    result[pair]["family"] = pointer_family([name])
                    result[pair]["collection"] = "single_session_launch_order"
        return result
    except Exception as exc:
        return {
            pair: {
                "available": False,
                "reason": repr(exc),
                "selected_cuda_names": [],
                "family": "unknown",
                "collection": "single_session_marker_grid",
            }
            for pair in pairs
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
        # Global absolute-or-relative gating avoids rejecting a long reduction
        # solely because its output scale is large. Both metrics remain recorded.
        "pass": bool(max_abs <= tolerance or relative_inf <= tolerance),
    }


def _validate_pointer_audit_args(
    output: Path,
    ns: tuple[int, ...],
    offsets: tuple[int, ...],
    cycles: int,
    iterations: int,
) -> None:
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


def _run_pointer_audit_isolated(
    *,
    output: Path,
    ns: tuple[int, ...],
    offsets: tuple[int, ...],
    cycles: int,
    iterations: int,
) -> dict[str, Any]:
    """One subprocess per N so Orin Kineto can collect names for the whole grid."""

    import torch

    output.mkdir(parents=True, exist_ok=True)
    write_environment(output / "environment.json")
    points: list[dict[str, Any]] = []
    device: dict[str, Any] | None = None
    software: dict[str, Any] | None = None
    offset_csv = ",".join(str(item) for item in offsets)
    for n in ns:
        print(f"pointer audit child N={n}", flush=True)
        with tempfile.TemporaryDirectory(prefix=f"layoutabi_ptr_n{n}_") as tmp:
            child_out = Path(tmp) / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "layoutabi.cli",
                    "audit-pointer",
                    "--output",
                    str(child_out),
                    "--ns",
                    str(n),
                    "--offsets",
                    offset_csv,
                    "--cycles",
                    str(cycles),
                    "--iterations",
                    str(iterations),
                ],
                capture_output=True,
                text=True,
                timeout=7200,
                check=False,
            )
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or f"return code {proc.returncode}")[-4000:]
                raise RuntimeError(f"Pointer audit child failed for N={n}: {detail}")
            child = json.loads((child_out / "pointer_audit.json").read_text(encoding="utf-8"))
        child_points = child.get("points")
        if not isinstance(child_points, list) or len(child_points) != 1:
            raise RuntimeError(f"Pointer audit child for N={n} did not return one N")
        points.extend(child_points)
        device = child.get("device")
        software = child.get("software")
    if device is None or software is None:
        raise RuntimeError("Pointer audit children returned no device fingerprint")
    payload = {
        "schema": POINTER_AUDIT_SCHEMA,
        "schema_version": current_version(POINTER_AUDIT_SCHEMA),
        "device": device,
        "software": software,
        "protocol": {
            "dtype": "fp16",
            "batch": 1,
            "heads": 4,
            "head_dim": 8,
            "ns": list(ns),
            "offsets_mod64_bytes": list(offsets),
            "operand_grid": "full Cartesian product of K and V pointer residues",
            "cycles": cycles,
            "iterations_per_measurement": iterations,
            "order": "rotating and alternating-direction offset-pair order",
            "setup_copy_measured": False,
            "correctness_gate": "max_abs <= tolerance OR relative_inf <= tolerance",
            "profiler_collection": "one subprocess and one Kineto session per N",
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


def run_pointer_audit(
    *,
    output: Path,
    ns: tuple[int, ...],
    offsets: tuple[int, ...],
    cycles: int,
    iterations: int,
) -> dict[str, Any]:
    """Run the full K-pointer by V-pointer grid at each fixed logical N."""

    _validate_pointer_audit_args(output, ns, offsets, cycles, iterations)
    if len(ns) > 1:
        return _run_pointer_audit_isolated(
            output=output,
            ns=ns,
            offsets=offsets,
            cycles=cycles,
            iterations=iterations,
        )

    prepare_runtime("layoutabi_pointer_audit")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("The pointer audit requires a CUDA-enabled PyTorch build")

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
            profiles = _profile_grid(functions, pairs)
            for pair in pairs:
                cell = cells[pair]
                cell["timing"] = _summary(cell.pop("samples_ms"))
                cell["profiler"] = profiles[pair]
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
            "correctness_gate": "max_abs <= tolerance OR relative_inf <= tolerance",
            "profiler_collection": "one Kineto session for this single-N child",
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
