"""Compiled-graph mechanism audit: FX/export, Inductor IR, and kernel families."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .environment import write_environment
from .schema import AUDIT_SCHEMA, sha256_file
AUDIT_POLICIES = ("direct", "repair_kv")
DEBUG_FILENAMES = {
    "pre_fusion": "ir_pre_fusion.txt",
    "post_fusion": "ir_post_fusion.txt",
    "output_code": "output_code.py",
    "fx_transformed": "fx_graph_transformed.py",
    "fx_runnable": "fx_graph_runnable.py",
}

COPY_TEXT_RE = re.compile(
    r"aten\.copy|aten\.clone|aten\.contiguous|\.copy_|memcpy|\bclone\(|\bClone\(",
    re.IGNORECASE,
)
COPY_KERNEL_RE = re.compile(
    r"copy|clone|contiguous|memcpy|CatArrayBatchedCopy",
    re.IGNORECASE,
)
GEMM_KERNEL_RE = re.compile(
    r"gemm|cutlass|wmma|cublas|triton_tem_fused_mm|extern_kernels\.mm|\bmm\b",
    re.IGNORECASE,
)
ALIGN_RE = re.compile(r"align\d+", re.IGNORECASE)
STRIDE_RE = re.compile(r"stride=\(([^)]+)\)")
EMPTY_STRIDED_RE = re.compile(r"empty_strided\(\(([^)]+)\),\s*\(([^)]+)\)")
CONTIGUOUS_RE = re.compile(r"\bcontiguous\b", re.IGNORECASE)


def kernel_family(name: str) -> str | None:
    matches = ALIGN_RE.findall(name.lower())
    if matches:
        return matches[0]
    lowered = name.lower()
    if "cutlass" in lowered:
        return "cutlass"
    if "triton" in lowered:
        return "triton"
    if "cublas" in lowered:
        return "cublas"
    if "wmma" in lowered:
        return "wmma"
    if GEMM_KERNEL_RE.search(name):
        return "gemm"
    return None


def is_copy_kernel(name: str) -> bool:
    return COPY_KERNEL_RE.search(name) is not None and not GEMM_KERNEL_RE.search(name)


def is_gemm_kernel(name: str) -> bool:
    return GEMM_KERNEL_RE.search(name) is not None


def parse_ir_flags(text: str) -> dict[str, Any]:
    strides = {f"stride=({match.group(1)})" for match in STRIDE_RE.finditer(text)}
    strides.update(
        f"stride=({match.group(2)})" for match in EMPTY_STRIDED_RE.finditer(text)
    )
    return {
        "copy_like": bool(COPY_TEXT_RE.search(text) or CONTIGUOUS_RE.search(text)),
        "contiguous": bool(CONTIGUOUS_RE.search(text)),
        "strides": sorted(strides)[:32],
    }


def interpret_evidence(
    *,
    ordered_cuda_names: list[str],
    pre_fusion: str | None,
    post_fusion: str | None,
    fx_graph: str | None,
    output_code: str | None,
) -> dict[str, Any]:
    """Turn saved graphs, IR, and profiler names into causal (not timing-only) flags."""

    copy_kernels = [name for name in ordered_cuda_names if is_copy_kernel(name)]
    gemm_kernels = [name for name in ordered_cuda_names if is_gemm_kernel(name)]
    gemm_families = [kernel_family(name) for name in gemm_kernels]
    pre = parse_ir_flags(pre_fusion or "")
    post = parse_ir_flags(post_fusion or "")
    fx = parse_ir_flags(fx_graph or "")
    generated = parse_ir_flags(output_code or "")
    copy_pre = pre["copy_like"]
    copy_post = post["copy_like"]
    if pre_fusion is None or post_fusion is None:
        copy_fused_or_eliminated = None
    else:
        copy_fused_or_eliminated = bool(copy_pre and not copy_post)
    return {
        "copy_in_profiler": bool(copy_kernels),
        "copy_kernels": copy_kernels[:20],
        "copy_in_fx": fx["copy_like"] or fx["contiguous"],
        "copy_in_pre_fusion": copy_pre,
        "copy_in_post_fusion": copy_post,
        "copy_fused_or_eliminated": copy_fused_or_eliminated,
        "materialization_retained_in_compiled": bool(copy_post or copy_kernels),
        "gemm_kernel_count": len(gemm_kernels),
        "gemm_families": [family for family in gemm_families if family is not None],
        "first_gemm_family": gemm_families[0] if gemm_families else None,
        "second_gemm_family": gemm_families[1] if len(gemm_families) > 1 else None,
        "generated_strides": generated["strides"] or post["strides"] or fx["strides"],
        "evidence_source": [
            source
            for source, present in (
                ("profiler", bool(ordered_cuda_names)),
                ("fx_graph", fx_graph is not None),
                ("pre_fusion", pre_fusion is not None),
                ("post_fusion", post_fusion is not None),
                ("output_code", output_code is not None),
            )
            if present
        ],
    }


def enable_compiler_debug(debug_root: Path) -> Path:
    dump = debug_root / "torch_compile_debug"
    dump.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    try:
        import torch._dynamo.config as dynamo_config

        dynamo_config.debug_dir_root = str(dump)
    except Exception:
        pass
    try:
        import torch._inductor.config as inductor_config

        inductor_config.trace.enabled = True
    except Exception:
        pass
    return dump


def harvest_debug_files(debug_root: Path, destination: Path) -> dict[str, str | None]:
    """Copy the latest Inductor debug files into ``destination``."""

    destination.mkdir(parents=True, exist_ok=True)
    stored: dict[str, str | None] = {}
    for key, filename in DEBUG_FILENAMES.items():
        matches = sorted(debug_root.rglob(filename))
        if not matches:
            stored[key] = None
            continue
        target = destination / filename
        shutil.copy2(matches[-1], target)
        stored[key] = target.name
    return stored


def _read_optional(directory: Path, name: str | None) -> str | None:
    if not name:
        return None
    path = directory / name
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def _profile_ordered_cuda_names(fn: Any) -> dict[str, Any]:
    import torch

    try:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn()
        torch.cuda.synchronize()
        ordered: list[str] = []
        for event in prof.events():
            name = str(event.name)
            cuda_time = getattr(event, "device_time", 0) or getattr(event, "cuda_time", 0)
            if cuda_time or is_gemm_kernel(name) or is_copy_kernel(name):
                if not ordered or ordered[-1] != name:
                    ordered.append(name)
        selected = [
            name
            for name in ordered
            if is_gemm_kernel(name) or is_copy_kernel(name) or "align" in name.lower()
        ]
        return {
            "available": True,
            "ordered_cuda_names": ordered[:240],
            "selected_cuda_names": selected[:160],
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": repr(exc),
            "ordered_cuda_names": [],
            "selected_cuda_names": [],
        }


def compiled_audit_worker(resolution: int, policy: str, artifact_dir: Path) -> dict[str, Any]:
    """Compile one isolated policy and save graph/IR/profiler evidence."""

    from .benchmark import _correctness, _measure, _summary
    from .runtime import prepare_runtime
    from .workload import PublicDiffusionLinearAttention

    prepare_runtime(f"layoutabi_audit_r{resolution}_{policy}")
    import torch

    if policy not in AUDIT_POLICIES:
        raise ValueError(f"Unsupported compiled audit policy: {policy}")
    if not torch.cuda.is_available():
        raise RuntimeError("The compiled audit requires CUDA")
    if not hasattr(torch, "compile"):
        raise RuntimeError("This PyTorch build does not provide torch.compile")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    debug_root = enable_compiler_debug(artifact_dir / "debug_tmp")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(170900 + resolution)
    x = torch.randn(1, 64, resolution, resolution, device="cuda", dtype=torch.float16)
    module = PublicDiffusionLinearAttention(policy=policy).cuda().half().eval()

    fx_name = None
    export_name = None
    try:
        graph = torch.fx.symbolic_trace(module)
        fx_path = artifact_dir / "fx_graph.txt"
        fx_path.write_text(str(graph.graph) + "\n", encoding="utf-8")
        fx_name = fx_path.name
    except Exception as exc:
        (artifact_dir / "fx_graph_error.txt").write_text(repr(exc), encoding="utf-8")
    if hasattr(torch, "export"):
        try:
            exported = torch.export.export(module, (x,))
            export_path = artifact_dir / "export_graph.txt"
            export_path.write_text(str(exported.graph) + "\n", encoding="utf-8")
            export_name = export_path.name
        except Exception as exc:
            (artifact_dir / "export_graph_error.txt").write_text(repr(exc), encoding="utf-8")

    compiled = torch.compile(module, mode="reduce-overhead")
    with torch.no_grad():
        reference = module(x)
        import time

        start = time.perf_counter()
        value = compiled(x)
        torch.cuda.synchronize()
        first_call_ms = (time.perf_counter() - start) * 1000.0
        correctness = _correctness(value, reference)
        for _ in range(8):
            compiled(x)
        torch.cuda.synchronize()
        samples = [_measure(lambda: compiled(x), 20) for _ in range(10)]
        profiler = _profile_ordered_cuda_names(lambda: compiled(x))

    harvested = harvest_debug_files(debug_root, artifact_dir)
    shutil.rmtree(artifact_dir / "debug_tmp", ignore_errors=True)
    evidence = interpret_evidence(
        ordered_cuda_names=profiler.get("ordered_cuda_names", []),
        pre_fusion=_read_optional(artifact_dir, harvested.get("pre_fusion")),
        post_fusion=_read_optional(artifact_dir, harvested.get("post_fusion")),
        fx_graph=_read_optional(artifact_dir, fx_name),
        output_code=_read_optional(artifact_dir, harvested.get("output_code")),
    )
    return {
        "available": True,
        "first_call_ms": first_call_ms,
        "steady_state": _summary(samples),
        "correctness": correctness,
        "artifacts": {
            "fx_graph": fx_name,
            "export_graph": export_name,
            **harvested,
        },
        "profiler": profiler,
        "evidence": evidence,
    }


def _run_audit_subprocess(
    resolution: int, policy: str, cache_root: Path, artifact_dir: Path
) -> dict[str, Any]:
    env = os.environ.copy()
    cache = cache_root / f"r{resolution}_{policy}"
    env.update(
        {
            "LAYOUTABI_RUNTIME_HOME": str(cache),
            "HOME": str(cache / "home"),
            "XDG_CACHE_HOME": str(cache / "xdg"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
            "TRITON_CACHE_DIR": str(cache / "triton"),
            "TMPDIR": str(cache / "tmp"),
            "TORCH_COMPILE_DEBUG": "1",
        }
    )
    for key in ("HOME", "XDG_CACHE_HOME", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TMPDIR"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "layoutabi.worker",
        "--resolution",
        str(resolution),
        "--policy",
        policy,
        "--audit-dir",
        str(artifact_dir),
    ]
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            timeout=1800,
            check=False,
        )
    except Exception as exc:
        return {"available": False, "reason": repr(exc)}
    marker = "LAYOUTABI_WORKER_JSON="
    lines = [line for line in proc.stdout.splitlines() if line.startswith(marker)]
    if proc.returncode != 0 or not lines:
        detail = (proc.stderr or proc.stdout or f"return code {proc.returncode}")[-4000:]
        return {"available": False, "reason": detail}
    return json.loads(lines[-1][len(marker) :])


def _write_summary(output: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Compiled mechanism audit",
        "",
        "Kernel families and copies are taken from profiler names and Inductor IR,",
        "not inferred from latency. A compiled win/loss needs this evidence before",
        "reusing an eager explanation.",
        "",
        "| Resolution | Policy | Available | Steady ms | Copy (profiler) | Copy post-fusion | "
        "Copy fused/eliminated | First GEMM | Second GEMM |",
        "|---:|---|---|---:|---|---|---|---|---|",
    ]
    for point in payload.get("points", []):
        for policy, cell in point.get("policies", {}).items():
            if not cell.get("available"):
                lines.append(
                    f"| {point.get('resolution')} | {policy} | no | — | — | — | — | — | — |"
                )
                continue
            evidence = cell.get("evidence", {})
            steady = cell.get("steady_state", {}).get("median_ms")
            fused = evidence.get("copy_fused_or_eliminated")
            fused_text = "—" if fused is None else str(fused)
            lines.append(
                f"| {point.get('resolution')} | {policy} | yes | "
                f"{'—' if steady is None else f'{steady:.6f}'} | "
                f"{evidence.get('copy_in_profiler')} | "
                f"{evidence.get('copy_in_post_fusion')} | {fused_text} | "
                f"{evidence.get('first_gemm_family') or '—'} | "
                f"{evidence.get('second_gemm_family') or '—'} |"
            )
    lines.extend(
        [
            "",
            "Nsight Systems/Compute is optional and only for cells where graph and",
            "profiler evidence still cannot separate KTV, the second GEMM, and fusion.",
            "See docs/COMPILE_AUDIT.md.",
        ]
    )
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _collect_file_hashes(output: Path, payload: dict[str, Any]) -> dict[str, str]:
    files = {"SUMMARY.md": sha256_file(output / "SUMMARY.md")}
    env_path = output / "environment.json"
    if env_path.is_file():
        files["environment.json"] = sha256_file(env_path)
    for point in payload.get("points", []):
        resolution = point.get("resolution")
        for policy, cell in point.get("policies", {}).items():
            artifacts = cell.get("artifacts") or {}
            cell_dir = output / "artifacts" / f"r{resolution}_{policy}"
            for name in artifacts.values():
                if not name:
                    continue
                path = cell_dir / str(name)
                if path.is_file():
                    relative = path.relative_to(output).as_posix()
                    files[relative] = sha256_file(path)
    return files


def validate_audit(directory: Path) -> list[str]:
    from .schema import load_json_object, normalize_document

    problems: list[str] = []
    path = directory / "compile_audit.json"
    if not path.is_file():
        return ["Missing required file: compile_audit.json"]
    try:
        payload = load_json_object(path)
    except Exception as exc:
        return [f"Could not parse compile_audit.json: {exc}"]
    migrated, schema_problems = normalize_document(payload, AUDIT_SCHEMA)
    problems.extend(schema_problems)
    document = migrated or payload
    files = document.get("files", {})
    if isinstance(files, dict):
        for name, expected in files.items():
            if name == "compile_audit.json":
                continue
            file_path = directory / str(name)
            if not file_path.is_file():
                problems.append(f"Audit references a missing file: {name}")
            elif sha256_file(file_path) != expected:
                problems.append(f"Checksum mismatch: {name}")
    for point in document.get("points", []):
        if not isinstance(point, dict):
            continue
        for policy, cell in (point.get("policies") or {}).items():
            if not isinstance(cell, dict):
                continue
            if cell.get("available") and "evidence" not in cell:
                problems.append(
                    f"Available compiled audit cell missing evidence at "
                    f"resolution={point.get('resolution')}, policy={policy}"
                )
    return problems


def run_compile_audit(*, output: Path, resolutions: tuple[int, ...]) -> dict[str, Any]:
    """Run isolated compiled audits for each resolution and policy."""

    import torch

    output.mkdir(parents=True, exist_ok=True)
    write_environment(output / "environment.json")
    points: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="layoutabi_audit_") as cache_directory:
        cache_root = Path(cache_directory)
        for resolution in resolutions:
            record: dict[str, Any] = {"resolution": resolution, "policies": {}}
            for policy in AUDIT_POLICIES:
                artifact_dir = output / "artifacts" / f"r{resolution}_{policy}"
                artifact_dir.mkdir(parents=True, exist_ok=True)
                record["policies"][policy] = _run_audit_subprocess(
                    resolution, policy, cache_root, artifact_dir
                )
            points.append(record)
    payload: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "schema_version": 1,
        "software": {"torch": torch.__version__, "cuda_build": torch.version.cuda},
        "protocol": {
            "policies": list(AUDIT_POLICIES),
            "resolutions": list(resolutions),
            "compile_mode": "reduce-overhead",
            "isolated_process_and_cache": True,
            "nsight_required": False,
        },
        "points": points,
    }
    _write_summary(output, payload)
    payload["files"] = _collect_file_hashes(output, payload)
    (output / "compile_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload
