"""Order-balanced eager and isolated compiled reproduction protocols."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from .environment import write_environment
from .runtime import prepare_runtime
from .identity import DEFAULT_GRAPH_FINGERPRINT
from .schema import COMPILE_SCHEMA, EAGER_SCHEMA, MANIFEST_SCHEMA, current_version

POLICIES = ("direct", "repair_k", "repair_kv")


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "min_ms": min(values),
        "max_ms": max(values),
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


def _profile_names(fn: Callable[[], Any]) -> dict[str, Any]:
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
            if any(
                token in name.lower()
                for token in ("gemm", "cutlass", "ampere", "wmma", "copy", "clone")
            )
        ]
        return {"available": True, "selected_cuda_names": selected[:160]}
    except Exception as exc:
        return {"available": False, "reason": repr(exc), "selected_cuda_names": []}


def run_eager(
    *,
    resolutions: tuple[int, ...],
    seeds: tuple[int, ...],
    cycles: int,
    iterations: int,
) -> dict[str, Any]:
    """Run the multi-seed protocol while rotating all six policy permutations."""

    prepare_runtime("layoutabi_eager")
    import torch

    from .workload import (
        PublicDiffusionLinearAttention,
        context_from_kv,
        make_chain_inputs,
        public_chain,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("The eager reproduction requires a CUDA-enabled PyTorch build")
    if cycles < 1 or iterations < 1 or not seeds:
        raise ValueError("Seeds, cycles, and iterations must all be non-empty and positive")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    permutations = list(itertools.permutations(POLICIES))
    points: list[dict[str, Any]] = []

    for resolution in resolutions:
        aggregate = {
            "chain": {policy: [] for policy in POLICIES},
            "module": {policy: [] for policy in POLICIES},
        }
        seed_records = []
        module_profiler = None
        ktv_profiler = None
        for seed_index, seed in enumerate(seeds):
            torch.manual_seed(seed + resolution)
            q, k, v = make_chain_inputs(resolution, batch=1, dtype=torch.float16)
            x = torch.randn(1, 64, resolution, resolution, device="cuda", dtype=torch.float16)
            direct = PublicDiffusionLinearAttention(policy="direct").cuda().half().eval()
            repair_k = copy.deepcopy(direct)
            repair_k.policy = "repair_k"
            repair_kv = copy.deepcopy(direct)
            repair_kv.policy = "repair_kv"
            modules = {"direct": direct, "repair_k": repair_k, "repair_kv": repair_kv}
            chain_fns = {
                policy: (lambda selected=policy: public_chain(q, k, v, selected))
                for policy in POLICIES
            }
            ktv_fns = {
                policy: (lambda selected=policy: context_from_kv(k, v, selected))
                for policy in ("direct", "repair_kv")
            }
            module_fns = {
                policy: (lambda selected=module: selected(x))
                for policy, module in modules.items()
            }
            samples = {
                "chain": {policy: [] for policy in POLICIES},
                "module": {policy: [] for policy in POLICIES},
            }
            with torch.no_grad():
                chain_reference = chain_fns["direct"]()
                module_reference = module_fns["direct"]()
                correctness = {
                    "chain": {
                        policy: _correctness(fn(), chain_reference)
                        for policy, fn in chain_fns.items()
                    },
                    "module": {
                        policy: _correctness(fn(), module_reference)
                        for policy, fn in module_fns.items()
                    },
                }
                for fn in (*chain_fns.values(), *module_fns.values()):
                    for _ in range(5):
                        fn()
                torch.cuda.synchronize()

                for cycle in range(cycles):
                    chain_order = permutations[(cycle + seed_index) % len(permutations)]
                    module_order = permutations[(cycle * 5 + seed_index + 1) % len(permutations)]
                    for policy in chain_order:
                        samples["chain"][policy].append(_measure(chain_fns[policy], iterations))
                    for policy in module_order:
                        samples["module"][policy].append(_measure(module_fns[policy], iterations))

                if seed_index == 0:
                    # Profile the isolated first K^T V consumer at every length. Full-module
                    # names contain unrelated GEMMs and cannot establish the three-level
                    # align8/align2/align1 family ladder on their own.
                    ktv_profiler = {
                        policy: _profile_names(ktv_fns[policy])
                        for policy in ("direct", "repair_kv")
                    }
                if resolution == max(resolutions) and seed_index == 0:
                    module_profiler = {
                        policy: _profile_names(module_fns[policy])
                        for policy in ("direct", "repair_kv")
                    }

            seed_record: dict[str, Any] = {
                "seed": seed,
                "correctness": correctness,
                "chain": {},
                "module": {},
            }
            for group in ("chain", "module"):
                for policy in POLICIES:
                    seed_record[group][policy] = _summary(samples[group][policy])
                    aggregate[group][policy].extend(samples[group][policy])
                seed_record[group]["direct_over_repair_kv"] = (
                    seed_record[group]["direct"]["median_ms"]
                    / seed_record[group]["repair_kv"]["median_ms"]
                )
            seed_records.append(seed_record)
            del q, k, v, x, direct, repair_k, repair_kv, modules
            torch.cuda.empty_cache()

        aggregate_summary: dict[str, Any] = {"chain": {}, "module": {}}
        for group in ("chain", "module"):
            for policy in POLICIES:
                aggregate_summary[group][policy] = _summary(aggregate[group][policy])
            aggregate_summary[group]["direct_over_repair_kv"] = (
                aggregate_summary[group]["direct"]["median_ms"]
                / aggregate_summary[group]["repair_kv"]["median_ms"]
            )
            ratios = [record[group]["direct_over_repair_kv"] for record in seed_records]
            aggregate_summary[group]["repair_kv_wins_all_seeds"] = all(
                ratio > 1.0 for ratio in ratios
            )
            aggregate_summary[group]["seed_ratios"] = ratios

        points.append(
            {
                "dtype": "fp16",
                "batch": 1,
                "resolution": resolution,
                "consumer_n": resolution * resolution + 4,
                "n_mod_8": (resolution * resolution + 4) % 8,
                "seeds": seed_records,
                "aggregate": aggregate_summary,
                "ktv_profiler": ktv_profiler,
                "module_profiler": module_profiler,
            }
        )

    props = torch.cuda.get_device_properties(0)
    return {
        "schema": EAGER_SCHEMA,
        "schema_version": current_version(EAGER_SCHEMA),
        "graph_fingerprint": DEFAULT_GRAPH_FINGERPRINT,
        "device": {
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "sm_count": int(props.multi_processor_count),
        },
        "software": {"torch": torch.__version__, "cuda_build": torch.version.cuda},
        "measurement": {
            "seeds": list(seeds),
            "cycles": cycles,
            "iterations_per_measurement": iterations,
            "order": "rotating all six policy permutations with distinct chain/module rotations",
        },
        "points": points,
    }


def compiled_worker(resolution: int, policy: str) -> dict[str, Any]:
    """Compile one policy in an isolated process and return one JSON-compatible cell."""

    prepare_runtime(f"layoutabi_compile_r{resolution}_{policy}")
    import torch

    from .workload import PublicDiffusionLinearAttention

    if policy not in {"direct", "repair_kv"}:
        raise ValueError(f"Unsupported compiled policy: {policy}")
    if not torch.cuda.is_available():
        raise RuntimeError("The compiled reproduction requires CUDA")
    if not hasattr(torch, "compile"):
        raise RuntimeError("This PyTorch build does not provide torch.compile")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(170900 + resolution)
    x = torch.randn(1, 64, resolution, resolution, device="cuda", dtype=torch.float16)
    module = PublicDiffusionLinearAttention(policy=policy).cuda().half().eval()
    compiled = torch.compile(module, mode="reduce-overhead")
    with torch.no_grad():
        reference = module(x)
        start = time.perf_counter()
        value = compiled(x)
        torch.cuda.synchronize()
        first_call_ms = (time.perf_counter() - start) * 1000.0
        correctness = _correctness(value, reference)
        for _ in range(8):
            compiled(x)
        torch.cuda.synchronize()
        samples = [_measure(lambda: compiled(x), 20) for _ in range(10)]
    return {
        "available": True,
        "first_call_ms": first_call_ms,
        "steady_state": _summary(samples),
        "correctness": correctness,
    }


def _run_compiled_subprocess(resolution: int, policy: str, cache_root: Path) -> dict[str, Any]:
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
    ]
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            timeout=1200,
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


def run_compiled(resolutions: tuple[int, ...], cache_root: Path) -> dict[str, Any]:
    """Run each compiled policy in a clean subprocess and cache directory."""

    import torch

    points = []
    for resolution in resolutions:
        record = {
            "resolution": resolution,
            "consumer_n": resolution * resolution + 4,
            "policies": {},
        }
        for policy in ("direct", "repair_kv"):
            record["policies"][policy] = _run_compiled_subprocess(
                resolution, policy, cache_root
            )
        direct = record["policies"]["direct"].get("steady_state", {}).get("median_ms")
        repair = record["policies"]["repair_kv"].get("steady_state", {}).get("median_ms")
        record["direct_over_repair_kv"] = direct / repair if direct and repair else None
        points.append(record)
    return {
        "schema": COMPILE_SCHEMA,
        "schema_version": current_version(COMPILE_SCHEMA),
        "software": {"torch": torch.__version__, "cuda_build": torch.version.cuda},
        "points": points,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_summary(
    output: Path, eager: dict[str, Any], compiled: dict[str, Any] | None
) -> None:
    lines = [
        "# Layout ABI reproduction summary",
        "",
        f"Device: **{eager['device']['name']}**",
        "",
        "## Order-balanced eager results",
        "",
        "| Resolution | N%8 | Scope | Direct ms | Repair-K ms | Repair-KV ms | Direct / Repair-KV | Repair wins every seed |",
        "|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for point in eager["points"]:
        n_mod = point.get("n_mod_8", "—")
        for scope in ("chain", "module"):
            result = point["aggregate"][scope]
            lines.append(
                f"| {point['resolution']} | {n_mod} | {scope} | "
                f"{result['direct']['median_ms']:.6f} | "
                f"{result['repair_k']['median_ms']:.6f} | "
                f"{result['repair_kv']['median_ms']:.6f} | "
                f"{result['direct_over_repair_kv']:.3f} | "
                f"{result['repair_kv_wins_all_seeds']} |"
            )
    if compiled is not None:
        lines.extend(
            [
                "",
                "## Isolated torch.compile results",
                "",
                "| Resolution | Direct ms | Repair-KV ms | Direct / Repair-KV |",
                "|---:|---:|---:|---:|",
            ]
        )
        for point in compiled["points"]:
            direct = point["policies"]["direct"].get("steady_state", {}).get("median_ms")
            repair = point["policies"]["repair_kv"].get("steady_state", {}).get("median_ms")
            ratio = point.get("direct_over_repair_kv")
            lines.append(
                f"| {point['resolution']} | "
                f"{'unavailable' if direct is None else f'{direct:.6f}'} | "
                f"{'unavailable' if repair is None else f'{repair:.6f}'} | "
                f"{'unavailable' if ratio is None else f'{ratio:.3f}'} |"
            )
    lines.extend(
        [
            "",
            "A ratio above 1 means repair-KV was faster at that scope. Kernel-family",
            "names live in `eager_results.json` (`ktv_profiler`); they identify the",
            "isolated consumer-GEMM mechanism. Ratio is whether materialization paid off.",
            "This result is scoped to the recorded graph, device, shape, dtype, and stack.",
        ]
    )
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def reproduce(
    *,
    output: Path,
    resolutions: tuple[int, ...],
    seeds: tuple[int, ...],
    cycles: int,
    iterations: int,
    skip_compile: bool,
) -> None:
    """Run the complete public protocol and write an integrity manifest."""

    output.mkdir(parents=True, exist_ok=True)
    write_environment(output / "environment.json")
    eager = run_eager(
        resolutions=resolutions,
        seeds=seeds,
        cycles=cycles,
        iterations=iterations,
    )
    _write_json(output / "eager_results.json", eager)
    compiled = None
    if not skip_compile:
        # Compilation caches are intentionally temporary. They can be several gigabytes
        # and are neither portable nor part of a scientific result bundle.
        with tempfile.TemporaryDirectory(prefix="layoutabi_compile_") as cache_directory:
            compiled = run_compiled(resolutions, Path(cache_directory))
        _write_json(output / "compile_results.json", compiled)
    _write_summary(output, eager, compiled)

    measured_files = ["environment.json", "eager_results.json", "SUMMARY.md"]
    if compiled is not None:
        measured_files.append("compile_results.json")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": current_version(MANIFEST_SCHEMA),
        "files": {name: _sha256(output / name) for name in measured_files},
    }
    _write_json(output / "manifest.json", manifest)
