"""Command-line interface for reproduction and result validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("Expected a comma-separated list of positive integers")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="layoutabi",
        description="Reproduce and validate cross-operator layout profitability experiments.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check", help="Print the local hardware and software fingerprint")

    reproduce_parser = subparsers.add_parser(
        "reproduce", help="Run order-balanced eager and isolated compiled controls"
    )
    reproduce_parser.add_argument("--output", type=Path, required=True)
    reproduce_parser.add_argument("--resolutions", type=_csv_ints, default=(256, 128))
    reproduce_parser.add_argument("--seeds", type=_csv_ints, default=(1701, 1702, 1703))
    reproduce_parser.add_argument("--cycles", type=int, default=12)
    reproduce_parser.add_argument("--iterations", type=int, default=12)
    reproduce_parser.add_argument(
        "--skip-compile", action="store_true", help="Skip the torch.compile control"
    )

    validate_parser = subparsers.add_parser(
        "validate", help="Validate schemas, correctness gates, and checksums"
    )
    validate_parser.add_argument("directory", type=Path)
    validate_parser.add_argument(
        "--strict", action="store_true", help="Require every compiled cell to be available"
    )

    tree_parser = subparsers.add_parser(
        "validate-tree", help="Validate every result bundle discovered below a directory"
    )
    tree_parser.add_argument("directory", type=Path)
    tree_parser.add_argument(
        "--strict", action="store_true", help="Require every compiled cell to be available"
    )

    aggregate_parser = subparsers.add_parser(
        "aggregate", help="Generate the deterministic reference and community result index"
    )
    aggregate_parser.add_argument("--results-root", type=Path, default=Path("results"))
    aggregate_parser.add_argument("--output-json", type=Path, default=Path("results/index.json"))
    aggregate_parser.add_argument("--output-markdown", type=Path, default=Path("RESULTS_INDEX.md"))
    aggregate_parser.add_argument(
        "--check", action="store_true", help="Fail instead of writing if generated files are stale"
    )

    submit_parser = subparsers.add_parser(
        "prepare-submission",
        help="Copy a local bundle into results/community after checksum and privacy checks",
    )
    submit_parser.add_argument("directory", type=Path)
    submit_parser.add_argument(
        "--name",
        required=True,
        help="Destination directory name under results/community/",
    )
    submit_parser.add_argument("--results-root", type=Path, default=Path("results"))
    submit_parser.add_argument(
        "--strict",
        action="store_true",
        help="Require every compiled cell to be available",
    )
    submit_parser.add_argument(
        "--strict-privacy",
        action="store_true",
        help="Fail if hostname, username, private path, or extra metadata is found",
    )

    migrate_parser = subparsers.add_parser(
        "migrate-schema",
        help="Forward-migrate a result bundle to the current schema_version",
    )
    migrate_parser.add_argument("directory", type=Path)
    migrate_parser.add_argument(
        "--write",
        action="store_true",
        help="Write migrated JSON and recomputed checksums; default is a dry run",
    )

    inspect_parser = subparsers.add_parser(
        "inspect-model",
        help="Capture and match the bundled LinearAttention module without rewriting",
    )
    inspect_parser.add_argument("--resolution", type=int, default=128)
    inspect_parser.add_argument("--device", default="cpu")
    inspect_parser.add_argument(
        "--workload",
        default="diffusion_linear_attention",
        help="Named public workload id; see layoutabi list-workloads",
    )
    inspect_parser.add_argument("--batch", type=int, default=1)
    inspect_parser.add_argument("--dtype", default="fp16")

    optimize_parser = subparsers.add_parser(
        "optimize-model",
        help="Optimize the bundled LinearAttention module or no-op if unsupported",
    )
    optimize_parser.add_argument("--resolution", type=int, default=128)
    optimize_parser.add_argument("--device", default="cpu")
    optimize_parser.add_argument(
        "--workload",
        default="diffusion_linear_attention",
        help="Named public workload id; see layoutabi list-workloads",
    )
    optimize_parser.add_argument("--batch", type=int, default=1)
    optimize_parser.add_argument("--dtype", default="fp16")
    optimize_parser.add_argument(
        "--policy",
        default="autotune",
        choices=(
            "off",
            "direct",
            "repair_k",
            "repair_kv",
            "autotune",
            "n_mod_8",
            "cost_model",
        ),
    )
    optimize_parser.add_argument("--compile", action="store_true")
    optimize_parser.add_argument("--cache-dir", type=Path)
    optimize_parser.add_argument(
        "--shape-mode",
        choices=("exact", "bucket"),
        default="exact",
    )
    optimize_parser.add_argument(
        "--unseen-shape",
        choices=("direct", "noop", "autotune"),
        default="direct",
        help="Action for sizes outside the published shape buckets",
    )
    optimize_parser.add_argument(
        "--no-sync-autotune",
        action="store_true",
        help="Never run synchronous autotune; use cache or the unseen-shape action",
    )

    audit_parser = subparsers.add_parser(
        "audit-compile",
        help="Save compiled FX/export graphs, Inductor IR, and profiler kernel evidence",
    )
    audit_parser.add_argument("--output", type=Path, required=True)
    audit_parser.add_argument("--resolutions", type=_csv_ints, default=(256, 128))

    audit_validate = subparsers.add_parser(
        "validate-audit",
        help="Validate a compiled mechanism-audit directory",
    )
    audit_validate.add_argument("directory", type=Path)

    subparsers.add_parser(
        "list-workloads",
        help="Print the public and synthetic workload catalog",
    )

    planner_parser = subparsers.add_parser(
        "evaluate-planner",
        help="Score N-mod-8 and other planner baselines against published result oracles",
    )
    planner_parser.add_argument("--results-root", type=Path, default=Path("results"))
    planner_parser.add_argument(
        "--held-out-resolutions",
        type=_csv_ints,
        default=(128,),
        help="Resolutions treated as held-out shapes (default: 128)",
    )

    cache_info_parser = subparsers.add_parser(
        "cache-info", help="Print optimizer decision-cache status"
    )
    cache_info_parser.add_argument("--cache-dir", type=Path, required=True)
    cache_clear_parser = subparsers.add_parser(
        "cache-clear", help="Erase optimizer decision-cache entries"
    )
    cache_clear_parser.add_argument("--cache-dir", type=Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "check":
        from .environment import collect_environment
        from .runtime import prepare_runtime

        prepare_runtime("layoutabi_check")
        payload = collect_environment()
        print(json.dumps(payload, indent=2))
        cuda_available = payload.get("torch", {}).get("cuda_available", False)
        return 0 if cuda_available else 2

    if args.command == "reproduce":
        if args.cycles < 1 or args.iterations < 1:
            raise SystemExit("--cycles and --iterations must be positive")
        from .benchmark import reproduce

        reproduce(
            output=args.output.resolve(),
            resolutions=args.resolutions,
            seeds=args.seeds,
            cycles=args.cycles,
            iterations=args.iterations,
            skip_compile=args.skip_compile,
        )
        print(f"Result bundle: {args.output.resolve()}")
        return 0

    if args.command == "validate":
        from .validation import validate_result

        problems = validate_result(args.directory.resolve(), strict=args.strict)
        if problems:
            print("Validation failed:", file=sys.stderr)
            for problem in problems:
                print(f"- {problem}", file=sys.stderr)
            return 1
        print(f"Valid result bundle: {args.directory.resolve()}")
        return 0

    if args.command == "validate-tree":
        from .validation import discover_result_bundles, validate_result

        root = args.directory.resolve()
        bundles = discover_result_bundles(root)
        if not bundles:
            print(f"No result bundles found below {root}", file=sys.stderr)
            return 1
        failed = False
        for bundle in bundles:
            problems = validate_result(bundle, strict=args.strict)
            if problems:
                failed = True
                print(f"Invalid result bundle: {bundle}", file=sys.stderr)
                for problem in problems:
                    print(f"- {problem}", file=sys.stderr)
            else:
                print(f"Valid result bundle: {bundle}")
        return 1 if failed else 0

    if args.command == "aggregate":
        from .aggregation import write_index

        try:
            write_index(
                results_root=args.results_root.resolve(),
                output_json=args.output_json.resolve(),
                output_markdown=args.output_markdown.resolve(),
                check=args.check,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        action = "Current" if args.check else "Generated"
        print(f"{action} result index: {args.output_markdown.resolve()}")
        return 0

    if args.command == "prepare-submission":
        from .submission import format_submission_report, prepare_submission

        try:
            result = prepare_submission(
                args.directory.resolve(),
                args.name,
                results_root=args.results_root.resolve(),
                strict=args.strict,
                strict_privacy=args.strict_privacy,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(format_submission_report(result))
        return 0

    if args.command == "migrate-schema":
        from .schema import migrate_bundle_documents

        directory = args.directory.resolve()
        try:
            migrated = migrate_bundle_documents(directory, write=args.write)
        except (OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        action = "Wrote" if args.write else "Would migrate"
        print(f"{action} {directory}: {', '.join(migrated)}")
        return 0

    if args.command == "cache-info":
        from .optimizer.cache import cache_info

        print(json.dumps(cache_info(args.cache_dir.resolve()), indent=2))
        return 0

    if args.command == "cache-clear":
        from .optimizer.cache import clear_cache

        clear_cache(args.cache_dir.resolve())
        print(f"Cleared optimizer cache: {args.cache_dir.resolve()}")
        return 0

    if args.command == "evaluate-planner":
        from .aggregation import build_index
        from .planner.evaluate import evaluate_index, render_markdown

        index = build_index(args.results_root.resolve())
        report = evaluate_index(
            index, held_out_resolutions=args.held_out_resolutions
        )
        print(render_markdown(report))
        print(json.dumps(report, indent=2, default=str))
        return 0

    if args.command == "list-workloads":
        from .workloads import list_workloads, synthetic_cells

        print(
            json.dumps(
                {"workloads": list_workloads(), "synthetic_cells": synthetic_cells()},
                indent=2,
            )
        )
        return 0

    if args.command in {"inspect-model", "optimize-model"}:
        try:
            model, example_inputs = _bundled_module(
                args.resolution,
                args.device,
                workload=args.workload,
                batch=args.batch,
                dtype=args.dtype,
            )
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if args.command == "inspect-model":
            from .optimizer.api import inspect as inspect_model

            payload = inspect_model(model, example_inputs)
            print(json.dumps(payload, indent=2, default=str))
            return 0 if payload.get("matches") else 2
        from .optimizer.api import optimize as optimize_model

        result = optimize_model(
            model,
            example_inputs,
            policy=args.policy,
            compile=args.compile,
            cache_dir=args.cache_dir,
            shape_mode=args.shape_mode,
            unseen_shape=args.unseen_shape,
            allow_sync_autotune=not args.no_sync_autotune,
        )
        print(json.dumps(result.diagnostics, indent=2, default=str))
        print(f"decision: {result.decision}")
        return 0 if result.decision != "noop" else 2

    if args.command == "audit-compile":
        from .audit import run_compile_audit

        try:
            payload = run_compile_audit(
                output=args.output.resolve(),
                resolutions=args.resolutions,
            )
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"Compiled audit: {args.output.resolve()}")
        n_points = len(payload.get("points", []))
        print(json.dumps({"software": payload.get("software"), "points": n_points}))
        return 0

    if args.command == "validate-audit":
        from .audit import validate_audit

        problems = validate_audit(args.directory.resolve())
        if problems:
            print("Audit validation failed:", file=sys.stderr)
            for problem in problems:
                print(f"- {problem}", file=sys.stderr)
            return 1
        print(f"Valid compile audit: {args.directory.resolve()}")
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


def _bundled_module(
    resolution: int,
    device: str,
    *,
    workload: str = "diffusion_linear_attention",
    batch: int = 1,
    dtype: str = "fp16",
) -> tuple[Any, tuple[Any, ...]]:
    try:
        import torch

        from .workloads import make_workload
    except ImportError as exc:
        raise RuntimeError(
            "This command requires PyTorch. Install a CUDA-enabled build appropriate "
            "for this GPU; this package does not install PyTorch."
        ) from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return make_workload(
        workload,
        resolution=resolution,
        batch=batch,
        dtype=dtype,
        device=device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
