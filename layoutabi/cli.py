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

    optimize_parser = subparsers.add_parser(
        "optimize-model",
        help="Optimize the bundled LinearAttention module or no-op if unsupported",
    )
    optimize_parser.add_argument("--resolution", type=int, default=128)
    optimize_parser.add_argument("--device", default="cpu")
    optimize_parser.add_argument(
        "--policy",
        default="autotune",
        choices=("off", "direct", "repair_k", "repair_kv", "autotune"),
    )
    optimize_parser.add_argument("--compile", action="store_true")
    optimize_parser.add_argument("--cache-dir", type=Path)

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

    if args.command in {"inspect-model", "optimize-model"}:
        try:
            model, example_inputs = _bundled_module(args.resolution, args.device)
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
        )
        print(json.dumps(result.diagnostics, indent=2, default=str))
        print(f"decision: {result.decision}")
        return 0 if result.decision != "noop" else 2

    raise AssertionError(f"Unhandled command: {args.command}")


def _bundled_module(resolution: int, device: str) -> tuple[Any, tuple[Any, ...]]:
    try:
        import torch

        from .workload import PublicDiffusionLinearAttention
    except ImportError as exc:
        raise RuntimeError(
            "This command requires PyTorch. Install a CUDA-enabled build appropriate "
            "for this GPU; this package does not install PyTorch."
        ) from exc
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model = PublicDiffusionLinearAttention(policy="direct").eval().half()
    sample = torch.randn(1, 64, resolution, resolution, dtype=torch.float16)
    if device == "cuda":
        model = model.cuda()
        sample = sample.cuda()
    return model, (sample,)


if __name__ == "__main__":
    raise SystemExit(main())
