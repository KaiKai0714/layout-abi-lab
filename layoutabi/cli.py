"""Command-line interface for reproduction and result validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
