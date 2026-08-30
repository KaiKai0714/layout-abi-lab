"""Private subprocess entry point used to isolate TorchInductor compilation caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--policy", choices=("direct", "repair_kv"), required=True)
    parser.add_argument(
        "--audit-dir",
        type=Path,
        help="If set, save FX/export/Inductor/profiler artifacts for a mechanism audit",
    )
    args = parser.parse_args()

    if args.audit_dir is not None:
        from .audit import compiled_audit_worker

        result = compiled_audit_worker(args.resolution, args.policy, args.audit_dir)
    else:
        from .benchmark import compiled_worker

        result = compiled_worker(args.resolution, args.policy)
    print("LAYOUTABI_WORKER_JSON=" + json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
