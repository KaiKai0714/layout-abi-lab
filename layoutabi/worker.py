"""Private subprocess entry point used to isolate TorchInductor compilation caches."""

from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--policy", choices=("direct", "repair_kv"), required=True)
    args = parser.parse_args()

    from .benchmark import compiled_worker

    result = compiled_worker(args.resolution, args.policy)
    print("LAYOUTABI_WORKER_JSON=" + json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

