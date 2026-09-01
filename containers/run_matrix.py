#!/usr/bin/env python3
"""Run pinned stacks sequentially and remove every newly pulled image after use."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any


def load_matrix(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != "layoutabi_container_matrix_v1":
        raise ValueError(f"Unsupported matrix schema in {path}")
    return payload


def safe_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-")
    if not sanitized:
        raise ValueError(f"Stack name is not usable as a path or image tag: {value!r}")
    return sanitized


def run(command: list[str], *, cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def image_exists(image: str, *, cwd: Path) -> bool:
    """Return whether an image tag existed before this matrix invocation."""

    proc = subprocess.run(
        ["docker", "image", "inspect", image],
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


def remove_image(image: str, *, cwd: Path) -> bool:
    """Best-effort removal used only for an image pulled by this invocation."""

    print("+", "docker image rm", image, flush=True)
    proc = subprocess.run(
        ["docker", "image", "rm", image],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        print(f"Warning: could not remove newly pulled image {image}: {detail}", file=sys.stderr)
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--gpu-device",
        help="Override the physical GPU index from the matrix file (for example, 2)",
    )
    parser.add_argument(
        "--keep-images",
        action="store_true",
        help="Keep images pulled by this run instead of reclaiming their disk space",
    )
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--gpu-tag",
        required=True,
        help="Short GPU token for output dirs, for example l40s or orin",
    )
    args = parser.parse_args()

    matrix_path = args.matrix.resolve()
    root = matrix_path.parents[1]
    matrix = load_matrix(matrix_path)
    benchmark = matrix.get("benchmark", {})
    gpu_device = str(args.gpu_device or matrix.get("gpu_device", "0"))
    cleanup_new_images = bool(matrix.get("cleanup_new_images", True)) and not args.keep_images
    enabled = [stack for stack in matrix.get("stacks", []) if stack.get("enabled", True)]
    if not enabled:
        print("No enabled stacks. Edit containers/matrix.json first.", file=sys.stderr)
        return 2

    failures = []
    user = f"{os.getuid()}:{os.getgid()}"
    for stack in enabled:
        name = safe_name(str(stack["name"]))
        image = str(stack["image"])
        gpu_tag = safe_name(str(args.gpu_tag))
        day = date.today().isoformat()
        output = (args.output_root / f"local_{gpu_tag}_{name}_{day}").resolve()
        existed_before = image_exists(image, cwd=root)
        stop_after_failure = False
        try:
            if not existed_before:
                run(["docker", "pull", image], cwd=root)
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(
                    f"Output already exists and is not empty: {output}. "
                    "Use a different --output-root or move the previous run."
                )
            output.mkdir(parents=True, exist_ok=True)
            run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--gpus",
                    f"device={gpu_device}",
                    "--ipc=host",
                    "--user",
                    user,
                    "--workdir",
                    "/workspace/layout-abi-lab",
                    "--env",
                    "PYTHONPATH=/workspace/layout-abi-lab",
                    "--entrypoint",
                    "python",
                    "-v",
                    f"{root}:/workspace/layout-abi-lab:ro",
                    "-v",
                    f"{output}:/results",
                    image,
                    "-m",
                    "layoutabi.cli",
                    "reproduce",
                    "--output",
                    "/results",
                    "--resolutions",
                    str(benchmark.get("resolutions", "256,128")),
                    "--seeds",
                    str(benchmark.get("seeds", "1701,1702,1703")),
                    "--cycles",
                    str(benchmark.get("cycles", 12)),
                    "--iterations",
                    str(benchmark.get("iterations", 12)),
                ],
                cwd=root,
            )
            run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--user",
                    user,
                    "--workdir",
                    "/workspace/layout-abi-lab",
                    "--env",
                    "PYTHONPATH=/workspace/layout-abi-lab",
                    "--entrypoint",
                    "python",
                    "-v",
                    f"{root}:/workspace/layout-abi-lab:ro",
                    "-v",
                    f"{output}:/results:ro",
                    image,
                    "-m",
                    "layoutabi.cli",
                    "validate",
                    "/results",
                    "--strict",
                ],
                cwd=root,
            )
        except Exception as exc:
            failures.append({"stack": name, "error": repr(exc)})
            print(f"Stack {name} failed: {exc}", file=sys.stderr)
            if not args.keep_going:
                stop_after_failure = True
        finally:
            if cleanup_new_images and not existed_before and image_exists(image, cwd=root):
                if not remove_image(image, cwd=root):
                    failures.append(
                        {"stack": name, "error": f"cleanup failed for newly pulled image {image}"}
                    )
        if stop_after_failure:
            break

    if failures:
        print(json.dumps({"failures": failures}, indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
