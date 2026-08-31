"""Inspect the bundled LinearAttention graph and apply a conservative rewrite."""

from __future__ import annotations

from layoutabi import MissingPyTorchError, explain, inspect, optimize


def main() -> int:
    try:
        import torch

        from layoutabi.workloads import make_workload
    except ImportError as exc:
        raise MissingPyTorchError("examples/inspect_and_optimize.py") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, example_inputs = make_workload(
        "diffusion_linear_attention",
        resolution=128,
        device=device,
    )
    matched = inspect(model, example_inputs)
    print(explain(matched))
    policy = "autotune" if device == "cuda" else "direct"
    result = optimize(model, example_inputs, policy=policy)
    print(explain(result))
    print(f"device={device} decision={result.decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
