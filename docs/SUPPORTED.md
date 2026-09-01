# Support matrix

This package does not install PyTorch. Install a CUDA-enabled build that matches
the GPU first, then install Layout ABI Lab. An automatic `pip` PyTorch wheel can
silently pick the wrong CUDA runtime.

```bash
python -m pip install -e .
layoutabi supported
```

`layoutabi supported` and `layoutabi.supported()` do not import PyTorch.

## CPU tools (no PyTorch)

After a dependency-free install, these commands do not import PyTorch:

`supported`, `list-workloads`, `validate`, `validate-tree`, `aggregate`,
`prepare-submission`, `migrate-schema`, `evaluate-planner`, `cache-info`,
`cache-clear`, `rc-status`, and `scan-release`.

`layoutabi check` and the reproduce/optimizer commands import PyTorch when
invoked. Result JSON Schema validation and index generation are CPU-only.

## Optimizer APIs (require PyTorch)

`layoutabi.optimize`, `layoutabi.inspect`, `layoutabi inspect-model`, and
`layoutabi optimize-model` raise `MissingPyTorchError` when PyTorch is missing.
Autotune additionally needs CUDA example inputs; without CUDA the optimizer
keeps the original graph instead of raising `UnsupportedCUDAError`.

Capture uses public `torch.fx.symbolic_trace` first, then `torch.export`.
Framework-specific failures stay in `layoutabi/optimizer/adapters/` so they
cannot break `import layoutabi`.

The research question is producer layout versus vendor GEMM family. The table
below is only what `optimize()` will rewrite today.

## Declared optimizer contract

| Item | This release |
|---|---|
| Pattern | `linear_attention_ktv_v1` (see [Pattern contract](PATTERN_CONTRACT.md)) |
| Dtype | FP16 |
| Shapes | Fixed example inputs; optional buckets 32…512 |
| Mode | Inference only |
| Policies | `off`, `direct`, `repair_k`, `repair_kv`, `autotune`, `n_mod_8`, `cost_model` |
| Mechanism audits | Compiled graph; FP16 K-pointer × V-pointer alignment |
| Unseen sizes | `direct`, `noop`, or `autotune` — never an unverified repair |

BF16, INT8, FP8, dynamic shapes, and graphs that do not match the frozen pattern
are documented boundaries, not silent rewrites.

## Reference software stacks

Complete public L40S protocol runs used:

- PyTorch `2.11.0+cu128` / CUDA 12.8
- PyTorch `2.11.0+cu126` / CUDA 12.6
- PyTorch `2.10.0+cu128` / CUDA 12.8

Those are reference stacks, not extra devices. Community bundles on other GPUs
are the generalization test.

Public diagnostics objects use schema `layoutabi_optimizer_diagnostics_v1`.
Explain a decision with `layoutabi.explain(result)` without re-running capture.
