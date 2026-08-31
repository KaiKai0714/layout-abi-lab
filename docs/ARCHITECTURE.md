# Artifact architecture

## Reproduction path

```text
public source-equivalent module
            |
            +-- direct: preserve the post-Softmax physical layout
            +-- repair-K: materialize a BHND-backed logical BHDN K view
            +-- repair-KV: materialize both K and V in that backing layout
            |
order-balanced eager benchmark
            |
isolated torch.compile workers
            |
correctness + profiler + environment records
```

Compiled policies run in separate processes and cache directories. This avoids sharing
TorchInductor or Triton artifacts between candidates and keeps first-call measurements
separate from steady-state CUDA-event latency.

## Why there is no automatic pass in version 0.1

The measured decision is not a safe static `N % 8` rewrite. Profitability changes with
the full graph, device, data type, batch, framework lowering, and vendor-library version.
Version 0.1 therefore freezes the measurement contract before introducing an optimizer.

Result documents carry a `schema` name and integer `schema_version`. Missing versions on
v0.1/v0.2 bundles migrate forward in memory. Unknown schema names and newer versions are
rejected. JSON Schema files live in `layoutabi/schemas/`. Community re-runs that match an
existing graph, device, software stack, and measurement protocol are indexed as
replicates rather than as additional devices.

Version 0.3 ships an external optimizer outside TorchInductor. It captures a public FX
or `torch.export` graph, matches the frozen LinearAttention KTV pattern documented in
`docs/PATTERN_CONTRACT.md`, generates direct and repaired candidates, runs a
correctness canary, autotunes on full-module CUDA-event latency, caches the decision,
and optionally compiles the selected graph. Unsupported graphs remain unchanged.

Version 0.4 adds a compiled mechanism audit outside the default reproduce bundle.
`layoutabi audit-compile` records FX/export graphs, Inductor IR, and profiler kernel
names for isolated compiled cells so kernel-family claims have causal evidence. See
`docs/COMPILE_AUDIT.md`.

Version 0.5 names graphs through drop-in cases under `layoutabi/workloads/cases/`:
the original diffusion LinearAttention (positive reference), Shen et al. Efficient
Attention (second independent public source), and scaled dot-product attention
(public no-op). Additional JSON+`build()` pairs are picked up automatically.
Synthetic shapes fill boundaries only.

Version 0.6 scores `N % 8` and a conservative cost model against published oracles.
New devices should run `layoutabi evaluate-planner` rather than assuming the L40S
FP16 heuristic. See `docs/PLANNER.md`.

Version 0.7 persists those decisions in a locked cache, buckets shapes explicitly,
and refuses unverified repair on unseen sizes or when synchronous autotune is
disabled. See `docs/RUNTIME.md`.

Version 0.8 freezes the public package surface: `optimize`, `inspect`,
`clear_cache`, `explain`, and `supported`, with structured exceptions and a
diagnostics JSON Schema. Capture goes through `layoutabi/optimizer/adapters/`
so `torch.export` or FX failures cannot break `import layoutabi`. See
`docs/SUPPORTED.md`. The package still does not depend on a PyTorch wheel.

