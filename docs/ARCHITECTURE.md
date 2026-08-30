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

