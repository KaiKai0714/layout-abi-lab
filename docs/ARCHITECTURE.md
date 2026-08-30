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

The planned optimizer will sit outside TorchInductor initially. It will capture a graph,
match a bounded pattern, generate direct and repaired candidates, run a correctness
canary, benchmark both candidates, cache the decision, and then compile the selected
graph. Unsupported graphs will remain unchanged.

