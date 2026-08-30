# Roadmap

## 0.1: Reproducibility artifact

- One-command eager and compiled reproduction.
- Environment fingerprinting and correctness gates.
- Reference L40S result and community result schema.
- Container-based software-stack matrix.

## 0.2: Community characterization

- Results from additional GPU architectures and software stacks.
- Automated aggregation of wins, losses, unsupported cells, and kernel families.
- Stable profiler and generated-code audit protocol.
- Additional independent public graph provenance.

## 0.3: External graph optimizer MVP

- Capture fixed-shape inference graphs with supported public APIs.
- Match bounded normalize/view-to-GEMM patterns.
- Generate direct and materialized candidates.
- Run a correctness canary and profile-guided candidate selection.
- Cache decisions by graph, shape, dtype, device, and software stack.
- Fall back to the original graph for every unsupported case.

## 1.0: Compiler-quality evaluation

- Held-out shape and device evaluation against a two-action oracle.
- False-repair, coverage, and regret metrics.
- Compile-time and cold-start overhead.
- Dynamic-shape policy or an explicit fixed-shape contract.
- Multiple framework versions and public workloads.
- Evaluation of whether an upstream TorchInductor integration is maintainable.

