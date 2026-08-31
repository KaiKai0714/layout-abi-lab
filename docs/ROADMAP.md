# Roadmap

## 0.1: Reproducibility artifact

- One-command eager and compiled reproduction.
- Environment fingerprinting and correctness gates.
- Reference L40S result and community result schema.
- Container-based software-stack matrix.

## 0.2.0: Community result platform

- Deterministic Markdown and JSON result indexes.
- Automatic checksum and correctness validation before aggregation.
- Reference/community separation and local-result privacy boundary.
- Pull-request workflow with continuous-integration freshness checks.

## 0.2.1: Result platform hardening

- JSON Schema contracts, `schema_version`, and forward migration for result documents.
- `layoutabi prepare-submission` with checksum rewrite and privacy findings.
- Duplicate/replicate detection that does not count same-device same-stack re-runs as
  new device evidence.
- Index separation of reference, community, and replicate bundles, including
  unavailable compiled cells.

## 0.2.x: Community characterization expansion

- Results from additional GPU architectures and software stacks.
- Automated aggregation of wins, losses, unsupported cells, and kernel families.
- Stable profiler and generated-code audit protocol.
- Additional independent public graph provenance.

## 0.3.0: External graph optimizer MVP

- Capture fixed-shape inference graphs with public `torch.fx` / `torch.export` APIs.
- Match one frozen LinearAttention K-softmax → KTV GEMM pattern without class-name keys.
- Generate direct, repair-K, and repair-KV candidates with a correctness canary.
- Select by full-module CUDA-event autotune and cache the decision.
- Fall back to the original graph for every unsupported, unguardable, or failing case.

## 0.4.0: Compiler mechanism audit

- Isolated compiled `direct` / `repair_kv` process and cache directories.
- Saved FX/export graphs, Inductor pre/post fusion IR, and ordered profiler kernel names.
- Structured evidence for copy survival, fusion/elimination, GEMM families, and strides.
- Compiled win/loss interpretations require graph or profiler evidence, not latency only.

## 0.5.0: Workload generalization

- Second independent public graph: Shen et al. Efficient Attention reconstruction.
- Public negative graph: scaled dot-product attention, which must no-op.
- Synthetic resolution/batch/dtype cells for boundary coverage only.
- L40S remains the only complete public device reference. Extra software stacks are
  not extra devices.

## 0.6.0: Planner and cost model

- Named baselines including `N % 8`, always-repair, always-direct, and autotune.
- Conservative cost model with autotune fallback; false-repair/regret gates frozen.
- `layoutabi evaluate-planner` scores those rules on published community/reference
  oracles so new environments can test whether `N % 8` holds.

## 0.7.0: Runtime cache and shape contract

- Locked, versioned decision cache with corruption recovery.
- Explicit shape buckets; unseen shapes do not take an unverified repair.
- Latency-critical processes can disable synchronous autotune.
- Cold-start timings and break-even invocation counts in diagnostics.

## 0.8.0: Stable package and API

- Public `optimize` / `inspect` / `clear_cache` / `explain` / `supported`.
- Structured exceptions and a versioned diagnostics schema.
- FX/`torch.export` adapters isolated from package import.
- Install without forcing a PyTorch wheel; documented support matrix.

## 1.0: Compiler-quality evaluation

- Held-out shape and device evaluation against a two-action oracle.
- False-repair, coverage, and regret metrics.
- Compile-time and cold-start overhead.
- Dynamic-shape policy or an explicit fixed-shape contract.
- Multiple framework versions and public workloads.
- Evaluation of whether an upstream TorchInductor integration is maintainable.
