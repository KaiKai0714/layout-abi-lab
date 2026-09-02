# Changelog

All notable changes to this project are documented in this file.

## 1.0.0

- Published the matching Jetson Orin 100-cell K-pointer × V-pointer audit.
  Correctness and profiler family identification pass in all cells.
- Reproduced the least-aligned N/K/V family-tier rule in 200/200 controlled
  cells across L40S and Orin, while preserving device-specific latency ratios.
- Isolated each requested N in a fresh subprocess to avoid exhaustion of
  embedded Kineto profiler collection state; process startup is not timed.
- Closed all frozen v1 measurement gates without expanding the optimizer's
  single-pattern scope or its bounded FP16 claims.
- Classified author-run Orin artifacts under `results/reference_orin/` and made
  the mechanism/profitability evidence boundary explicit.
- Reframed the README and package description around hidden GEMM dispatch
  contracts; repair is documented as a causal intervention and secondary
  cost-benefit experiment rather than the main contribution.

## 0.9.1

- Fixed pointer-audit portability on embedded PyTorch: profiler collection now
  uses one marker-tagged session per N instead of 100 consecutive sessions, and
  correctness accepts the recorded absolute-or-relative tolerance rather than
  requiring both scale-dependent metrics simultaneously.

- Added a controlled `audit-pointer` / `validate-pointer-audit` protocol covering
  both GEMM operand pointers, all three N-residue tiers, exact pointer residues,
  profiler kernel families, correctness, checksums, and a portable summary.
- Published the L40S 100-cell pointer grid. The least-aligned tier among N, K
  pointer, and V pointer predicts the profiler family in all measured cells.

- Clarified the project object: producer layout → vendor GEMM family, with
  feature-map / reduction length as a testable prior. Public LinearAttention
  graphs are witnesses; the automatic optimizer still matches only the frozen
  KTV pattern.
- Result index rows now separate the three-level FP16 residue prior, isolated
  KTV profiler tokens, the conservative binary safety action, and module oracle.
- Published the Orin eager-128 author-run bundle (repair slower; compiled
  unavailable), then closed the residue-mechanism gate with one L40S bundle
  spanning all three FP16 classes.
- Added a dedicated L40S boundary sweep covering fastest (`align8/ldg8`),
  intermediate (`align2`), and slowest (`align1`) residue classes. New runs
  profile the isolated KTV consumer at every resolution.
- Published the six-shape L40S sweep: direct KTV selects `align8`, `align2`,
  and `align1` exactly as the three-level prior predicts; repair-KV selects
  `align8` in all six cells.

## 0.9.0

- Release-candidate freeze for the v1.0 API, pattern, candidate implementation,
  cache protocol, and document schemas. New matcher patterns are out of scope.
- `layoutabi rc-status` reports remaining v1.0 gates honestly: Orin and a second
  GPU architecture are still open. `layoutabi scan-release` checks git-tracked
  files for secrets, private paths, and required license files.
- RC feedback issue template for crash, incorrect rewrite, regression, and
  unexpected no-op reports.

## 0.8.0

- Frozen public API: `layoutabi.optimize`, `inspect`, `clear_cache`, `explain`,
  and `supported`, plus structured exceptions (`LayoutABIError`,
  `MissingPyTorchError`, `InvalidArgumentError`).
- Optimizer diagnostics use schema `layoutabi_optimizer_diagnostics_v1`.
- FX and `torch.export` capture live behind versioned adapters so a private
  framework failure cannot break `import layoutabi`.
- `layoutabi supported` prints the declared PyTorch/CUDA matrix. The package
  still does not install a PyTorch wheel; CPU result tools work without one.
- Added `examples/` scripts and [docs/SUPPORTED.md](docs/SUPPORTED.md).

## 0.7.0

- Hardened the optimizer decision cache: process lock, atomic replace, cache
  protocol v2, corrupt-file recovery, and a human-readable `DIAGNOSTICS.md`.
- Cache keys now include bucketed or exact shapes, strides, pointer class, GPU
  UUID/CC, and software-stack fields so a stack change cannot reuse old decisions.
- Unseen sizes outside published buckets, and latency-critical runs with
  `--no-sync-autotune`, never apply an unverified repair.
- Diagnostics report capture/autotune/compile timings and autotune break-even
  invocations. CLI: `cache-info`, `cache-clear`.
- Fixed argparse help for `evaluate-planner` (`N % 8` broke `layoutabi --help`).

## 0.6.0

- Added planner baselines `always_direct`, `always_repair_kv`, `n_mod_8`, and
  `cost_model`, scored against a two-action oracle on published result rows.
- `N % 8` is a community-testable FP16 hypothesis (`layoutabi evaluate-planner`),
  not a universal rule. The conservative cost model falls back to autotune when
  N is misaligned so false-repair is not baked into a static rewrite.
- `layoutabi.optimize(..., policy="n_mod_8"|"cost_model")` uses those rules live.

## 0.5.0

- Added a second independent public graph: source-equivalent Efficient Attention from
  `cmsflash/efficient-attention` (Shen et al., WACV 2021, MIT).
- Added a public negative graph: scaled dot-product attention, which the optimizer
  must leave unchanged.
- Added a synthetic resolution/batch/dtype grid for boundary coverage, not as a
  substitute for public graphs.
- Added `layoutabi list-workloads` and `--workload` on inspect/optimize.
- Workload cases are drop-in JSON specs plus a `build()` module; catalog tests run
  every registered case instead of hard-coding the current two graphs.

## 0.4.0

- Added `layoutabi audit-compile` to save FX/export graphs, TorchInductor pre/post
  fusion IR, generated code, ordered CUDA profiler names, and full-module latency for
  isolated compiled `direct` and `repair_kv` cells.
- Derived copy-present, copy-fused/eliminated, first/second GEMM family, and generated
  stride evidence from those artifacts. Compiled kernel-family claims cannot be inferred
  from timing alone.
- Added `layoutabi validate-audit` and a JSON Schema for audit documents.
- Nsight remains optional and is not part of the default matrix.

## 0.3.0

- Added `layoutabi.optimize()` and `layoutabi.inspect()` as an external FX/export
  optimizer for one frozen LinearAttention K-softmax → KTV GEMM pattern.
- Added capture, bounded graph matching, dtype/layout guards, direct/repair-K/repair-KV
  rewrites, a correctness canary, CUDA-event autotune, and a versioned decision cache.
- Matcher uses public graph ops only; it does not key off module class names or
  repository paths. Unsupported, unguardable, or incorrect rewrites keep the original
  graph.
- Added `layoutabi inspect-model` and `layoutabi optimize-model` for the bundled
  public module. Autotune requires CUDA; CPU rewrite policies remain testable.
- Scope is inference, fixed shapes, FP16, and a single pattern. This is not a general
  TorchInductor pass.

## 0.2.1

- Added JSON Schema contracts and `schema_version` for environment, eager, compile,
  manifest, and index documents.
- Added forward migration so v0.1/v0.2 bundles remain readable; newer schema versions
  are rejected instead of being silently misread.
- Added `layoutabi prepare-submission` to copy a local bundle into `results/community/`,
  recompute checksums, and report hostname, username, private-path, and extra-metadata
  findings without uploading.
- Added `layoutabi migrate-schema` for explicit bundle migration.
- Added duplicate/replicate detection from graph, device, software stack, and
  measurement protocol. Re-runs are kept but indexed as replicates, not new devices.
- Separated the generated index into reference, community, and replicate sections with
  filterable role, device, dtype, stack, resolution, and outcome fields.
- Recorded community bundles without compiled controls as compiled-unavailable rather
  than as a direct/repair loss.

## 0.2.0

- Added deterministic `layoutabi aggregate` Markdown and JSON result indexes.
- Added automatic discovery of reference and community result bundles.
- Added aggregate win/loss, device, stack, and availability counts.
- Added CI checks for bundle integrity and generated-index freshness.
- Added a tracked `results/community/` pull-request workflow.
- Preserved and indexed the validated three-stack L40S matrix.
- Updated the sole author metadata to Sheng-Kai Ku.

## 0.1.0

- Initial public reproducibility artifact.
- Added order-balanced eager and isolated `torch.compile` controls.
- Added environment fingerprints, correctness gates, checksums, and strict validation.
- Added the low-disk sequential container matrix runner.
