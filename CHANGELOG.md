# Changelog

All notable changes to this project are documented in this file.

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

