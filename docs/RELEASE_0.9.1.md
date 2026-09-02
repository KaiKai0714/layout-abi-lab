# v0.9.1 — L40S mechanism-complete checkpoint

Version 0.9.1 corrects the project framing and publishes the complete L40S
mechanism evidence available at this release. It remains a v1.0 release
candidate: the public optimizer API and matcher are frozen, while cross-device
pointer transfer remains open.

## What changed since v0.9.0

- Reframed the scientific object as producer layout → vendor GEMM family, with
  Linear Attention as a public witness rather than the theory boundary.
- Separated three distinct concepts throughout the index and documentation:
  kernel-family mechanism, conservative safety action, and full-module
  profitability oracle.
- Published a six-shape L40S sweep spanning all three FP16 residue tiers.
- Added isolated KTV profiler evidence at every boundary shape.
- Published a six-shape compiled mechanism audit with FX/export graphs,
  TorchInductor pre/post-fusion IR, generated code, ordered kernel names,
  correctness, and latency.
- Added and published a full 4 × 5 × 5 L40S operand-pointer audit.
- Published the existing Jetson Orin eager-128 negative/boundary result without
  presenting it as a complete cross-device regime map.

## Frozen L40S evidence

| Evidence | Coverage | Main result |
|---|---|---|
| Software-stack matrix | PyTorch 2.10/2.11; CUDA 12.6/12.8 | Compiled repair-KV improves 256 module latency by 13.95–15.19% |
| Three-level sweep | resolutions 126–128 and 254–256 | Eager isolated KTV reproduces `align8`, `align2`, `align1` |
| Compiled audit | same six resolutions; direct/repair-KV | Compilation may insert/retain copies and change the family selected by direct execution |
| Pointer audit | four N classes × five K residues × five V residues | 100/100 cells follow the least-aligned tier among N, K pointer, and V pointer |

At aligned N=65536, isolated median latency was:

| Family | Median | Relative to `align8` |
|---|---:|---:|
| `align8` | 0.148685 ms | 1.00× |
| `align2` | 0.183219 ms | 1.23× |
| `align1` | 0.352358 ms | 2.37× |

These values characterize the recorded L40S, FP16, PyTorch 2.11/CUDA 12.8
consumer GEMM. They are not universal constants.

## Evidence locations

- `results/reference_l40s/software_stack_matrix/`
- `results/reference_l40s/three_level_sweep/torch2.11_cuda12.8/`
- `results/reference_l40s/compile_audit/torch2.11_cuda12.8/`
- `results/reference_l40s/pointer_alignment/torch2.11_cuda12.8/`
- `results/reference_orin/orin_torch2.7_cuda12.8_2026-08-31/`

## Explicit boundary for this release

The complete K-pointer × V-pointer grid has not yet been run on Orin. The
published Orin bundle is an eager-128 direct-win boundary only. Consequently:

- v0.9.1 may claim complete L40S mechanism evidence;
- it may not claim that the 100-cell least-aligned-tier rule transfers unchanged
  to Orin or every GPU;
- the matching Orin audit is the only open measurement gate reported by
  `layoutabi rc-status`.

The static `n_mod_8` and `cost_model` planners are unchanged. They do not inspect
internal K/V pointer addresses, so the pointer result is mechanism evidence for
future guards or compiler integration, not a new deployed pointer-aware policy.

## Release checks

Before tagging:

```bash
python -m unittest discover -s tests -p 'test_*.py'
layoutabi validate-tree results/reference_l40s --strict
layoutabi validate-pointer-audit \
  results/reference_l40s/pointer_alignment/torch2.11_cuda12.8
layoutabi validate-audit \
  results/reference_l40s/compile_audit/torch2.11_cuda12.8
layoutabi aggregate --check
layoutabi rc-status --check
layoutabi scan-release
```
