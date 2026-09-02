# Layout ABI result index

This file is generated from checksum-validated reference and community bundles.
The primary research object is layout ABI → vendor GEMM family. The FP16
mechanism prior has three residue tiers: N divisible by 8 maps to
align8/ldg8, even non-multiples of 8 to align2, and odd N to align1.
Tokens are extracted from profiler names, not portable GEMM-family identifiers.
The workload tables below are secondary intervention-cost measurements.
The safety action remains binary: direct for N%8==0, otherwise repair.
Oracle and ratio are whether materialization paid off at full-module scope;
a ratio above 1 means repair-KV was faster. Replicates are not extra devices.

## Coverage

| Bundles | Reference | Community | Replicates | Devices | Software stacks | Primary rows | All rows |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 5 | 0 | 1 | 2 | 4 | 13 | 15 |

## Outcomes

Outcome counts use primary bundles only.

| Scope | Repair wins | Direct wins | Unavailable |
|---|---:|---:|---:|
| Eager full module | 6 | 7 | 0 |
| Compiled full module | 8 | 4 | 1 |

## Filters

| Dimension | Values |
|---|---|
| Role | reference, replicate |
| Device | NVIDIA L40S, Orin |
| Dtype | fp16 |
| Stack | 2.10.0+cu128 / 12.8, 2.11.0+cu126 / 12.6, 2.11.0+cu128 / 12.8, 2.7.0 / 12.8 |
| Resolution | 126, 127, 128, 254, 255, 256 |
| Eager outcome | direct_win, repair_win |
| Compiled outcome | direct_win, repair_win, unavailable |

## Reference measurements

| Bundle | Device | PyTorch / CUDA | Res | N%8 | FP16 residue tier prior | Profiler | Direct observed token(s) | Repair observed token(s) | Safety action | Eager oracle | Eager ratio | Compiled oracle | Compiled ratio |
|---|---|---|---:|---:|---|---|---|---|---|---|---:|---|---:|
| [reference/software_stack_matrix/torch2.10_cuda12.8](results/reference_l40s/software_stack_matrix/torch2.10_cuda12.8/SUMMARY.md) | NVIDIA L40S | 2.10.0+cu128 / 12.8 | 256 | 4 | intermediate: align2 | legacy_full_module | align2+align8+ldg8 | align8+ldg8 | repair_kv | repair_kv | 1.109x | repair_kv | 1.162x |
| [reference/software_stack_matrix/torch2.10_cuda12.8](results/reference_l40s/software_stack_matrix/torch2.10_cuda12.8/SUMMARY.md) | NVIDIA L40S | 2.10.0+cu128 / 12.8 | 128 | 4 | intermediate: align2 | unavailable | — | — | repair_kv | direct | 0.899x | repair_kv | 1.127x |
| [reference/software_stack_matrix/torch2.11_cuda12.6](results/reference_l40s/software_stack_matrix/torch2.11_cuda12.6/SUMMARY.md) | NVIDIA L40S | 2.11.0+cu126 / 12.6 | 256 | 4 | intermediate: align2 | legacy_full_module | align2+align8+ldg8 | align8+ldg8 | repair_kv | repair_kv | 1.106x | repair_kv | 1.179x |
| [reference/software_stack_matrix/torch2.11_cuda12.6](results/reference_l40s/software_stack_matrix/torch2.11_cuda12.6/SUMMARY.md) | NVIDIA L40S | 2.11.0+cu126 / 12.6 | 128 | 4 | intermediate: align2 | unavailable | — | — | repair_kv | direct | 0.898x | repair_kv | 1.115x |
| [reference/software_stack_matrix/torch2.11_cuda12.8](results/reference_l40s/software_stack_matrix/torch2.11_cuda12.8/SUMMARY.md) | NVIDIA L40S | 2.11.0+cu128 / 12.8 | 256 | 4 | intermediate: align2 | legacy_full_module | align2+align8+ldg8 | align8+ldg8 | repair_kv | repair_kv | 1.103x | repair_kv | 1.165x |
| [reference/software_stack_matrix/torch2.11_cuda12.8](results/reference_l40s/software_stack_matrix/torch2.11_cuda12.8/SUMMARY.md) | NVIDIA L40S | 2.11.0+cu128 / 12.8 | 128 | 4 | intermediate: align2 | unavailable | — | — | repair_kv | direct | 0.894x | repair_kv | 1.117x |
| [reference/three_level_sweep/torch2.11_cuda12.8](results/reference_l40s/three_level_sweep/torch2.11_cuda12.8/SUMMARY.md) | NVIDIA L40S | 2.11.0+cu128 / 12.8 | 256 | 4 | intermediate: align2 | isolated_ktv | align2 | align8 | repair_kv | repair_kv | 1.107x | repair_kv | 1.176x |
| [reference/three_level_sweep/torch2.11_cuda12.8](results/reference_l40s/three_level_sweep/torch2.11_cuda12.8/SUMMARY.md) | NVIDIA L40S | 2.11.0+cu128 / 12.8 | 255 | 5 | slowest: align1 | isolated_ktv | align1 | align8 | repair_kv | repair_kv | 1.374x | direct | 0.978x |
| [reference/three_level_sweep/torch2.11_cuda12.8](results/reference_l40s/three_level_sweep/torch2.11_cuda12.8/SUMMARY.md) | NVIDIA L40S | 2.11.0+cu128 / 12.8 | 254 | 0 | fastest: align8/ldg8 | isolated_ktv | align8 | align8 | direct | repair_kv | 1.054x | direct | 0.972x |
| [reference/three_level_sweep/torch2.11_cuda12.8](results/reference_l40s/three_level_sweep/torch2.11_cuda12.8/SUMMARY.md) | NVIDIA L40S | 2.11.0+cu128 / 12.8 | 128 | 4 | intermediate: align2 | isolated_ktv | align2 | align8 | repair_kv | direct | 0.900x | repair_kv | 1.112x |
| [reference/three_level_sweep/torch2.11_cuda12.8](results/reference_l40s/three_level_sweep/torch2.11_cuda12.8/SUMMARY.md) | NVIDIA L40S | 2.11.0+cu128 / 12.8 | 127 | 5 | slowest: align1 | isolated_ktv | align1 | align8 | repair_kv | direct | 0.901x | direct | 0.985x |
| [reference/three_level_sweep/torch2.11_cuda12.8](results/reference_l40s/three_level_sweep/torch2.11_cuda12.8/SUMMARY.md) | NVIDIA L40S | 2.11.0+cu128 / 12.8 | 126 | 0 | fastest: align8/ldg8 | isolated_ktv | align8 | align8 | direct | direct | 0.903x | direct | 0.974x |
| [reference/orin/orin_torch2.7_cuda12.8_2026-08-31](results/reference_orin/orin_torch2.7_cuda12.8_2026-08-31/SUMMARY.md) | Orin | 2.7.0 / 12.8 | 128 | 4 | intermediate: align2 | legacy_full_module | align2+ldg8 | ldg8 | repair_kv | direct | 0.876x | unavailable | — |

## Community measurements

None.

## Replicate measurements

| Bundle | Device | PyTorch / CUDA | Res | N%8 | FP16 residue tier prior | Profiler | Direct observed token(s) | Repair observed token(s) | Safety action | Eager oracle | Eager ratio | Compiled oracle | Compiled ratio |
|---|---|---|---:|---:|---|---|---|---|---|---|---:|---|---:|
| [reference/v0_1_bundle](results/reference_l40s/v0_1_bundle/SUMMARY.md) (replicate of reference/software_stack_matrix/torch2.11_cuda12.8) | NVIDIA L40S | 2.11.0+cu128 / 12.8 | 256 | 4 | intermediate: align2 | legacy_full_module | align2+align8+ldg8 | align8+ldg8 | repair_kv | repair_kv | 1.104x | repair_kv | 1.184x |
| [reference/v0_1_bundle](results/reference_l40s/v0_1_bundle/SUMMARY.md) (replicate of reference/software_stack_matrix/torch2.11_cuda12.8) | NVIDIA L40S | 2.11.0+cu128 / 12.8 | 128 | 4 | intermediate: align2 | unavailable | — | — | repair_kv | direct | 0.900x | repair_kv | 1.140x |

## Standalone mechanism audits

These validator-backed artifacts are excluded from workload, device, and
profitability-row counts because they are factorial mechanism controls:

- [L40S compiled six-shape audit](results/reference_l40s/compile_audit/torch2.11_cuda12.8/SUMMARY.md)
- [L40S 100-cell operand-pointer audit](results/reference_l40s/pointer_alignment/torch2.11_cuda12.8/SUMMARY.md)
- [Orin 100-cell operand-pointer audit](results/reference_orin/pointer_alignment/torch2.7_cuda12.8/SUMMARY.md)

## Interpretation boundary

The scientific object is producer layout → vendor GEMM family, not a single
named operator. Public LinearAttention graphs are witnesses. The dedicated
L40S sweep spans fastest/intermediate/slowest residue classes; older 128/256
rows are retained as legacy profitability evidence, not isolated KTV proof.
Positive and negative outcomes are both evidence. Replicates are not extra
devices. Compiled-unavailable is not a direct/repair loss. Matching L40S
and Orin pointer audits reproduce the bounded least-aligned-tier rule in
all 200 controlled cells. This mechanism transfer is separate from repair
profitability: Orin has only one eager module row and no compiled result.
