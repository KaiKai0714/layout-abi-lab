# Layout ABI result index

This file is generated from checksum-validated reference and community bundles.
A ratio above 1 means repair-KV was faster than direct execution.
Replicates are listed separately and are not counted as additional devices.

## Coverage

| Bundles | Reference | Community | Replicates | Devices | Software stacks | Primary rows | All rows |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 3 | 0 | 1 | 1 | 3 | 6 | 8 |

## Outcomes

Outcome counts use primary bundles only.

| Scope | Repair wins | Direct wins | Unavailable |
|---|---:|---:|---:|
| Eager full module | 3 | 3 | 0 |
| Compiled full module | 6 | 0 | 0 |

## Filters

| Dimension | Values |
|---|---|
| Role | reference, replicate |
| Device | NVIDIA L40S |
| Dtype | fp16 |
| Stack | 2.10.0+cu128 / 12.8, 2.11.0+cu126 / 12.6, 2.11.0+cu128 / 12.8 |
| Resolution | 128, 256 |
| Eager outcome | direct_win, repair_win |
| Compiled outcome | repair_win |

## Reference measurements

| Bundle | Device | PyTorch / CUDA | Resolution | Eager direct | Eager repair-KV | Eager ratio | Compiled direct | Compiled repair-KV | Compiled ratio |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| [reference/software_stack_matrix/torch2.10_cuda12.8](results/reference_l40s/software_stack_matrix/torch2.10_cuda12.8/SUMMARY.md) | NVIDIA L40S | 2.10.0+cu128 / 12.8 | 256 | 0.668996 | 0.603073 | 1.109x | 0.486926 | 0.419005 | 1.162x |
| [reference/software_stack_matrix/torch2.10_cuda12.8](results/reference_l40s/software_stack_matrix/torch2.10_cuda12.8/SUMMARY.md) | NVIDIA L40S | 2.10.0+cu128 / 12.8 | 128 | 0.247257 | 0.275100 | 0.899x | 0.176495 | 0.156608 | 1.127x |
| [reference/software_stack_matrix/torch2.11_cuda12.6](results/reference_l40s/software_stack_matrix/torch2.11_cuda12.6/SUMMARY.md) | NVIDIA L40S | 2.11.0+cu126 / 12.6 | 256 | 0.664357 | 0.600817 | 1.106x | 0.438666 | 0.372043 | 1.179x |
| [reference/software_stack_matrix/torch2.11_cuda12.6](results/reference_l40s/software_stack_matrix/torch2.11_cuda12.6/SUMMARY.md) | NVIDIA L40S | 2.11.0+cu126 / 12.6 | 128 | 0.245469 | 0.273252 | 0.898x | 0.141012 | 0.126438 | 1.115x |
| [reference/software_stack_matrix/torch2.11_cuda12.8](results/reference_l40s/software_stack_matrix/torch2.11_cuda12.8/SUMMARY.md) | NVIDIA L40S | 2.11.0+cu128 / 12.8 | 256 | 0.664245 | 0.602023 | 1.103x | 0.435392 | 0.373582 | 1.165x |
| [reference/software_stack_matrix/torch2.11_cuda12.8](results/reference_l40s/software_stack_matrix/torch2.11_cuda12.8/SUMMARY.md) | NVIDIA L40S | 2.11.0+cu128 / 12.8 | 128 | 0.246949 | 0.276285 | 0.894x | 0.141194 | 0.126380 | 1.117x |

## Community measurements

None.

## Replicate measurements

| Bundle | Device | PyTorch / CUDA | Resolution | Eager direct | Eager repair-KV | Eager ratio | Compiled direct | Compiled repair-KV | Compiled ratio |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| [reference/v0_1_bundle](results/reference_l40s/v0_1_bundle/SUMMARY.md) (replicate of reference/software_stack_matrix/torch2.11_cuda12.8) | NVIDIA L40S | 2.11.0+cu128 / 12.8 | 256 | 0.664105 | 0.601472 | 1.104x | 0.436155 | 0.368314 | 1.184x |
| [reference/v0_1_bundle](results/reference_l40s/v0_1_bundle/SUMMARY.md) (replicate of reference/software_stack_matrix/torch2.11_cuda12.8) | NVIDIA L40S | 2.11.0+cu128 / 12.8 | 128 | 0.248357 | 0.275872 | 0.900x | 0.141085 | 0.123709 | 1.140x |

## Interpretation boundary

Rows are scoped to their recorded graph, device, shape, dtype, and software stack.
They are not end-to-end model results. Positive and negative outcomes are both
part of the public evidence. Same-device same-stack replicates are not extra
cross-device evidence.
