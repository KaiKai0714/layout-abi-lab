# Layout ABI reproduction summary

Device: **NVIDIA L40S**

## Order-balanced eager results

| Resolution | N%8 | Scope | Direct ms | Repair-K ms | Repair-KV ms | Direct / Repair-KV | Repair wins every seed |
|---:|---:|---|---:|---:|---:|---:|---|
| 256 | 4 | chain | 0.379976 | 0.471988 | 0.395383 | 0.961 | False |
| 256 | 4 | module | 0.663612 | 0.690592 | 0.599517 | 1.107 | True |
| 255 | 5 | chain | 0.609913 | 0.693511 | 0.363955 | 1.676 | True |
| 255 | 5 | module | 0.749481 | 0.809912 | 0.545625 | 1.374 | True |
| 254 | 0 | chain | 0.343015 | 0.340879 | 0.352388 | 0.973 | False |
| 254 | 0 | module | 0.564204 | 0.572303 | 0.535416 | 1.054 | True |
| 128 | 4 | chain | 0.094169 | 0.104476 | 0.090072 | 1.045 | True |
| 128 | 4 | module | 0.242172 | 0.258141 | 0.269013 | 0.900 | False |
| 127 | 5 | chain | 0.134948 | 0.152083 | 0.091471 | 1.475 | True |
| 127 | 5 | module | 0.242837 | 0.256605 | 0.269489 | 0.901 | False |
| 126 | 0 | chain | 0.072001 | 0.078581 | 0.087371 | 0.824 | False |
| 126 | 0 | module | 0.244393 | 0.259211 | 0.270671 | 0.903 | False |

## Isolated torch.compile results

| Resolution | Direct ms | Repair-KV ms | Direct / Repair-KV |
|---:|---:|---:|---:|
| 256 | 0.435434 | 0.370153 | 1.176 |
| 255 | 0.380644 | 0.389382 | 0.978 |
| 254 | 0.365167 | 0.375581 | 0.972 |
| 128 | 0.141042 | 0.126850 | 1.112 |
| 127 | 0.121714 | 0.123551 | 0.985 |
| 126 | 0.117346 | 0.120446 | 0.974 |

A ratio above 1 means repair-KV was faster at that scope. Kernel-family
names live in `eager_results.json` (`ktv_profiler`); they identify the
isolated consumer-GEMM mechanism. Ratio is whether materialization paid off.
This result is scoped to the recorded graph, device, shape, dtype, and stack.
