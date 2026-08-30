# Layout ABI reproduction summary

Device: **NVIDIA L40S**

## Order-balanced eager results

| Resolution | Scope | Direct ms | Repair-K ms | Repair-KV ms | Direct / Repair-KV | Repair wins every seed |
|---:|---|---:|---:|---:|---:|---|
| 256 | chain | 0.379833 | 0.475349 | 0.394221 | 0.964 | False |
| 256 | module | 0.664245 | 0.688707 | 0.602023 | 1.103 | True |
| 128 | chain | 0.094029 | 0.104193 | 0.089748 | 1.048 | True |
| 128 | module | 0.246949 | 0.263304 | 0.276285 | 0.894 | False |

## Isolated torch.compile results

| Resolution | Direct ms | Repair-KV ms | Direct / Repair-KV |
|---:|---:|---:|---:|
| 256 | 0.435392 | 0.373582 | 1.165 |
| 128 | 0.141194 | 0.126380 | 1.117 |

A ratio above 1 means repair-KV was faster. This result is scoped to the
recorded graph, device, shape, dtype, and software stack.
