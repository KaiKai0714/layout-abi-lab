# Layout ABI reproduction summary

Device: **NVIDIA L40S**

## Order-balanced eager results

| Resolution | Scope | Direct ms | Repair-K ms | Repair-KV ms | Direct / Repair-KV | Repair wins every seed |
|---:|---|---:|---:|---:|---:|---|
| 256 | chain | 0.377771 | 0.476941 | 0.394880 | 0.957 | False |
| 256 | module | 0.664357 | 0.690235 | 0.600817 | 1.106 | True |
| 128 | chain | 0.094207 | 0.104467 | 0.090064 | 1.046 | True |
| 128 | module | 0.245469 | 0.260013 | 0.273252 | 0.898 | False |

## Isolated torch.compile results

| Resolution | Direct ms | Repair-KV ms | Direct / Repair-KV |
|---:|---:|---:|---:|
| 256 | 0.438666 | 0.372043 | 1.179 |
| 128 | 0.141012 | 0.126438 | 1.115 |

A ratio above 1 means repair-KV was faster. This result is scoped to the
recorded graph, device, shape, dtype, and software stack.
