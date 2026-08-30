# Layout ABI reproduction summary

Device: **NVIDIA L40S**

## Order-balanced eager results

| Resolution | Scope | Direct ms | Repair-K ms | Repair-KV ms | Direct / Repair-KV | Repair wins every seed |
|---:|---|---:|---:|---:|---:|---|
| 256 | chain | 0.378740 | 0.477611 | 0.394243 | 0.961 | False |
| 256 | module | 0.668996 | 0.692436 | 0.603073 | 1.109 | True |
| 128 | chain | 0.094063 | 0.104411 | 0.089921 | 1.046 | True |
| 128 | module | 0.247257 | 0.261657 | 0.275100 | 0.899 | False |

## Isolated torch.compile results

| Resolution | Direct ms | Repair-KV ms | Direct / Repair-KV |
|---:|---:|---:|---:|
| 256 | 0.486926 | 0.419005 | 1.162 |
| 128 | 0.176495 | 0.156608 | 1.127 |

A ratio above 1 means repair-KV was faster. This result is scoped to the
recorded graph, device, shape, dtype, and software stack.
