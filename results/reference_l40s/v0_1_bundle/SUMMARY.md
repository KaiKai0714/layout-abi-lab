# Layout ABI reproduction summary

Device: **NVIDIA L40S**

## Order-balanced eager results

| Resolution | Scope | Direct ms | Repair-K ms | Repair-KV ms | Direct / Repair-KV | Repair wins every seed |
|---:|---|---:|---:|---:|---:|---|
| 256 | chain | 0.384640 | 0.474141 | 0.396295 | 0.971 | False |
| 256 | module | 0.664105 | 0.688905 | 0.601472 | 1.104 | True |
| 128 | chain | 0.094173 | 0.104531 | 0.089985 | 1.047 | True |
| 128 | module | 0.248357 | 0.263344 | 0.275872 | 0.900 | False |

## Isolated torch.compile results

| Resolution | Direct ms | Repair-KV ms | Direct / Repair-KV |
|---:|---:|---:|---:|
| 256 | 0.436155 | 0.368314 | 1.184 |
| 128 | 0.141085 | 0.123709 | 1.140 |

A ratio above 1 means repair-KV was faster. This result is scoped to the
recorded graph, device, shape, dtype, and software stack.

