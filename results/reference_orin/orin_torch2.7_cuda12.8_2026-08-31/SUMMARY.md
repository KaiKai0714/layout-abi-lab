# Layout ABI reproduction summary

Device: **Orin**

## Order-balanced eager results

| Resolution | Scope | Direct ms | Repair-K ms | Repair-KV ms | Direct / Repair-KV | Repair wins every seed |
|---:|---|---:|---:|---:|---:|---|
| 128 | chain | 1.036589 | 1.283793 | 1.391020 | 0.745 | False |
| 128 | module | 2.484399 | 2.731245 | 2.837493 | 0.876 | False |

A ratio above 1 means repair-KV was faster. This result is scoped to the
recorded graph, device, shape, dtype, and software stack.
