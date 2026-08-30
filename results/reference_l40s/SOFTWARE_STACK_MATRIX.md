# L40S software-stack matrix

All bundles passed strict schema, checksum, availability, and numerical-correctness
validation. Measurements used one NVIDIA L40S with driver 575.57.08, FP16, batch 1,
three seeds, twelve order-balanced cycles, and twelve iterations per measurement.

## Full-module and compiled results

| PyTorch | CUDA build | cuDNN | Eager 256 direct | Eager 256 repair-KV | Eager reduction | Compiled 256 direct | Compiled 256 repair-KV | Compiled reduction |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2.10.0 | 12.8 | 91002 | 0.668996 ms | 0.603073 ms | 9.85% | 0.486926 ms | 0.419005 ms | 13.95% |
| 2.11.0 | 12.6 | 91002 | 0.664357 ms | 0.600817 ms | 9.56% | 0.438666 ms | 0.372043 ms | 15.19% |
| 2.11.0 | 12.8 | 91900 | 0.664245 ms | 0.602023 ms | 9.37% | 0.435392 ms | 0.373582 ms | 14.20% |

| PyTorch | CUDA build | Compiled 128 direct | Compiled 128 repair-KV | Compiled reduction |
|---|---|---:|---:|---:|
| 2.10.0 | 12.8 | 0.176495 ms | 0.156608 ms | 11.27% |
| 2.11.0 | 12.6 | 0.141012 ms | 0.126438 ms | 10.34% |
| 2.11.0 | 12.8 | 0.141194 ms | 0.126380 ms | 10.49% |

## Mechanism and boundary

Every stack reproduced the same eager kernel-family transition:

```text
direct K-transpose-V: align2 family
repair-KV: explicit copy + align8 family
```

Every stack also reproduced the same graph-scope boundary:

- At 256x256, repair-KV lost in the isolated eager chain but won in the complete eager
  module for every seed.
- At 128x128, repair-KV won in the isolated eager chain but lost in the complete eager
  module for every seed.
- After `torch.compile`, repair-KV won at both tested resolutions in every stack.

The effect therefore survives a PyTorch version change and a CUDA 12 minor-build change,
but the decision cannot be inferred from an isolated chain alone. These results compare
complete software stacks; they do not isolate CUDA, cuBLAS, Triton, or Inductor as a
single causal variable.

