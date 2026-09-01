# Compiled mechanism audit

Kernel families and copies are taken from profiler names and Inductor IR,
not inferred from latency. A compiled win/loss needs this evidence before
reusing an eager explanation.

| Resolution | Policy | Available | Steady ms | Copy (profiler) | Copy post-fusion | Copy fused/eliminated | First GEMM | Second GEMM |
|---:|---|---|---:|---|---|---|---|---|
| 256 | direct | yes | 0.435174 | True | False | False | gemm | align2 |
| 256 | repair_kv | yes | 0.367462 | True | False | False | gemm | align8 |
| 255 | direct | yes | 0.381694 | True | False | False | gemm | align8 |
| 255 | repair_kv | yes | 0.389169 | True | False | False | gemm | align8 |
| 254 | direct | yes | 0.359347 | True | False | False | gemm | align8 |
| 254 | repair_kv | yes | 0.374755 | True | False | False | gemm | align8 |
| 128 | direct | yes | 0.140826 | True | False | False | gemm | align2 |
| 128 | repair_kv | yes | 0.123418 | True | False | False | gemm | align8 |
| 127 | direct | yes | 0.123878 | True | False | False | gemm | align8 |
| 127 | repair_kv | yes | 0.125258 | True | False | False | gemm | align8 |
| 126 | direct | yes | 0.119319 | True | False | False | gemm | align8 |
| 126 | repair_kv | yes | 0.123162 | True | False | False | gemm | align8 |

Nsight Systems/Compute is optional and only for cells where graph and
profiler evidence still cannot separate KTV, the second GEMM, and fusion.
See docs/COMPILE_AUDIT.md.
