# Operand pointer-alignment audit

Device: **Orin**

Each matrix cell is `family / median ms`. Rows vary K-pointer residue;
columns vary V-pointer residue. Logical shapes and strides stay fixed
within each N. Setup copies are excluded from timing.

## N=65536 (N%8=0, fastest)

| K ptr%64 \ V ptr%64 | 0 | 2 | 4 | 8 | 16 |
|---:|---:|---:|---:|---:|---:|
| 0 | ldg8 / 0.350091 | align1 / 1.442432 | align2 / 0.639935 | align2 / 0.640031 | ldg8 / 0.350635 |
| 2 | align1 / 1.118376 | align1 / 1.430408 | align1 / 1.458644 | align1 / 1.452707 | align1 / 1.119082 |
| 4 | align2 / 0.651826 | align1 / 1.443694 | align2 / 0.641386 | align2 / 0.650097 | align2 / 0.650082 |
| 8 | align2 / 0.649301 | align1 / 1.434313 | align2 / 0.633206 | align2 / 0.642106 | align2 / 0.648438 |
| 16 | ldg8 / 0.352048 | align1 / 1.436043 | align2 / 0.636432 | align2 / 0.636829 | ldg8 / 0.350424 |

## N=65538 (N%8=2, intermediate)

| K ptr%64 \ V ptr%64 | 0 | 2 | 4 | 8 | 16 |
|---:|---:|---:|---:|---:|---:|
| 0 | align2 / 0.648675 | align1 / 1.435278 | align2 / 0.652907 | align2 / 0.652406 | align2 / 0.650938 |
| 2 | align1 / 1.126636 | align1 / 1.435283 | align1 / 1.445140 | align1 / 1.437474 | align1 / 1.127649 |
| 4 | align2 / 0.650503 | align1 / 1.434623 | align2 / 0.648994 | align2 / 0.647962 | align2 / 0.655068 |
| 8 | align2 / 0.650671 | align1 / 1.440350 | align2 / 0.651813 | align2 / 0.652031 | align2 / 0.652792 |
| 16 | align2 / 0.653232 | align1 / 1.434503 | align2 / 0.655414 | align2 / 0.650872 | align2 / 0.652303 |

## N=65540 (N%8=4, intermediate)

| K ptr%64 \ V ptr%64 | 0 | 2 | 4 | 8 | 16 |
|---:|---:|---:|---:|---:|---:|
| 0 | align2 / 0.648538 | align1 / 1.439098 | align2 / 0.645146 | align2 / 0.647816 | align2 / 0.651890 |
| 2 | align1 / 1.125411 | align1 / 1.447020 | align1 / 1.448066 | align1 / 1.445775 | align1 / 1.129430 |
| 4 | align2 / 0.646075 | align1 / 1.444587 | align2 / 0.648213 | align2 / 0.649306 | align2 / 0.648719 |
| 8 | align2 / 0.647882 | align1 / 1.437318 | align2 / 0.648796 | align2 / 0.656232 | align2 / 0.647200 |
| 16 | align2 / 0.649546 | align1 / 1.449334 | align2 / 0.644741 | align2 / 0.647174 | align2 / 0.647482 |

## N=65543 (N%8=7, slowest)

| K ptr%64 \ V ptr%64 | 0 | 2 | 4 | 8 | 16 |
|---:|---:|---:|---:|---:|---:|
| 0 | align1 / 1.131741 | align1 / 1.201222 | align1 / 1.205002 | align1 / 1.199638 | align1 / 1.125610 |
| 2 | align1 / 1.128862 | align1 / 1.205023 | align1 / 1.215798 | align1 / 1.203878 | align1 / 1.132829 |
| 4 | align1 / 1.133844 | align1 / 1.197154 | align1 / 1.197013 | align1 / 1.194347 | align1 / 1.127125 |
| 8 | align1 / 1.126892 | align1 / 1.202435 | align1 / 1.201082 | align1 / 1.203114 | align1 / 1.130007 |
| 16 | align1 / 1.136351 | align1 / 1.199116 | align1 / 1.201989 | align1 / 1.200106 | align1 / 1.128489 |

Kernel families come from CUDA profiler names, never from latency.
This audit characterizes an isolated FP16 consumer GEMM; it does not
by itself establish full-module repair profitability.
