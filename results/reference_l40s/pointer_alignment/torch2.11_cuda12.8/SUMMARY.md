# Operand pointer-alignment audit

Device: **NVIDIA L40S**

Each matrix cell is `family / median ms`. Rows vary K-pointer residue;
columns vary V-pointer residue. Logical shapes and strides stay fixed
within each N. Setup copies are excluded from timing.

## N=65536 (N%8=0, fastest)

| K ptr%64 \ V ptr%64 | 0 | 2 | 4 | 8 | 16 |
|---:|---:|---:|---:|---:|---:|
| 0 | align8 / 0.148634 | align1 / 0.352358 | align2 / 0.182426 | align2 / 0.182682 | align8 / 0.148738 |
| 2 | align1 / 0.300902 | align1 / 0.352331 | align1 / 0.353331 | align1 / 0.349901 | align1 / 0.301388 |
| 4 | align2 / 0.182733 | align1 / 0.352461 | align2 / 0.182630 | align2 / 0.182989 | align2 / 0.183245 |
| 8 | align2 / 0.183320 | align1 / 0.352358 | align2 / 0.183374 | align2 / 0.183194 | align2 / 0.184218 |
| 16 | align8 / 0.148685 | align1 / 0.352896 | align2 / 0.184678 | align2 / 0.184678 | align8 / 0.148685 |

## N=65538 (N%8=2, intermediate)

| K ptr%64 \ V ptr%64 | 0 | 2 | 4 | 8 | 16 |
|---:|---:|---:|---:|---:|---:|
| 0 | align2 / 0.182302 | align1 / 0.355248 | align2 / 0.182628 | align2 / 0.182272 | align2 / 0.182640 |
| 2 | align1 / 0.303400 | align1 / 0.355209 | align1 / 0.356277 | align1 / 0.351770 | align1 / 0.304194 |
| 4 | align2 / 0.182327 | align1 / 0.355127 | align2 / 0.182328 | align2 / 0.181919 | align2 / 0.182343 |
| 8 | align2 / 0.182030 | align1 / 0.355524 | align2 / 0.181930 | align2 / 0.182290 | align2 / 0.181932 |
| 16 | align2 / 0.181743 | align1 / 0.355847 | align2 / 0.181714 | align2 / 0.181454 | align2 / 0.181789 |

## N=65540 (N%8=4, intermediate)

| K ptr%64 \ V ptr%64 | 0 | 2 | 4 | 8 | 16 |
|---:|---:|---:|---:|---:|---:|
| 0 | align2 / 0.182865 | align1 / 0.354769 | align2 / 0.182923 | align2 / 0.183098 | align2 / 0.183666 |
| 2 | align1 / 0.303027 | align1 / 0.355220 | align1 / 0.356734 | align1 / 0.352079 | align1 / 0.303650 |
| 4 | align2 / 0.182906 | align1 / 0.355902 | align2 / 0.183027 | align2 / 0.183082 | align2 / 0.183551 |
| 8 | align2 / 0.182955 | align1 / 0.354503 | align2 / 0.183092 | align2 / 0.183288 | align2 / 0.183574 |
| 16 | align2 / 0.182887 | align1 / 0.355489 | align2 / 0.183054 | align2 / 0.183212 | align2 / 0.183853 |

## N=65543 (N%8=7, slowest)

| K ptr%64 \ V ptr%64 | 0 | 2 | 4 | 8 | 16 |
|---:|---:|---:|---:|---:|---:|
| 0 | align1 / 0.303290 | align1 / 0.349688 | align1 / 0.349708 | align1 / 0.345766 | align1 / 0.300788 |
| 2 | align1 / 0.300943 | align1 / 0.349354 | align1 / 0.349466 | align1 / 0.345394 | align1 / 0.300898 |
| 4 | align1 / 0.300899 | align1 / 0.349102 | align1 / 0.349203 | align1 / 0.345392 | align1 / 0.300713 |
| 8 | align1 / 0.300882 | align1 / 0.349102 | align1 / 0.349258 | align1 / 0.345467 | align1 / 0.300668 |
| 16 | align1 / 0.300830 | align1 / 0.349046 | align1 / 0.349223 | align1 / 0.345418 | align1 / 0.300618 |

Kernel families come from CUDA profiler names, never from latency.
This audit characterizes an isolated FP16 consumer GEMM; it does not
by itself establish full-module repair profitability.
