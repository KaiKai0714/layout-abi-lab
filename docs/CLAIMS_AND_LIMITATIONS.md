# Scientific claims and boundaries

## Supported observations

The experiments support the following bounded observations:

1. Numerically equivalent producer layouts can select different opaque vendor GEMM
   kernel families.
2. Kernel-family changes can create discrete latency cliffs without changing the
   mathematical operation or nominal FLOP count.
3. An explicit layout materialization can be profitable when its cost is smaller than
   the downstream GEMM improvement.
4. Profitability must be evaluated at graph or module scope. An isolated-chain result
   does not reliably predict the full module.
5. Stock compilation does not eliminate every profitable materialization opportunity in
   the current L40S reference software stack.
6. The L40S mechanism and win/loss boundary reproduce across PyTorch 2.10/2.11 and
   CUDA 12.6/12.8 builds in the tested container matrix.

## Reference L40S observation

The order-balanced eager confirmation at FP16, batch 1, and 256x256 measured:

| Scope | Direct | Repair-KV | Direct / Repair-KV |
|---|---:|---:|---:|
| Isolated chain | 0.362731 ms | 0.394453 ms | 0.920x |
| Full LinearAttention module | 0.628408 ms | 0.558692 ms | 1.125x |

The isolated compiled control measured:

| Resolution | Compiled direct | Compiled repair-KV | Direct / Repair-KV |
|---:|---:|---:|---:|
| 256 | 0.434739 ms | 0.369050 ms | 1.178x |
| 128 | 0.143462 ms | 0.126029 ms | 1.138x |

The profiler associated the direct K-transpose-V path with an `align2` CUTLASS-family
kernel and the repaired path with an `align8` family plus an explicit copy. Exact names
are implementation details and may change across software releases.

## Claims that are not supported

The current evidence does not support the following claims:

- Repair is always faster than zero-copy execution.
- `N % 8` is a universal rule across data types, operators, or GPUs. Score it with
  `layoutabi evaluate-planner`; do not treat a match on L40S FP16 as proof.
- The reference result accelerates a complete diffusion pipeline.
- The method improves model accuracy.
- The project contains a general production-ready compiler pass.
- The behavior is caused by CUDA alone; PyTorch, cuBLAS/cuBLASLt, Triton, framework
  lowering, and the GPU architecture may all contribute.
- The three L40S stacks constitute cross-device or cross-architecture validation.
- The second public graph (Efficient Attention) is matcher-covered in v0.5; it is not
  a claimed end-to-end or cross-device speedup.

## Known negative and boundary cases

- Some Orin measurements prefer direct execution.
- Some BF16 and larger-batch cells prefer direct execution.
- INT8 and FP8 do not follow a single precision-derived alignment formula.
- A repair can improve one GEMM while losing at full-module scope.
- Framework compilation can materialize, fuse, or rewrite a graph and change the result.
