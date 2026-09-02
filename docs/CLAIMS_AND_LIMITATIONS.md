# Scientific claims and boundaries

## Supported observations

Linear Attention (and Efficient Attention) are public witnesses of a broader
layout-to-GEMM contract, not the exclusive scope of the theory.

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
7. In matching L40S and Orin 100-cell FP16 audits, the selected family tier equals
   the least-aligned tier among N, the K pointer, and the V pointer in all 200
   measured cells.

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

Controlled prior experiments identified a three-level FP16 consumer-GEMM ladder:
`N%8==0` maps to `align8`/`ldg8`, even non-multiples to `align2`, and odd N to
`align1`. Pointer alignment can independently trigger the same tiers. The new
artifact sweep profiles the isolated K-transpose-V consumer at every tested length;
exact names remain tied to the recorded software stack.

For the L40S pointer grid, define the tested tiers as:

```text
N tier:       8 if N%8==0, 2 if N is even, otherwise 1
pointer tier: 8 if ptr%16==0, 2 if ptr%4==0, otherwise 1
family tier:  min(N tier, K-pointer tier, V-pointer tier)
```

This rule identified `align8`, `align2`, or `align1` in 100/100 cells. At
N=65536, median isolated latency was 0.148685 ms (`align8`), 0.183219 ms
(`align2`), and 0.352358 ms (`align1`). The rule predicts the discrete family,
not exact latency: K-side and V-side misalignment still differed within `align1`.

On Orin, the same least-aligned-tier rule identified `ldg8`, `align2`, or
`align1` in 100/100 cells. At N=65536, median isolated latency was 0.350530 ms
(`ldg8`), 0.641746 ms (`align2`), and 1.436043 ms (`align1`). This transfers
the bounded family-selection mechanism across two architectures; it does not
make the exact latency ratios or byte quantum universal.

The six-shape compiled audit is deliberately separate. Inductor-generated copies
and layout rewrites caused several compiled direct cells to use `align8` even when
their eager counterpart used a lower tier. An eager mechanism label is therefore
not evidence for a compiled graph without compiler/profiler inspection.

## Claims that are not supported

The current evidence does not support the following claims:

- Repair is always faster than zero-copy execution.
- The three-level FP16 ladder, least-aligned-tier rule, or binary safety action is
  universal across data types, operators, GPUs, offsets, or software stacks. Score
  the policy and inspect isolated KTV names; do not infer one from the other.
- The reference result accelerates a complete diffusion pipeline.
- The method improves model accuracy.
- The project contains a general production-ready compiler pass, or that every
  GEMM producer is rewritten today. The automatic optimizer matches one frozen
  KTV pattern; unmatched graphs no-op.
- The behavior is caused by CUDA alone; PyTorch, cuBLAS/cuBLASLt, Triton, framework
  lowering, and the GPU architecture may all contribute.
- The three L40S stacks constitute extra devices. They are extra software stacks
  on one GPU. Orin is a second architecture; it is not an L40S speedup replica.
- The second public graph (Efficient Attention) is matcher-covered in v0.5; it is not
  a claimed end-to-end or cross-device speedup.
- Ordinary reproduction bundles do not vary pointer alignment. Pointer claims
  require a separately validated `audit-pointer` bundle; shape-residue evidence
  must not be presented as pointer causality.
- That the shipping static planner is pointer-aware. It currently uses N/dtype/batch
  features; the runtime cache records external input pointer classes, not guaranteed
  addresses of internal K/V tensors after framework lowering.

## Known negative and boundary cases

- Published Orin eager 128 prefers direct; `N % 8` predicted repair. Compiled cells
  are unavailable.
- Some BF16 and larger-batch cells prefer direct execution.
- INT8 and FP8 do not follow a single precision-derived alignment formula.
- A repair can improve one GEMM while losing at full-module scope.
- Framework compilation can materialize, fuse, or rewrite a graph and change the result.
