# Compiled mechanism audit

Eager profiling can show an `align2` → `align8` kernel-family change after repair.
Compiled direct vs repair-KV still needs graph and profiler evidence. Latency alone
must not be used to name a kernel family.

## Protocol

```bash
layoutabi audit-compile --output results/local_compile_audit --resolutions 256,128
layoutabi validate-audit results/local_compile_audit
```

Each compiled policy runs in an isolated process and cache directory. The audit writes:

- FX graph and, when `torch.export` succeeds, the exported graph
- TorchInductor debug files (`ir_pre_fusion.txt`, `ir_post_fusion.txt`, `output_code.py`)
- ordered CUDA profiler names
- a structured `evidence` object: copy present, copy fused/eliminated, first and second
  GEMM family, generated strides, and whether inserted materialization survived lowering
- full-module CUDA-event latency and the existing correctness gate

Nsight Systems or Nsight Compute is optional. Use it only for cells where graph and
profiler evidence still cannot separate KTV, the second GEMM, fusion, and allocation
effects. Do not NCU-scan a full matrix.

## Research questions

- Why can eager module repair lose at 128 while compiled repair wins?
- Does Inductor keep the BHND materialization inserted by repair?
- Does a compiled win come from the KTV GEMM, the second GEMM, fusion, or a broader
  layout/allocation change?
- Which generated kernel or graph explains a 2.10 vs 2.11 absolute-latency gap?

If the compiled mechanism differs from eager, record it as a separate model. Do not
reuse the eager `align2`/`align8` story without compiled graph or profiler evidence.

## Scope

This protocol audits the bundled LinearAttention `direct` and `repair_kv` compiled
cells. It is not a general Inductor debugger and does not modify TorchInductor source.
