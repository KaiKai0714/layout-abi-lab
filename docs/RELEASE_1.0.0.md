# v1.0.0 — Cross-architecture mechanism release

Version 1.0.0 freezes the first public Layout ABI Lab artifact. It preserves the
0.9.x optimizer API and single KTV matcher while closing the planned
cross-architecture operand-pointer measurement gate.

## Published evidence

- Three L40S PyTorch/CUDA software stacks for the full-module workload protocol.
- A six-shape L40S sweep spanning the FP16 `align8`, `align2`, and `align1`
  residue classes.
- A six-shape compiled L40S mechanism audit with graph, Inductor IR, copy, and
  profiler evidence.
- Matching 4 × 5 × 5 operand-pointer audits on L40S and Jetson Orin.
- An Orin eager-128 no-repair boundary; compiled Orin remains unavailable.

The pointer grids establish cross-architecture mechanism transfer only. Orin
module-level profitability remains limited to the single eager-128 row, where
direct wins; this release does not claim broad Orin repair profitability.

The two pointer audits contain 200/200 correctness-passing cells and 200/200
identified profiler families. On each tested device, the observed family tier
equals the least-aligned tier among logical N, the K pointer, and the V pointer.
This is a bounded FP16 observation for the recorded GEMM shape and software
stacks, not a universal CUDA alignment rule.

## Embedded-profiler isolation

On the tested Orin stack, repeated Kineto collection in one CUDA process stopped
returning GEMM names after earlier grids. The v1 command runs each requested N in
a fresh subprocess and uses one marker-tagged profiler session for its full
pointer grid. Warm-up and timing remain identical within each child, and process
startup is outside CUDA-event measurements. Independent single-N controls agree
with the merged result in family selection and latency.

## Frozen scope

- Public APIs: `optimize`, `inspect`, `clear_cache`, `explain`, `supported`, and
  `cache_info`.
- Automatic rewrite: fixed-shape FP16 inference for the frozen
  `linear_attention_ktv_v1` matcher.
- Unsupported graphs remain unchanged; profitability is never assumed from the
  family rule alone.
- The static planner is not internal-pointer-aware. Pointer evidence motivates a
  future compiler/runtime guard but does not silently change v1 policy.

## Validation

```bash
python -m unittest discover -s tests -v
python -m compileall layoutabi containers
layoutabi validate-pointer-audit \
  results/reference_l40s/pointer_alignment/torch2.11_cuda12.8
layoutabi validate-pointer-audit \
  results/reference_orin/pointer_alignment/torch2.7_cuda12.8
layoutabi aggregate --check
layoutabi rc-status --check
layoutabi scan-release
```

See [README](../README.md), [claims and limitations](CLAIMS_AND_LIMITATIONS.md),
and the generated [result index](../RESULTS_INDEX.md) for interpretation.
