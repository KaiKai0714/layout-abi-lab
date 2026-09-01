# Release candidate

Version 0.9.1 is the current release candidate (same frozen API and matcher as
0.9.0). New matcher patterns are out of scope until a later RC. Bug fixes,
documentation, and community result bundles are still welcome.

The research object is producer layout → vendor GEMM family. Linear Attention
is a public witness. The automatic optimizer still rewrites only
`linear_attention_ktv_v1`.

```bash
layoutabi rc-status --check
layoutabi scan-release
```

## Frozen

- Public API: `optimize`, `inspect`, `clear_cache`, `explain`, `supported`,
  and `cache_info`
- Pattern `linear_attention_ktv_v1` and candidate implementation
  `bhnd_transpose_contiguous_transpose_v1`
- Result and diagnostics JSON Schema versions
- Optimizer cache protocol v2
- Repair remains a correctness-gated no-op on unsupported graphs

A drift in those values fails `layoutabi rc-status --check`.

## Verified in this repository

- CPU package, schema, index, workload, planner, and cache tests
- Published L40S reference bundles (one device; extra software stacks are not
  extra devices)
- Published L40S six-shape compiled audit and 100-cell operand-pointer audit
- Two independent public match graphs and a public SDPA no-op
- License / secret / private-path scan of git-tracked files

## Measurement gates completed

- The L40S three-level sweep spans `N%8==0`, even non-multiples, and odd N.
  Isolated KTV profiler names reproduce `align8`, `align2`, and `align1`.
- Orin eager 128 supplies a second-architecture no-repair boundary.
- L40S pointer-family selection follows the least-aligned N/K/V tier in 100/100
  controlled cells.

The remaining v1 mechanism gate is a published, cross-device controlled
K-pointer × V-pointer audit. Its CLI and validator are frozen; the L40S 100-cell
grid is published and the matching Orin measurement remains open.

No known silent incorrect rewrite exists on supported cells. Continue reporting
crashes, incorrect rewrites, regressions, and unexpected no-ops with the RC
issue template.

Orin eager 128 is in `results/community/`. Repair is slower there; compiled
cells are unavailable. That is a second architecture and a no-repair boundary,
not an L40S speedup replica.

The community path is: reproduce the three residue tiers on another device,
submit the full bundle, score the binary safety planner with
`layoutabi evaluate-planner`, and file RC issues
for crashes or wrong rewrites. Do not treat this RC as a guarantee that repair
is profitable on a new GPU.
