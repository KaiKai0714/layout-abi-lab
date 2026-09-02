# Frozen v1 release

Version 1.0.0 is the first released artifact. It retains the API and matcher
frozen during 0.9.x. New matcher patterns require a later minor release;
bug fixes, documentation, and community result bundles remain welcome.

The research object is producer layout → vendor GEMM family. Linear Attention
is a public witness. The automatic optimizer still rewrites only
`linear_attention_ktv_v1`.

Repair is a controlled layout intervention used to expose the dispatch
mechanism. Its profitability is secondary and is reported separately at module
scope; this release does not present repair as a universal optimization.

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
- Published L40S six-shape compiled audit and matching L40S/Orin 100-cell
  operand-pointer audits
- Two independent public match graphs and a public SDPA no-op
- License / secret / private-path scan of git-tracked files

## Measurement gates completed

- The L40S three-level sweep spans `N%8==0`, even non-multiples, and odd N.
  Isolated KTV profiler names reproduce `align8`, `align2`, and `align1`.
- Orin eager 128 supplies a second-architecture no-repair boundary.
- L40S and Orin pointer-family selection follows the least-aligned N/K/V tier
  in 200/200 controlled cells across the two matching audits.

No known silent incorrect rewrite exists on supported cells. Continue reporting
crashes, incorrect rewrites, regressions, and unexpected no-ops with the issue
template.

The author-run Orin eager 128 result is in `results/reference_orin/`. Repair is
slower there; compiled cells and other module resolutions are unavailable. It
is a second-architecture no-repair boundary, not broad Orin profitability
coverage and not an L40S speedup replica.

The community path is: reproduce the three residue tiers on another device,
submit the full bundle, score the binary safety planner with
`layoutabi evaluate-planner`, and file RC issues
for crashes or wrong rewrites. Do not treat this release as a guarantee that
repair is profitable on a new GPU.
