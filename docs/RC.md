# Release candidate

Version 0.9.0 freezes the v1.0 software surface. New matcher patterns are out of
scope until a later RC. Bug fixes, documentation, and community result bundles
are still welcome.

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
- Two independent public match graphs and a public SDPA no-op
- License / secret / private-path scan of git-tracked files

## Still open before v1.0

- A second GPU architecture: a published Orin bundle, an explicit measured
  no-op boundary, or a community held-out device. None of these is claimed yet.
- No known silent incorrect rewrite on supported cells. Report crashes,
  incorrect rewrites, regressions, and unexpected no-ops with the RC issue
  template.

Do not treat this RC as a guarantee that repair is profitable on a new GPU.
Run `layoutabi evaluate-planner` on new bundles instead of assuming `N % 8`.
