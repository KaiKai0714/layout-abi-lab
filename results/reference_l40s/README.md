# Frozen L40S reference

This directory records the aggregate values and original per-seed JSON from the
order-balanced E170 confirmation and isolated stock-compile control that motivated this
public artifact. Files beginning with `raw_` retain their original E170 schema so the
published numbers remain auditable.

The public release intentionally labels this as a reference observation rather than a
portable expectation. New local runs use the versioned Layout ABI schema and should be
submitted as validator-compatible result bundles, including their exact environment
fingerprint and raw per-seed cells.

The validated three-stack experiment is summarized in
[`SOFTWARE_STACK_MATRIX.md`](SOFTWARE_STACK_MATRIX.md), with complete bundles under
`software_stack_matrix/`.

The controlled operand-pointer reference is under
`pointer_alignment/torch2.11_cuda12.8/`. Across 100 FP16 cells, its observed
family equals the least-aligned tier among N, K pointer, and V pointer in every
cell. This mechanism audit is separate from the full-module profitability index.

The compiled six-shape evidence is under
`compile_audit/torch2.11_cuda12.8/`. Generated debug paths are normalized to
`<REPO>` and `<AUDIT_CACHE>` for publication; graphs, IR, generated code, kernel
names, timings, and correctness values are unchanged and checksums are recomputed.
