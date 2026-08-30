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
