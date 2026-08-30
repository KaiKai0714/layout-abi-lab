"""Frozen v0.3 pattern contract for the audited LinearAttention KTV GEMM."""

from __future__ import annotations

PATTERN_ID = "linear_attention_ktv_v1"
CANDIDATE_IMPL_HASH = "bhnd_transpose_contiguous_transpose_v1"
SUPPORTED_POLICIES = ("off", "direct", "repair_k", "repair_kv", "autotune")
REWRITE_POLICIES = ("direct", "repair_k", "repair_kv")
REQUIRED_DTYPE_NAME = "float16"
CORRECTNESS_TOLERANCE = 0.08

# Rank-4 producer K with softmax on the last (sequence) dimension, consumed by
# K @ V.transpose(-2, -1), then context.transpose(-2, -1) @ Q.
PATTERN_SUMMARY = (
    "Match a rank-4 K-softmax (dim=-1) that feeds K @ V.transpose(-2, -1), "
    "optionally followed by context.transpose(-2, -1) @ Q. Repair materializes "
    "K and optionally V as BHND-backed logical BHDN storage before the first GEMM."
)
