"""Human-readable explanations of optimizer decisions. No PyTorch import."""

from __future__ import annotations

from typing import Any

_REASON_TEXT = {
    "user_policy": "The caller selected this rewrite policy explicitly.",
    "cache_hit": "A previous measured decision for this cache key was reused.",
    "autotune_fastest": "Full-module CUDA-event autotune selected the fastest correct candidate.",
    "autotune_failed": "Autotune ran but could not select a correct candidate; the original graph was kept.",
    "autotune_requires_cuda": "Autotune needs CUDA example inputs; the original graph was kept.",
    "n_mod_8!=0_predicted_unaligned_kernel": (
        "The N-mod-8 hypothesis predicted an unaligned producer layout, so repair-KV was applied. "
        "This is a testable heuristic, not a universal rule."
    ),
    "n_mod_8==0_predicted_aligned_kernel": (
        "The N-mod-8 hypothesis predicted an already aligned layout, so the graph was left direct."
    ),
    "n_mod_8_requires_fp16_and_known_n": (
        "N-mod-8 only applies to FP16 with a known reduction size; the graph was left direct."
    ),
    "cost_model_non_fp16_boundary": "The cost model treats non-FP16 as a direct/no-repair boundary.",
    "cost_model_batch_boundary": "The cost model treats batch != 1 as a direct/no-repair boundary.",
    "cost_model_aligned_n_skip_autotune": "Aligned N skipped autotune; the graph was left direct.",
    "cost_model_misaligned_n_profile": "Misaligned N fell back to profile-guided autotune.",
    "cost_model_misaligned_n_without_cuda": (
        "Misaligned N without CUDA cannot autotune; repair-KV was applied as a fallback."
    ),
    "unseen_shape": "The example size is outside published shape buckets; unverified repair was not applied.",
    "sync_autotune_disabled": (
        "Synchronous autotune is disabled for this latency-critical call; "
        "the cache or the unseen-shape action was used instead of measuring."
    ),
    "no_supported_pattern": "No supported K-softmax to KTV GEMM pattern was matched; the original graph was kept.",
    "guard_failed": "Inference, dtype, or shape guards rejected the graph; the original module was kept.",
    "correctness_failed": "A rewrite candidate failed the numerical canary; the original graph was kept.",
    "exception": "Capture or rewrite raised; the original executable module was returned.",
    "off": "Rewrite is turned off.",
    "unsupported": "The graph is unsupported.",
    "inspect": "Inspect-only; no rewrite was applied.",
}


def _from_object(decision: Any) -> tuple[str, str, dict[str, Any]]:
    if hasattr(decision, "decision") and hasattr(decision, "diagnostics"):
        diagnostics = dict(getattr(decision, "diagnostics") or {})
        return str(decision.decision), str(diagnostics.get("reason") or ""), diagnostics
    if isinstance(decision, dict):
        return (
            str(decision.get("decision") or "unknown"),
            str(decision.get("reason") or ""),
            decision,
        )
    text = str(decision)
    return text, text, {}


def explain(decision: Any) -> str:
    """Return a short English explanation of an optimize/inspect decision."""

    action, reason, diagnostics = _from_object(decision)
    if reason.startswith("matched "):
        body = (
            f"The bounded pattern {reason[len('matched '):]} was found. "
            "No rewrite is applied in inspect mode."
        )
    else:
        body = _REASON_TEXT.get(reason, f"Decision reason {reason!r}.")
    lines = [f"decision: {action}", body]
    matches = diagnostics.get("matches") or []
    if matches:
        lines.append(f"matched sites: {len(matches)}")
    cache = diagnostics.get("cache") or {}
    if cache.get("hit"):
        lines.append("cache: hit")
    elif cache.get("key"):
        lines.append("cache: miss")
    timings = diagnostics.get("timings_ms") or {}
    if timings.get("autotune") is not None:
        lines.append(f"autotune_ms: {timings['autotune']:.3f}")
    break_even = diagnostics.get("break_even_invocations")
    if break_even is not None:
        lines.append(f"break_even_invocations: {break_even}")
    return "\n".join(lines)
