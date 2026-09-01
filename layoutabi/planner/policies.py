"""Frozen interpretable decision rules. Thresholds are set before held-out scoring."""

from __future__ import annotations

from typing import Any

from .features import DecisionFeatures

# Conservative E128-style safety rule. The FP16 mechanism prior has three tiers:
# N%8==0 -> align8/ldg8, even non-multiples -> align2, odd -> align1. This binary
# policy deliberately merges the intermediate and slowest tiers into "repair";
# it is not a complete vendor-family predictor or a universal law.
N_MOD_8_REPAIR = 0
FP16_NAMES = {"fp16", "float16", "torch.float16"}

EVAL_POLICY_NAMES = (
    "always_direct",
    "always_repair_kv",
    "n_mod_8",
    "cost_model",
    "autotune",
)
LIVE_PLANNER_POLICIES = ("n_mod_8", "cost_model")


def _is_fp16(features: DecisionFeatures) -> bool:
    return features.dtype.lower() in FP16_NAMES and features.dtype_bytes in {0, 2}


def decide_always_direct(features: DecisionFeatures, **_kwargs: Any) -> dict[str, Any]:
    return {"action": "direct", "reason": "always_direct"}


def decide_always_repair_kv(features: DecisionFeatures, **_kwargs: Any) -> dict[str, Any]:
    return {"action": "repair_kv", "reason": "always_repair_kv"}


def decide_n_mod_8(features: DecisionFeatures, **_kwargs: Any) -> dict[str, Any]:
    if not _is_fp16(features) or features.n is None:
        return {
            "action": "direct",
            "reason": "n_mod_8_requires_fp16_and_known_n",
        }
    if features.n_mod_8 == N_MOD_8_REPAIR:
        return {
            "action": "direct",
            "reason": "n_mod_8==0_predicted_aligned_kernel",
        }
    return {
        "action": "repair_kv",
        "reason": "n_mod_8!=0_predicted_unaligned_kernel",
    }


def decide_cost_model(
    features: DecisionFeatures, *, allow_autotune: bool = True
) -> dict[str, Any]:
    """Conservative model: skip repair unless a cheap prior is confident.

    Known boundary cells stay direct. Aligned N skips autotune. Misaligned N
    falls back to profile-guided autotune so false-repair is not baked into a
    static rule. Public 128/256 must not be hardcoded.
    """

    if not _is_fp16(features):
        return {"action": "direct", "reason": "cost_model_non_fp16_boundary"}
    if features.batch != 1:
        return {"action": "direct", "reason": "cost_model_batch_boundary"}
    if features.n is None:
        if allow_autotune and features.cuda:
            return {"action": "autotune", "reason": "cost_model_unknown_n"}
        return {"action": "direct", "reason": "cost_model_unknown_n_no_autotune"}
    if features.n_mod_8 == N_MOD_8_REPAIR:
        return {
            "action": "direct",
            "reason": "cost_model_aligned_n_skip_autotune",
        }
    if allow_autotune and features.cuda:
        return {
            "action": "autotune",
            "reason": "cost_model_misaligned_n_profile",
        }
    return {
        "action": "repair_kv",
        "reason": "cost_model_misaligned_n_without_cuda",
    }


def decide_autotune(features: DecisionFeatures, **_kwargs: Any) -> dict[str, Any]:
    if features.cuda:
        return {"action": "autotune", "reason": "profile_guided"}
    return {"action": "direct", "reason": "autotune_requires_cuda"}


DECISION_FNS = {
    "always_direct": decide_always_direct,
    "always_repair_kv": decide_always_repair_kv,
    "n_mod_8": decide_n_mod_8,
    "cost_model": decide_cost_model,
    "autotune": decide_autotune,
}


def decide(features: DecisionFeatures, policy: str, **kwargs: Any) -> dict[str, Any]:
    try:
        fn = DECISION_FNS[policy]
    except KeyError as exc:
        raise ValueError(
            f"Unknown planner policy {policy!r}; expected one of {tuple(DECISION_FNS)}"
        ) from exc
    return fn(features, **kwargs)
