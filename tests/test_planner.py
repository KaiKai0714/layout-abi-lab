"""CPU tests for N % 8 evaluation and the conservative cost model."""

from __future__ import annotations

import unittest
from pathlib import Path

from layoutabi.planner.evaluate import evaluate_index, n_mod_8_separates_oracle
from layoutabi.planner.features import features_from_sizes
from layoutabi.planner.metrics import score_predictions
from layoutabi.planner.policies import decide


class PlannerPolicyTest(unittest.TestCase):
    def test_n_mod_8_repairs_only_when_fp16_n_is_unaligned(self) -> None:
        aligned = features_from_sizes(n=16, dtype="fp16")
        unaligned = features_from_sizes(n=20, dtype="fp16")
        self.assertEqual(decide(aligned, "n_mod_8")["action"], "direct")
        self.assertEqual(decide(unaligned, "n_mod_8")["action"], "repair_kv")
        bf16 = features_from_sizes(n=20, dtype="bf16")
        self.assertEqual(decide(bf16, "n_mod_8")["action"], "direct")

    def test_cost_model_does_not_statically_repair_misaligned_cells(self) -> None:
        misaligned = features_from_sizes(n=20, dtype="fp16", batch=1, cuda=True)
        planned = decide(misaligned, "cost_model")
        self.assertEqual(planned["action"], "autotune")
        aligned = features_from_sizes(n=16, dtype="fp16", batch=1, cuda=True)
        self.assertEqual(decide(aligned, "cost_model")["action"], "direct")
        batched = features_from_sizes(n=20, dtype="fp16", batch=2, cuda=True)
        self.assertEqual(decide(batched, "cost_model")["action"], "direct")

    def test_false_repair_and_regret_math(self) -> None:
        rows = [
            {
                "oracle": "direct",
                "action": "repair_kv",
                "direct_ms": 1.0,
                "repair_ms": 1.2,
            },
            {
                "oracle": "repair_kv",
                "action": "repair_kv",
                "direct_ms": 2.0,
                "repair_ms": 1.0,
            },
        ]
        metrics = score_predictions(rows)
        self.assertEqual(metrics["false_repair_count"], 1)
        self.assertEqual(metrics["oracle_match_rate"], 0.5)
        self.assertAlmostEqual(metrics["geomean_regret"], (1.2 * 1.0) ** 0.5)
        self.assertFalse(metrics["pass_false_repair_zero"])

    def test_autotune_prediction_is_scored_as_oracle(self) -> None:
        rows = [
            {
                "oracle": "direct",
                "action": "autotune",
                "direct_ms": 1.0,
                "repair_ms": 1.5,
            }
        ]
        metrics = score_predictions(rows)
        self.assertEqual(metrics["oracle_match_rate"], 1.0)
        self.assertEqual(metrics["false_repair_count"], 0)
        self.assertEqual(metrics["geomean_regret"], 1.0)


class PlannerIndexTest(unittest.TestCase):
    def test_reference_index_shows_n_mod_8_cannot_split_128_and_256(self) -> None:
        from layoutabi.aggregation import build_index

        repo = Path(__file__).resolve().parents[1]
        index = build_index(repo / "results")
        report = evaluate_index(index)
        diagnostic = report["n_mod_8_diagnostic"]
        self.assertEqual(diagnostic["unique_n_mod_8"], [4])
        self.assertFalse(diagnostic["separates_oracle"])
        n_mod = report["all"]["policies"]["n_mod_8"]
        always_repair = report["all"]["policies"]["always_repair_kv"]
        cost = report["all"]["policies"]["cost_model"]
        self.assertGreater(n_mod["false_repair_count"], 0)
        self.assertGreater(always_repair["false_repair_count"], 0)
        self.assertEqual(cost["false_repair_count"], 0)
        self.assertTrue(cost["pass_false_repair_zero"])
        self.assertGreaterEqual(
            cost["oracle_match_rate"], n_mod["oracle_match_rate"]
        )
        self.assertGreaterEqual(
            cost["oracle_match_rate"], always_repair["oracle_match_rate"]
        )

    def test_held_out_128_eager_is_a_false_repair_for_n_mod_8(self) -> None:
        rows = [
            {
                "n_mod_8": 4,
                "oracle": "direct",
                "resolution": 128,
                "scope": "eager_module",
            },
            {
                "n_mod_8": 4,
                "oracle": "repair_kv",
                "resolution": 256,
                "scope": "eager_module",
            },
        ]
        diagnostic = n_mod_8_separates_oracle(rows)
        self.assertFalse(diagnostic["separates_oracle"])


try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class PlannerOptimizeTest(unittest.TestCase):
    def test_n_mod_8_policy_rewrites_unaligned_ktv(self) -> None:
        from layoutabi.optimizer.api import optimize

        class TinyKTV(nn.Module):
            def forward(self, q, k, v):  # type: ignore[no-untyped-def]
                q = q.softmax(dim=-2)
                k = k.softmax(dim=-1)
                context = k @ v.transpose(-2, -1)
                return context.transpose(-2, -1) @ q

        def _inputs(n: int):
            q = torch.randn(1, 4, 32, n - 4, dtype=torch.float16)
            k = torch.randn(1, 4, 32, n, dtype=torch.float16)
            v = torch.randn(1, 4, 32, n, dtype=torch.float16)
            return q, k, v

        model = TinyKTV().eval().half()
        unaligned = optimize(model, _inputs(20), policy="n_mod_8")
        self.assertEqual(unaligned.decision, "repair_kv")
        self.assertEqual(unaligned.diagnostics["planner"]["features"]["n_mod_8"], 4)
        aligned = optimize(model, _inputs(16), policy="n_mod_8")
        self.assertEqual(aligned.decision, "direct")


if __name__ == "__main__":
    unittest.main()
