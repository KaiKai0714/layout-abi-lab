"""CPU tests for the v0.3 external graph optimizer. Skipped without PyTorch."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from layoutabi.optimizer.cache import load_entry, store_entry
from layoutabi.optimizer.pattern import PATTERN_ID

if TORCH_AVAILABLE:

    class TinyKTV(nn.Module):
        def forward(self, q, k, v):  # type: ignore[no-untyped-def]
            q = q.softmax(dim=-2)
            k = k.softmax(dim=-1)
            context = k @ v.transpose(-2, -1)
            return context.transpose(-2, -1) @ q

    class TinySDPA(nn.Module):
        def forward(self, q, k, v):  # type: ignore[no-untyped-def]
            attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
            return attn @ v

    class WrongSoftmaxDim(nn.Module):
        def forward(self, q, k, v):  # type: ignore[no-untyped-def]
            q = q.softmax(dim=-2)
            k = k.softmax(dim=-2)
            context = k @ v.transpose(-2, -1)
            return context.transpose(-2, -1) @ q

    class LinearOnly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(8, 8)

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.linear(x)


def _cuda() -> bool:
    return TORCH_AVAILABLE and torch.cuda.is_available()


def _ktv_inputs(device: str = "cpu"):
    q = torch.randn(1, 4, 32, 16, device=device, dtype=torch.float16)
    k = torch.randn(1, 4, 32, 20, device=device, dtype=torch.float16)
    v = torch.randn(1, 4, 32, 20, device=device, dtype=torch.float16)
    return q, k, v


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class OptimizerTest(unittest.TestCase):
    def test_anonymous_module_matches_without_class_name(self) -> None:
        from layoutabi.optimizer.api import inspect

        class OrdinaryKTV(nn.Module):
            def forward(self, q, k, v):  # type: ignore[no-untyped-def]
                q = q.softmax(dim=-2)
                k = k.softmax(dim=-1)
                context = k @ v.transpose(-2, -1)
                return context.transpose(-2, -1) @ q

        payload = inspect(OrdinaryKTV().eval().half(), _ktv_inputs())
        self.assertEqual(payload["reason"].split()[0], "matched")
        self.assertEqual(payload["matches"][0]["pattern_id"], PATTERN_ID)
        self.assertNotIn("OrdinaryKTV", str(payload["matches"]))
        self.assertNotIn("denoising-diffusion-pytorch", str(payload["matches"]))

    def test_public_module_matches_without_policy_switch(self) -> None:
        from layoutabi.optimizer.api import inspect
        from layoutabi.workload import PublicDiffusionLinearAttention

        model = PublicDiffusionLinearAttention(policy="direct").eval().half()
        x = torch.randn(1, 64, 8, 8, dtype=torch.float16)
        payload = inspect(model, (x,))
        self.assertTrue(payload["matches"])
        self.assertEqual(payload["capture_method"], "symbolic_trace")

    def test_negative_controls_are_rejected(self) -> None:
        from layoutabi.optimizer.api import optimize

        qkv = _ktv_inputs()
        sdpa_inputs = tuple(
            torch.randn(1, 4, 16, 32, dtype=torch.float16) for _ in range(3)
        )
        sdpa = optimize(TinySDPA().eval().half(), sdpa_inputs, policy="repair_kv")
        self.assertEqual(sdpa.decision, "noop")
        wrong = optimize(WrongSoftmaxDim().eval().half(), qkv, policy="repair_kv")
        self.assertEqual(wrong.decision, "noop")
        linear = optimize(
            LinearOnly().eval().half(),
            (torch.randn(2, 8, dtype=torch.float16),),
            policy="autotune",
        )
        self.assertEqual(linear.decision, "noop")

    def test_training_mode_is_a_no_op(self) -> None:
        from layoutabi.optimizer.api import optimize

        model = TinyKTV().train().half()
        result = optimize(model, _ktv_inputs(), policy="repair_kv")
        self.assertIs(result.module, model)
        self.assertEqual(result.decision, "noop")
        self.assertEqual(result.diagnostics["reason"], "guard_failed")

    def test_fp32_is_a_no_op(self) -> None:
        from layoutabi.optimizer.api import optimize

        model = TinyKTV().eval()
        inputs = tuple(tensor.float() for tensor in _ktv_inputs())
        result = optimize(model, inputs, policy="repair_kv")
        self.assertEqual(result.decision, "noop")

    def test_repair_candidates_pass_correctness(self) -> None:
        from layoutabi.optimizer.api import optimize

        model = TinyKTV().eval().half()
        inputs = _ktv_inputs()
        with torch.no_grad():
            reference = model(*inputs)
        for policy in ("direct", "repair_k", "repair_kv"):
            result = optimize(model, inputs, policy=policy)
            self.assertEqual(result.decision, policy)
            self.assertTrue(result.diagnostics["candidate_correctness"][policy]["pass"])
            with torch.no_grad():
                value = result.module(*inputs)
            delta = (value.float() - reference.float()).abs().max().item()
            self.assertLessEqual(delta, 0.08)

    def test_exception_or_unsupported_returns_original(self) -> None:
        from layoutabi.optimizer.api import optimize

        model = LinearOnly().eval().half()
        inputs = (torch.randn(2, 8, dtype=torch.float16),)
        result = optimize(model, inputs, policy="autotune")
        self.assertIs(result.module, model)
        self.assertEqual(result.decision, "noop")

    def test_unknown_policy_raises(self) -> None:
        from layoutabi.optimizer.api import optimize

        with self.assertRaises(ValueError):
            optimize(TinyKTV().eval().half(), _ktv_inputs(), policy="always_repair")

    def test_off_policy_does_not_rewrite(self) -> None:
        from layoutabi.optimizer.api import optimize

        model = TinyKTV().eval().half()
        result = optimize(model, _ktv_inputs(), policy="off")
        self.assertIs(result.module, model)
        self.assertEqual(result.decision, "off")

    @unittest.skipUnless(_cuda(), "CUDA autotune is required")
    def test_autotune_and_cache_on_cuda(self) -> None:
        from layoutabi.optimizer.api import optimize

        model = TinyKTV().eval().half().cuda()
        inputs = _ktv_inputs("cuda")
        with tempfile.TemporaryDirectory() as temporary_directory:
            first = optimize(
                model, inputs, policy="autotune", cache_dir=temporary_directory
            )
            self.assertIn(first.decision, {"direct", "repair_k", "repair_kv"})
            self.assertEqual(first.diagnostics["reason"], "autotune_fastest")
            second = optimize(
                model, inputs, policy="autotune", cache_dir=temporary_directory
            )
            self.assertEqual(second.decision, first.decision)
            self.assertTrue(second.diagnostics["cache"]["hit"])


class OptimizerCacheTest(unittest.TestCase):
    def test_decision_cache_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory)
            store_entry(cache_dir, "abc", {"decision": "repair_kv"})
            entry = load_entry(cache_dir, "abc")
            self.assertIsNotNone(entry)
            assert entry is not None
            self.assertEqual(entry["decision"], "repair_kv")
            self.assertIsNone(load_entry(cache_dir, "missing"))


if __name__ == "__main__":
    unittest.main()
