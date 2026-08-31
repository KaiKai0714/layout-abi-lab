"""Stable public API, structured errors, and diagnostics schema tests."""

from __future__ import annotations

import unittest

import layoutabi
from layoutabi.errors import (
    InvalidArgumentError,
    LayoutABIError,
    MissingPyTorchError,
)
from layoutabi.explain import explain
from layoutabi.optimizer.diagnostics import empty_diagnostics, validate_diagnostics
from layoutabi.schema import DIAGNOSTICS_SCHEMA, load_json_schema, validate_schema_instance
from layoutabi.supported import supported

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PublicApiImportTest(unittest.TestCase):
    def test_version_and_exports(self) -> None:
        self.assertTrue(layoutabi.__version__.startswith("0.8."))
        for name in (
            "optimize",
            "inspect",
            "explain",
            "supported",
            "clear_cache",
            "cache_info",
            "LayoutABIError",
            "MissingPyTorchError",
            "InvalidArgumentError",
        ):
            self.assertTrue(hasattr(layoutabi, name))

    def test_missing_pytorch_error_is_import_error(self) -> None:
        exc = MissingPyTorchError("layoutabi.optimize")
        self.assertIsInstance(exc, ImportError)
        self.assertIsInstance(exc, LayoutABIError)
        self.assertIn("does not install PyTorch", str(exc))

    def test_explain_without_pytorch(self) -> None:
        text = explain({"decision": "direct", "reason": "user_policy"})
        self.assertIn("decision: direct", text)
        self.assertIn("explicitly", text)

    def test_supported_without_pytorch(self) -> None:
        payload = supported()
        self.assertEqual(payload["pattern_id"], "linear_attention_ktv_v1")
        self.assertFalse(payload["installs_pytorch"])
        self.assertIn("supported", payload["cpu_tools"])
        self.assertIn("optimize", payload["optimizer_requires_pytorch"])
        self.assertTrue(payload["dtype_requested_supported"])
        bf16 = supported(dtype="bf16")
        self.assertFalse(bf16["dtype_requested_supported"])
        with self.assertRaises(InvalidArgumentError):
            supported(workload="not_a_workload")

    def test_adapter_package_imports_without_capturing(self) -> None:
        from layoutabi.optimizer.adapters import try_export, try_symbolic_trace

        self.assertTrue(callable(try_symbolic_trace))
        self.assertTrue(callable(try_export))

    def test_empty_diagnostics_match_schema(self) -> None:
        payload = empty_diagnostics()
        self.assertEqual(payload["schema"], DIAGNOSTICS_SCHEMA)
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(validate_diagnostics(payload), [])
        self.assertEqual(
            validate_schema_instance(payload, load_json_schema(DIAGNOSTICS_SCHEMA)),
            [],
        )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class PublicApiOptimizeTest(unittest.TestCase):
    def test_invalid_policy_raises_structured_error(self) -> None:
        class Tiny(nn.Module):
            def forward(self, x):  # type: ignore[no-untyped-def]
                return x

        model = Tiny().eval().half()
        x = torch.zeros(1, 2, 4, 4, dtype=torch.float16)
        with self.assertRaises(InvalidArgumentError):
            layoutabi.optimize(model, (x,), policy="not_a_policy")
        with self.assertRaises(InvalidArgumentError):
            layoutabi.optimize(model, (x,), shape_mode="fuzzy")
        with self.assertRaises(InvalidArgumentError):
            layoutabi.optimize(model, (x,), unseen_shape="repair_kv")

    def test_inspect_diagnostics_validate(self) -> None:
        class TinyKTV(nn.Module):
            def forward(self, q, k, v):  # type: ignore[no-untyped-def]
                q = q.softmax(dim=-2)
                k = k.softmax(dim=-1)
                context = k @ v.transpose(-2, -1)
                return context.transpose(-2, -1) @ q

        q = torch.randn(1, 4, 32, 16, dtype=torch.float16)
        k = torch.randn(1, 4, 32, 20, dtype=torch.float16)
        v = torch.randn(1, 4, 32, 20, dtype=torch.float16)
        payload = layoutabi.inspect(TinyKTV().eval().half(), (q, k, v))
        self.assertEqual(validate_diagnostics(payload), [])
        self.assertIn("matched", explain(payload))


if __name__ == "__main__":
    unittest.main()
