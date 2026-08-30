"""Workload catalog and optimizer generalization tests."""

from __future__ import annotations

import unittest

from layoutabi.workloads import CATALOG, list_workloads, synthetic_cells

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class WorkloadCatalogTest(unittest.TestCase):
    def test_two_independent_public_sources_and_a_negative(self) -> None:
        catalog = {item["id"]: item for item in list_workloads()}
        self.assertEqual(catalog["diffusion_linear_attention"]["role"], "positive_reference")
        self.assertEqual(catalog["efficient_attention"]["role"], "second_public")
        self.assertEqual(catalog["scaled_dot_product"]["role"], "negative")
        self.assertNotEqual(
            catalog["efficient_attention"]["repository"],
            catalog["diffusion_linear_attention"]["repository"],
        )
        self.assertEqual(
            catalog["efficient_attention"]["expected_optimizer"],
            "match",
        )
        self.assertEqual(catalog["scaled_dot_product"]["expected_optimizer"], "noop")

    def test_synthetic_cells_cover_nearby_shapes_without_replacing_public_graphs(self) -> None:
        cells = synthetic_cells()
        resolutions = {cell["resolution"] for cell in cells}
        self.assertTrue({64, 128, 160, 256, 384}.issubset(resolutions))
        self.assertTrue(any(cell["role"] == "mainline" for cell in cells))
        self.assertTrue(any(cell["role"] == "boundary_batch" for cell in cells))
        self.assertTrue(any(cell["role"] == "boundary_unsupported_dtype" for cell in cells))
        self.assertNotIn("synthetic", CATALOG)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class WorkloadOptimizerTest(unittest.TestCase):
    def test_efficient_attention_matches_without_class_or_path_keys(self) -> None:
        from layoutabi.optimizer.api import inspect, optimize
        from layoutabi.workloads import make_workload
        from layoutabi.workloads.efficient_attention import published_loop_forward

        module, inputs = make_workload("efficient_attention", resolution=8)
        payload = inspect(module, inputs)
        self.assertTrue(payload["matches"])
        self.assertNotIn("PublicEfficientAttention", str(payload["matches"]))
        self.assertNotIn("cmsflash", str(payload["matches"]))
        result = optimize(module, inputs, policy="repair_kv")
        self.assertEqual(result.decision, "repair_kv")
        with torch.no_grad():
            reference = module(*inputs)
            looped = published_loop_forward(module, inputs[0])
            rewritten = result.module(*inputs)
        self.assertLessEqual((reference.float() - looped.float()).abs().max().item(), 0.08)
        self.assertLessEqual((rewritten.float() - reference.float()).abs().max().item(), 0.08)

    def test_scaled_dot_product_is_a_public_no_op(self) -> None:
        from layoutabi.optimizer.api import optimize
        from layoutabi.workloads import make_workload

        module, inputs = make_workload("scaled_dot_product", resolution=16)
        result = optimize(module, inputs, policy="repair_kv")
        self.assertIs(result.module, module)
        self.assertEqual(result.decision, "noop")

    def test_bf16_is_a_dtype_boundary(self) -> None:
        from layoutabi.optimizer.api import inspect
        from layoutabi.workloads import make_workload

        module, inputs = make_workload(
            "diffusion_linear_attention", resolution=8, dtype="bf16"
        )
        payload = inspect(module, inputs)
        self.assertEqual(payload["reason"], "guard_failed")
        self.assertTrue(any("float16" in item for item in payload["guard_problems"]))

    def test_batch_two_still_matches_as_a_boundary_shape(self) -> None:
        from layoutabi.optimizer.api import inspect
        from layoutabi.workloads import make_workload

        module, inputs = make_workload(
            "diffusion_linear_attention", resolution=8, batch=2
        )
        payload = inspect(module, inputs)
        self.assertTrue(payload["matches"])


if __name__ == "__main__":
    unittest.main()
