"""Workload catalog and data-driven optimizer case tests."""

from __future__ import annotations

import unittest
from pathlib import Path

from layoutabi.workloads import get_workload, list_workloads, load_catalog, synthetic_cells

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

CASES_DIR = Path(__file__).resolve().parents[1] / "layoutabi" / "workloads" / "cases"


class WorkloadCatalogTest(unittest.TestCase):
    def test_every_json_case_is_complete_and_has_a_builder(self) -> None:
        catalog = load_catalog()
        self.assertGreaterEqual(len(catalog), 1)
        json_stems = {path.stem for path in CASES_DIR.glob("*.json")}
        py_stems = {path.stem for path in CASES_DIR.glob("*.py")} - {"__init__"}
        self.assertEqual(json_stems, set(catalog))
        self.assertTrue(json_stems.issubset(py_stems))
        for spec in catalog.values():
            self.assertEqual(spec["id"], get_workload(spec["id"])["id"])
            self.assertIn(spec["expected_optimizer"], {"match", "noop"})
            self.assertIn(
                spec["role"],
                {"positive_reference", "public", "negative", "experimental"},
            )
            self.assertTrue(spec["graph_fingerprint"])

    def test_catalog_includes_both_match_and_noop_roles(self) -> None:
        expectations = {item["expected_optimizer"] for item in list_workloads()}
        self.assertIn("match", expectations)
        self.assertIn("noop", expectations)

    def test_public_match_cases_are_not_the_same_repository_as_the_reference(self) -> None:
        workloads = list_workloads()
        references = [item for item in workloads if item["role"] == "positive_reference"]
        public_matches = [
            item
            for item in workloads
            if item["role"] == "public" and item["expected_optimizer"] == "match"
        ]
        if not references or not public_matches:
            self.skipTest("no extra public match case registered yet")
        reference_repo = references[0].get("repository")
        for item in public_matches:
            self.assertNotEqual(item.get("repository"), reference_repo)

    def test_synthetic_cells_cover_nearby_shapes_without_replacing_public_graphs(self) -> None:
        cells = synthetic_cells()
        resolutions = {cell["resolution"] for cell in cells}
        self.assertTrue({64, 128, 160, 256, 384}.issubset(resolutions))
        self.assertTrue(any(cell["role"] == "mainline" for cell in cells))
        self.assertTrue(any(cell["role"] == "boundary_batch" for cell in cells))
        self.assertTrue(any(cell["role"] == "boundary_unsupported_dtype" for cell in cells))
        self.assertNotIn("synthetic", load_catalog())


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class WorkloadOptimizerTest(unittest.TestCase):
    def test_optimizer_follows_expected_behavior_for_every_case(self) -> None:
        from layoutabi.optimizer.api import inspect, optimize
        from layoutabi.workloads import make_workload

        for spec in list_workloads():
            with self.subTest(spec["id"]):
                resolution = int(spec.get("smoke_resolution") or 8)
                module, inputs = make_workload(spec["id"], resolution=resolution)
                result = optimize(module, inputs, policy="repair_kv")
                if spec["expected_optimizer"] == "noop":
                    self.assertIs(result.module, module)
                    self.assertEqual(result.decision, "noop")
                    continue
                payload = inspect(module, inputs)
                self.assertTrue(payload["matches"], payload.get("reason"))
                self.assertNotIn(type(module).__name__, str(payload["matches"]))
                self.assertEqual(result.decision, "repair_kv")
                with torch.no_grad():
                    reference = module(*inputs)
                    rewritten = result.module(*inputs)
                delta = (rewritten.float() - reference.float()).abs().max().item()
                self.assertLessEqual(delta, 0.08)

    def test_optional_reference_outputs_when_a_case_provides_them(self) -> None:
        import importlib

        from layoutabi.workloads import make_workload

        for spec in list_workloads():
            module_name = f"layoutabi.workloads.cases.{spec['id']}"
            case_module = importlib.import_module(module_name)
            check = getattr(case_module, "reference_outputs", None)
            if check is None:
                continue
            with self.subTest(spec["id"]):
                resolution = int(spec.get("smoke_resolution") or 8)
                module, inputs = make_workload(spec["id"], resolution=resolution)
                with torch.no_grad():
                    reference = module(*inputs)
                    other = check(module, inputs)
                delta = (reference.float() - other.float()).abs().max().item()
                self.assertLessEqual(delta, 0.08)

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
