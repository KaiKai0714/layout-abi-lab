"""CPU-only tests for deterministic community result aggregation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bundleutil import make_bundle
from layoutabi.aggregation import build_index, render_markdown, write_index


class AggregationTest(unittest.TestCase):
    def test_community_bundle_without_compile_is_unavailable_not_loss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            make_bundle(root / "results" / "community" / "test_gpu_stack_2026-08-30")
            results_root = root / "results"
            output_json = root / "results" / "index.json"
            output_markdown = root / "RESULTS_INDEX.md"

            index = build_index(results_root)
            summary = index["summary"]
            self.assertEqual(summary["bundles"], 1)
            self.assertEqual(summary["community_bundles"], 1)
            self.assertEqual(summary["replicate_bundles"], 0)
            self.assertEqual(summary["eager_module_repair_wins"], 1)
            self.assertEqual(summary["compiled_module_unavailable"], 1)
            self.assertEqual(summary["compiled_module_direct_wins"], 0)
            self.assertEqual(summary["compiled_module_repair_wins"], 0)
            row = index["bundles"][0]["rows"][0]["compiled_module"]
            self.assertEqual(row["outcome"], "unavailable")
            self.assertIsNone(row["ratio"])
            markdown = render_markdown(index, results_root, output_markdown)
            self.assertIn("Test GPU", markdown)
            self.assertIn("2.000x", markdown)
            self.assertIn("## Community measurements", markdown)

            write_index(
                results_root=results_root,
                output_json=output_json,
                output_markdown=output_markdown,
            )
            first_json = output_json.read_bytes()
            first_markdown = output_markdown.read_bytes()
            write_index(
                results_root=results_root,
                output_json=output_json,
                output_markdown=output_markdown,
                check=True,
            )
            self.assertEqual(first_json, output_json.read_bytes())
            self.assertEqual(first_markdown, output_markdown.read_bytes())
            write_index(
                results_root=results_root,
                output_json=output_json,
                output_markdown=output_markdown,
            )
            self.assertEqual(first_json, output_json.read_bytes())
            self.assertEqual(first_markdown, output_markdown.read_bytes())

    def test_same_device_stack_protocol_is_replicate_not_new_device(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            make_bundle(root / "results" / "community" / "gpu_a")
            make_bundle(root / "results" / "community" / "gpu_b")
            index = build_index(root / "results")
            roles = {record["id"]: record["role"] for record in index["bundles"]}
            self.assertEqual(roles["community/gpu_a"], "community")
            self.assertEqual(roles["community/gpu_b"], "replicate")
            self.assertEqual(index["bundles"][1]["replicate_of"], "community/gpu_a")
            self.assertEqual(index["summary"]["community_bundles"], 1)
            self.assertEqual(index["summary"]["replicate_bundles"], 1)
            self.assertEqual(index["summary"]["devices"], 1)
            self.assertEqual(index["summary"]["software_stacks"], 1)
            self.assertEqual(index["summary"]["eager_module_repair_wins"], 1)

    def test_different_stack_is_not_a_replicate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            make_bundle(root / "results" / "community" / "stack_a", torch_version="2.10")
            make_bundle(root / "results" / "community" / "stack_b", torch_version="2.11")
            index = build_index(root / "results")
            self.assertEqual(index["summary"]["community_bundles"], 2)
            self.assertEqual(index["summary"]["replicate_bundles"], 0)
            self.assertEqual(index["summary"]["software_stacks"], 2)
            self.assertEqual(index["summary"]["devices"], 1)

    def test_reference_index_marks_v0_1_as_replicate_and_is_byte_identical(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        results_root = repo / "results"
        first = json.dumps(build_index(results_root), indent=2, sort_keys=True)
        second = json.dumps(build_index(results_root), indent=2, sort_keys=True)
        self.assertEqual(first, second)
        index = json.loads(first)
        by_id = {record["id"]: record for record in index["bundles"]}
        self.assertEqual(by_id["reference_l40s/v0_1_bundle"]["role"], "replicate")
        self.assertEqual(
            by_id["reference_l40s/v0_1_bundle"]["replicate_of"],
            "reference_l40s/software_stack_matrix/torch2.11_cuda12.8",
        )
        self.assertEqual(index["summary"]["reference_bundles"], 3)
        self.assertEqual(index["summary"]["replicate_bundles"], 1)
        self.assertEqual(index["summary"]["devices"], 1)
        self.assertEqual(index["summary"]["software_stacks"], 3)
        self.assertEqual(index["summary"]["compiled_module_repair_wins"], 6)
        self.assertIn("reference", index["filters"]["roles"])
        self.assertIn("replicate", index["filters"]["roles"])
        self.assertIn("fp16", index["filters"]["dtypes"])
        self.assertEqual(index["filters"]["resolutions"], [128, 256])


if __name__ == "__main__":
    unittest.main()
