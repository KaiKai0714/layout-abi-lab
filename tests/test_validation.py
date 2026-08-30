"""CPU-only tests for the public result-bundle contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bundleutil import make_bundle, write_json
from layoutabi.schema import migrate_bundle_documents
from layoutabi.validation import discover_result_bundles, validate_result


class ValidationTest(unittest.TestCase):
    def test_minimal_valid_bundle_and_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            make_bundle(root)

            self.assertEqual(validate_result(root), [])
            self.assertEqual(discover_result_bundles(root), [root])
            (root / "SUMMARY.md").write_text("tampered\n", encoding="utf-8")
            self.assertIn("Checksum mismatch: SUMMARY.md", validate_result(root))

    def test_legacy_bundle_without_schema_version_is_readable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            make_bundle(root)
            self.assertEqual(validate_result(root), [])

    def test_malformed_json_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            make_bundle(root)
            (root / "environment.json").write_text("{", encoding="utf-8")
            problems = validate_result(root)
            self.assertTrue(any("Could not parse environment.json" in item for item in problems))

    def test_schema_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            make_bundle(root)
            payload = json.loads((root / "environment.json").read_text(encoding="utf-8"))
            payload["schema"] = "layoutabi_environment_v9"
            write_json(root / "environment.json", payload)
            problems = validate_result(root)
            self.assertIn("Unsupported environment schema", problems)

    def test_newer_schema_version_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            make_bundle(root)
            payload = json.loads((root / "environment.json").read_text(encoding="utf-8"))
            payload["schema_version"] = 99
            write_json(root / "environment.json", payload)
            problems = validate_result(root)
            self.assertTrue(any("schema_version 99" in item for item in problems))

    def test_missing_directory_discovery(self) -> None:
        self.assertEqual(discover_result_bundles(Path("missing-layoutabi-bundles")), [])

    def test_reference_v0_bundles_remain_valid(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        root = repo / "results" / "reference_l40s"
        bundles = discover_result_bundles(root)
        self.assertGreaterEqual(len(bundles), 4)
        for bundle in bundles:
            self.assertEqual(validate_result(bundle, strict=True), [])

    def test_migrate_schema_dry_run_and_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            make_bundle(root)
            migrated = migrate_bundle_documents(root, write=False)
            self.assertEqual(migrated["environment.json"]["schema_version"], 1)
            on_disk = json.loads((root / "environment.json").read_text(encoding="utf-8"))
            self.assertNotIn("schema_version", on_disk)
            migrate_bundle_documents(root, write=True)
            written = json.loads((root / "environment.json").read_text(encoding="utf-8"))
            self.assertEqual(written["schema_version"], 1)
            self.assertEqual(validate_result(root), [])


if __name__ == "__main__":
    unittest.main()
