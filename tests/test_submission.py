"""CPU-only tests for community submission preparation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bundleutil import make_bundle, sha256
from layoutabi.cli import main
from layoutabi.submission import prepare_submission, scan_privacy
from layoutabi.validation import validate_result


class SubmissionTest(unittest.TestCase):
    def test_prepare_submission_copies_rewrites_checksums_and_detects_privacy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = make_bundle(
                root / "local_gpu",
                extra_environment={
                    "operator": "alice",
                    "notes": "results live in /home/alice/layoutabi",
                },
            )
            results_root = root / "results"
            result = prepare_submission(
                source,
                "rtx4090_torch2.11_cuda12.8_2026-08-30",
                results_root=results_root,
            )
            destination = result.destination
            self.assertTrue((destination / "manifest.json").is_file())
            self.assertEqual(validate_result(destination), [])
            self.assertEqual(
                sha256(destination / "eager_results.json"),
                sha256(source / "eager_results.json"),
            )
            details = [finding.detail for finding in result.privacy_findings]
            self.assertTrue(any("custom metadata keys" in item for item in details))
            self.assertTrue(any("private path" in item for item in details))
            self.assertIsNone(result.replicate_of)

    def test_prepare_submission_marks_replicate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            make_bundle(root / "results" / "community" / "existing")
            source = make_bundle(root / "local_rerun")
            result = prepare_submission(source, "rerun", results_root=root / "results")
            self.assertEqual(result.replicate_of, "community/existing")
            self.assertEqual(result.exact_duplicate_of, "community/existing")

    def test_strict_privacy_refuses_to_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = make_bundle(
                root / "local_gpu",
                extra_environment={"hostname": "lab-secret.internal"},
            )
            with self.assertRaisesRegex(ValueError, "Privacy scan failed"):
                prepare_submission(
                    source,
                    "secret-box",
                    results_root=root / "results",
                    strict_privacy=True,
                )
            self.assertFalse((root / "results" / "community" / "secret-box").exists())

    def test_scan_privacy_on_clean_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            bundle = make_bundle(Path(temporary_directory) / "clean")
            self.assertEqual(scan_privacy(bundle), [])

    def test_cli_rejects_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = make_bundle(root / "local_gpu")
            prepare_submission(source, "once", results_root=root / "results")
            code = main(
                [
                    "prepare-submission",
                    str(source),
                    "--name",
                    "once",
                    "--results-root",
                    str(root / "results"),
                ]
            )
            self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
