"""CPU-only tests for the public result-bundle contract."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from layoutabi.validation import validate_result


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ValidationTest(unittest.TestCase):
    def test_minimal_valid_bundle_and_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_json(root / "environment.json", {"schema": "layoutabi_environment_v1"})
            write_json(
                root / "eager_results.json",
                {
                    "schema": "layoutabi_eager_v1",
                    "points": [
                        {
                            "resolution": 8,
                            "seeds": [
                                {
                                    "seed": 1,
                                    "correctness": {
                                        "chain": {"direct": {"pass": True}},
                                        "module": {"direct": {"pass": True}},
                                    },
                                }
                            ],
                        }
                    ],
                },
            )
            (root / "SUMMARY.md").write_text("# Test\n", encoding="utf-8")
            measured = ("environment.json", "eager_results.json", "SUMMARY.md")
            write_json(
                root / "manifest.json",
                {
                    "schema": "layoutabi_manifest_v1",
                    "files": {name: sha256(root / name) for name in measured},
                },
            )

            self.assertEqual(validate_result(root), [])
            (root / "SUMMARY.md").write_text("tampered\n", encoding="utf-8")
            self.assertIn("Checksum mismatch: SUMMARY.md", validate_result(root))


if __name__ == "__main__":
    unittest.main()

