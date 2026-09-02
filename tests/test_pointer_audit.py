"""CPU-only tests for the controlled pointer-alignment audit."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from layoutabi.pointer_audit import (
    _event_kernel_names,
    pointer_family,
    validate_pointer_audit,
)
from layoutabi.schema import ENVIRONMENT_SCHEMA, POINTER_AUDIT_SCHEMA


class PointerAuditTest(unittest.TestCase):
    def test_family_comes_from_exact_alignment_token(self) -> None:
        self.assertEqual(pointer_family(["cutlass_kernel_align1"]), "align1")
        self.assertEqual(pointer_family(["cutlass_kernel_align2"]), "align2")
        self.assertEqual(pointer_family(["cutlass_kernel_align8"]), "align8")
        self.assertEqual(pointer_family(["cutlass_kernel_align16"]), "align16")
        self.assertEqual(pointer_family(["ampere_fp16_s16816gemm_ldg8"]), "ldg8")

    def test_marker_tree_collects_nested_kernel_names(self) -> None:
        class Kernel:
            name = "cutlass_kernel_align2"

        class Event:
            def __init__(self, kernels=(), children=()) -> None:
                self.kernels = kernels
                self.cpu_children = children

        marker = Event(children=[Event(kernels=[Kernel()])])
        self.assertEqual(_event_kernel_names(marker), ["cutlass_kernel_align2"])

    def test_validator_requires_full_grid_and_detects_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            environment = {
                "schema": ENVIRONMENT_SCHEMA,
                "schema_version": 1,
            }
            (root / "environment.json").write_text(
                json.dumps(environment) + "\n", encoding="utf-8"
            )
            (root / "SUMMARY.md").write_text("# pointer audit\n", encoding="utf-8")
            rows = []
            for k_offset in (0, 2):
                for v_offset in (0, 2):
                    rows.append(
                        {
                            "k_offset_bytes": k_offset,
                            "v_offset_bytes": v_offset,
                            "k_pointer": {"actual_mod64": k_offset},
                            "v_pointer": {"actual_mod64": v_offset},
                            "correctness": {"pass": True},
                            "timing": {"median_ms": 1.0},
                            "profiler": {"available": True, "family": "align8"},
                        }
                    )
            payload = {
                "schema": POINTER_AUDIT_SCHEMA,
                "schema_version": 1,
                "device": {},
                "software": {},
                "protocol": {
                    "dtype": "fp16",
                    "ns": [16],
                    "offsets_mod64_bytes": [0, 2],
                    "cycles": 1,
                    "iterations_per_measurement": 1,
                },
                "points": [{"n": 16, "n_mod_8": 0, "n_class": "fastest", "rows": rows}],
                "files": {
                    "environment.json": hashlib.sha256(
                        (root / "environment.json").read_bytes()
                    ).hexdigest(),
                    "SUMMARY.md": hashlib.sha256((root / "SUMMARY.md").read_bytes()).hexdigest(),
                },
            }
            (root / "pointer_audit.json").write_text(
                json.dumps(payload) + "\n", encoding="utf-8"
            )
            self.assertEqual(validate_pointer_audit(root), [])
            payload["points"][0]["rows"].pop()
            (root / "pointer_audit.json").write_text(
                json.dumps(payload) + "\n", encoding="utf-8"
            )
            self.assertTrue(
                any("incomplete" in item for item in validate_pointer_audit(root))
            )


if __name__ == "__main__":
    unittest.main()
