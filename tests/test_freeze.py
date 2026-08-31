"""Release-candidate freeze and repository scan tests. No PyTorch required."""

from __future__ import annotations

import unittest
from pathlib import Path

import layoutabi
from layoutabi.freeze import freeze_problems, live_freeze, load_frozen, rc_status
from layoutabi.optimizer.pattern import CANDIDATE_IMPL_HASH
from layoutabi.scan import _scan_text, scan_release


class FreezeTest(unittest.TestCase):
    def test_live_snapshot_matches_frozen_file(self) -> None:
        self.assertEqual(freeze_problems(), [])
        frozen = load_frozen()
        live = live_freeze()
        self.assertEqual(frozen["pattern_id"], live["pattern_id"])
        self.assertEqual(frozen["candidate_impl"], CANDIDATE_IMPL_HASH)
        self.assertFalse(live["allow_new_patterns"])

    def test_public_api_has_not_grown(self) -> None:
        frozen = load_frozen()
        extras = [
            name
            for name in layoutabi.__all__
            if name not in {"__version__", *frozen["public_api"], *frozen["exceptions"]}
        ]
        self.assertEqual(extras, [])

    def test_candidate_impl_is_still_bhnd_transpose_contiguous(self) -> None:
        path = Path(layoutabi.__file__).resolve().parent / "optimizer" / "rewrite.py"
        text = path.read_text(encoding="utf-8")
        self.assertIn('call_method("transpose"', text)
        self.assertIn('call_method("contiguous"', text)
        self.assertEqual(CANDIDATE_IMPL_HASH, "bhnd_transpose_contiguous_transpose_v1")

    def test_rc_status_reports_open_architecture_gates(self) -> None:
        payload = rc_status()
        self.assertEqual(payload["release_status"], "candidate")
        self.assertTrue(payload["freeze_ok"])
        self.assertIn("orin", payload["v1_open_gates"])
        self.assertIn("third_architecture", payload["v1_open_gates"])

    def test_scan_release_is_clean(self) -> None:
        findings = scan_release()
        self.assertEqual(
            findings,
            [],
            msg="\n".join(f"{item.location}: {item.detail}" for item in findings),
        )

    def test_scan_detects_home_directory_outside_tests(self) -> None:
        findings = _scan_text("docs/leaked.md", "wrote /home/bob/layoutabi")
        self.assertTrue(any("private path" in item.detail for item in findings))


if __name__ == "__main__":
    unittest.main()
