"""Import-time tests that must work even when optimizer internals need PyTorch."""

from __future__ import annotations

import unittest

import layoutabi


class OptimizerImportTest(unittest.TestCase):
    def test_package_exports_optimize_and_inspect(self) -> None:
        self.assertTrue(callable(layoutabi.optimize))
        self.assertTrue(callable(layoutabi.inspect))
        self.assertTrue(layoutabi.__version__.startswith("0.8."))

    def test_cli_help_expands_without_percent_format_errors(self) -> None:
        from layoutabi.cli import build_parser

        text = build_parser().format_help()
        self.assertIn("evaluate-planner", text)
        self.assertIn("supported", text)


if __name__ == "__main__":
    unittest.main()
