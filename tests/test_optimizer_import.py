"""Import-time tests that must work even when optimizer internals need PyTorch."""

from __future__ import annotations

import unittest

import layoutabi


class OptimizerImportTest(unittest.TestCase):
    def test_package_exports_optimize_and_inspect(self) -> None:
        self.assertTrue(callable(layoutabi.optimize))
        self.assertTrue(callable(layoutabi.inspect))
        self.assertTrue(layoutabi.__version__.startswith("0.4."))


if __name__ == "__main__":
    unittest.main()
