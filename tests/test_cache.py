"""Decision-cache locking, invalidation, and shape-bucket tests."""

from __future__ import annotations

import json
import multiprocessing
import tempfile
import unittest
from pathlib import Path

from layoutabi.optimizer.cache import (
    CACHE_PROTOCOL,
    CACHE_SCHEMA,
    clear_cache,
    load_entry,
    make_cache_key,
    store_entry,
)
from layoutabi.optimizer.shapes import bucket_dim, bucket_shape, pointer_class


def _store_worker(cache_dir: str, key: str, decision: str) -> None:
    store_entry(Path(cache_dir), key, {"decision": decision})


class ShapeBucketTest(unittest.TestCase):
    def test_bucket_maps_inside_range_and_rejects_unseen(self) -> None:
        self.assertEqual(bucket_dim(100), 128)
        self.assertEqual(bucket_dim(256), 256)
        self.assertIsNone(bucket_dim(2048))
        self.assertIsNone(bucket_dim(0))
        self.assertEqual(bucket_shape((1, 64, 100, 100)), [32, 64, 128, 128])
        self.assertIsNone(bucket_shape((1, 64, 800, 800)))

    def test_pointer_class(self) -> None:
        self.assertEqual(pointer_class(256), "align256")
        self.assertEqual(pointer_class(8), "align8")
        self.assertEqual(pointer_class(4), "unaligned")


class DecisionCacheTest(unittest.TestCase):
    def test_roundtrip_and_human_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory)
            store_entry(cache_dir, "abc", {"decision": "repair_kv"})
            entry = load_entry(cache_dir, "abc")
            self.assertIsNotNone(entry)
            assert entry is not None
            self.assertEqual(entry["decision"], "repair_kv")
            diagnostics = cache_dir / "DIAGNOSTICS.md"
            self.assertTrue(diagnostics.is_file())
            self.assertIn("repair_kv", diagnostics.read_text(encoding="utf-8"))

    def test_corrupt_cache_is_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory)
            path = cache_dir / "decisions.json"
            cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text("{not-json", encoding="utf-8")
            self.assertIsNone(load_entry(cache_dir, "abc"))
            payload = json.loads((cache_dir / "decisions.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], CACHE_SCHEMA)
            self.assertEqual(payload["schema_version"], CACHE_PROTOCOL)
            self.assertTrue(payload.get("recovered"))
            self.assertTrue((cache_dir / "decisions.json.corrupt").is_file())

    def test_old_schema_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory)
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "decisions.json").write_text(
                json.dumps(
                    {
                        "schema": "layoutabi_optimizer_cache_v1",
                        "schema_version": 1,
                        "entries": {"abc": {"decision": "repair_kv"}},
                    }
                ),
                encoding="utf-8",
            )
            self.assertIsNone(load_entry(cache_dir, "abc"))

    def test_stack_fields_change_the_cache_key(self) -> None:
        base = {"graph_fingerprint": "g", "torch": "2.11.0", "cuda_build": "12.8"}
        other = dict(base)
        other["torch"] = "2.10.0"
        self.assertNotEqual(make_cache_key(base), make_cache_key(other))

    def test_multiprocess_writes_do_not_drop_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory)
            workers = [
                multiprocessing.Process(
                    target=_store_worker, args=(str(cache_dir), f"k{index}", "direct")
                )
                for index in range(4)
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=30)
                self.assertEqual(worker.exitcode, 0)
            for index in range(4):
                entry = load_entry(cache_dir, f"k{index}")
                self.assertIsNotNone(entry)

    def test_clear_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory)
            store_entry(cache_dir, "abc", {"decision": "direct"})
            clear_cache(cache_dir)
            self.assertIsNone(load_entry(cache_dir, "abc"))


try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class UnseenShapeOptimizeTest(unittest.TestCase):
    def test_out_of_range_bucket_does_not_repair(self) -> None:
        from layoutabi.optimizer.api import optimize

        class TinyKTV(nn.Module):
            def forward(self, q, k, v):  # type: ignore[no-untyped-def]
                q = q.softmax(dim=-2)
                k = k.softmax(dim=-1)
                context = k @ v.transpose(-2, -1)
                return context.transpose(-2, -1) @ q

        model = TinyKTV().eval().half()
        q = torch.randn(1, 4, 32, 800, dtype=torch.float16)
        k = torch.randn(1, 4, 32, 800, dtype=torch.float16)
        v = torch.randn(1, 4, 32, 800, dtype=torch.float16)
        result = optimize(
            model,
            (q, k, v),
            policy="repair_kv",
            shape_mode="bucket",
            unseen_shape="direct",
        )
        self.assertEqual(result.decision, "direct")
        self.assertEqual(result.diagnostics["reason"], "unseen_shape")

    def test_latency_critical_skips_sync_autotune(self) -> None:
        from layoutabi.optimizer.api import optimize

        class TinyKTV(nn.Module):
            def forward(self, q, k, v):  # type: ignore[no-untyped-def]
                q = q.softmax(dim=-2)
                k = k.softmax(dim=-1)
                context = k @ v.transpose(-2, -1)
                return context.transpose(-2, -1) @ q

        model = TinyKTV().eval().half()
        q = torch.randn(1, 4, 32, 16, dtype=torch.float16)
        k = torch.randn(1, 4, 32, 16, dtype=torch.float16)
        v = torch.randn(1, 4, 32, 16, dtype=torch.float16)
        result = optimize(
            model,
            (q, k, v),
            policy="autotune",
            allow_sync_autotune=False,
            unseen_shape="direct",
        )
        self.assertEqual(result.decision, "direct")
        self.assertEqual(result.diagnostics["reason"], "sync_autotune_disabled")
        self.assertIn("capture", result.diagnostics.get("timings_ms", {}))


if __name__ == "__main__":
    unittest.main()
