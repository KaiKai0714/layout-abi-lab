"""CPU-only tests for compiled mechanism-audit evidence parsing."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from layoutabi.audit import interpret_evidence, kernel_family, validate_audit
from layoutabi.schema import AUDIT_SCHEMA, normalize_document


DIRECT_EAGER_KERNELS = [
    "void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_tn_align2>(Params)",
    "void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x64_32x6_nn_align8>(Params)",
]
REPAIR_KERNELS = [
    "void at::native::(anonymous namespace)::CatArrayBatchedCopy_alignedK_contig<...>()",
    "void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x64_32x6_nn_align8>(Params)",
    "void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x64_32x6_nn_align8>(Params)",
]


class AuditEvidenceTest(unittest.TestCase):
    def test_kernel_family_uses_align_token_not_latency(self) -> None:
        self.assertEqual(kernel_family(DIRECT_EAGER_KERNELS[0]), "align2")
        self.assertEqual(kernel_family(DIRECT_EAGER_KERNELS[1]), "align8")
        self.assertEqual(kernel_family(REPAIR_KERNELS[1]), "align8")

    def test_direct_sequence_records_first_and_second_gemm_families(self) -> None:
        evidence = interpret_evidence(
            ordered_cuda_names=DIRECT_EAGER_KERNELS,
            pre_fusion="buf0 = clone(k_softmax)\nextern_kernels.mm(buf0, buf1)",
            post_fusion="extern_kernels.mm(buf0, buf1)",
            fx_graph="softmax_1 = k.softmax(dim = -1)\nmatmul = softmax_1 @ transpose",
            output_code="x = empty_strided((1, 4, 32, 32), (4096, 1024, 32, 1), device='cuda')",
        )
        self.assertEqual(evidence["first_gemm_family"], "align2")
        self.assertEqual(evidence["second_gemm_family"], "align8")
        self.assertFalse(evidence["copy_in_profiler"])
        self.assertTrue(evidence["copy_in_pre_fusion"])
        self.assertFalse(evidence["copy_in_post_fusion"])
        self.assertTrue(evidence["copy_fused_or_eliminated"])
        self.assertIn("profiler", evidence["evidence_source"])
        self.assertTrue(evidence["generated_strides"])

    def test_repair_sequence_keeps_copy_and_align8_from_profiler(self) -> None:
        evidence = interpret_evidence(
            ordered_cuda_names=REPAIR_KERNELS,
            pre_fusion="buf0 = clone(buf_k)\nextern_kernels.mm(buf0, buf1)",
            post_fusion="buf0 = clone(buf_k)\nextern_kernels.mm(buf0, buf1)",
            fx_graph="contiguous = softmax_1.contiguous()\nmatmul = contiguous @ transpose",
            output_code=None,
        )
        self.assertTrue(evidence["copy_in_profiler"])
        self.assertTrue(evidence["copy_in_fx"])
        self.assertTrue(evidence["copy_in_post_fusion"])
        self.assertFalse(evidence["copy_fused_or_eliminated"])
        self.assertTrue(evidence["materialization_retained_in_compiled"])
        self.assertEqual(evidence["first_gemm_family"], "align8")
        self.assertEqual(evidence["second_gemm_family"], "align8")

    def test_missing_ir_does_not_invent_fusion(self) -> None:
        evidence = interpret_evidence(
            ordered_cuda_names=DIRECT_EAGER_KERNELS,
            pre_fusion=None,
            post_fusion=None,
            fx_graph=None,
            output_code=None,
        )
        self.assertIsNone(evidence["copy_fused_or_eliminated"])
        self.assertEqual(evidence["evidence_source"], ["profiler"])

    def test_audit_document_validates_and_detects_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summary = root / "SUMMARY.md"
            summary.write_text("# audit\n", encoding="utf-8")
            payload = {
                "schema": AUDIT_SCHEMA,
                "schema_version": 1,
                "points": [
                    {
                        "resolution": 128,
                        "policies": {
                            "direct": {
                                "available": True,
                                "evidence": interpret_evidence(
                                    ordered_cuda_names=DIRECT_EAGER_KERNELS,
                                    pre_fusion=None,
                                    post_fusion=None,
                                    fx_graph=None,
                                    output_code=None,
                                ),
                            }
                        },
                    }
                ],
                "files": {"SUMMARY.md": hashlib.sha256(summary.read_bytes()).hexdigest()},
            }
            migrated, problems = normalize_document(payload, AUDIT_SCHEMA)
            self.assertEqual(problems, [])
            self.assertIsNotNone(migrated)
            (root / "compile_audit.json").write_text(
                json.dumps(payload) + "\n", encoding="utf-8"
            )
            self.assertEqual(validate_audit(root), [])
            summary.write_text("tampered\n", encoding="utf-8")
            self.assertTrue(any("Checksum mismatch" in item for item in validate_audit(root)))


if __name__ == "__main__":
    unittest.main()
