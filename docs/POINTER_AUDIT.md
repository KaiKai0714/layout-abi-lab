# Operand pointer-alignment audit

`N % 8` and operand-pointer alignment are separate Layout-ABI variables. The
ordinary reproduction command uses allocator-aligned tensors, so it cannot establish
the pointer effect by itself.

The controlled audit fixes FP16 GEMM shapes and strides while assigning exact
`data_ptr() % 64` residues to both operands. Its default design crosses:

- `N = 65536, 65538, 65540, 65543`, spanning divisible-by-8, even non-multiple,
  and odd length classes;
- K-pointer residue × V-pointer residue over `{0, 2, 4, 8, 16}` bytes;
- 12 rotating, alternating-direction measurement cycles;
- CUDA profiler kernel names for every cell.

This is a full 4 × 5 × 5 factorial audit (100 cells). Tensor values, logical
shape, strides, dtype, and GEMM dimensions stay fixed within each N. Allocation
and setup copies are outside the timed region. The validator requires the full
grid, exact observed pointer residues, correctness, profiler availability, kernel
family identification, and checksums.

Run:

```bash
layoutabi audit-pointer \
  --output results/local_pointer_audit \
  --ns 65536,65538,65540,65543 \
  --offsets 0,2,4,8,16 \
  --cycles 12 \
  --iterations 20

layoutabi validate-pointer-audit results/local_pointer_audit
```

The summary matrices report `family / median ms`; rows vary K-pointer residue
and columns vary V-pointer residue. Family names are evidence from CUDA profiler
events, not labels inferred from timing.

## Interpretation boundary

The audit establishes how operand pointer residues and N jointly affect an
isolated FP16 consumer GEMM on the recorded device and software stack. It does
not prove that the same byte quantum applies to every dtype, GPU, CUDA version,
or GEMM shape, and it does not measure full-module repair profitability.
