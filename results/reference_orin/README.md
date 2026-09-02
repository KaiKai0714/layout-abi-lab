# Author-run Orin reference

This tree contains measurements run by the project author on Jetson Orin. Its
`reference` classification records provenance and protocol ownership; it does
not mean the device has the same coverage as L40S.

Current evidence is deliberately split by question:

- `orin_torch2.7_cuda12.8_2026-08-31/` contains one eager full-module row at
  resolution 128. Direct execution wins; compiled execution and other module
  resolutions were not measured.
- `pointer_alignment/torch2.7_cuda12.8/` contains the complete 100-cell
  mechanism audit. It verifies how N and both operand pointers select the GEMM
  family, but does not include layout-repair cost or establish module-level
  profitability.

Do not summarize this tree as broad Orin repair validation. The supported
cross-architecture claim is limited to the controlled pointer-to-family
mechanism.
