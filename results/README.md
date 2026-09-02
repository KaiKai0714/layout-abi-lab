# Results

Three kinds of directory. Only the first two are committed.

| Location | What it is | Git |
|---|---|---|
| `reference_l40s/` | Frozen L40S evidence that the public claims cite | tracked |
| `community/` | Accepted third-party bundles (wins and losses) | tracked |
| `local_<name>/` | Raw output of a run on your machine | gitignored |

Do not commit `results/local_*`. Smoke runs belong in `/tmp` or a `local_*` folder
you delete afterwards.

## Bundle name

One full protocol run (one GPU × one PyTorch/CUDA stack × one day) is one
directory. Use the same `<name>` locally and in a pull request:

```text
<gpu>_torch<pytorch>_cuda<cuda>_<yyyy-mm-dd>
```

Examples:

```text
results/local_l40s_torch2.11_cuda12.8_2026-08-31
results/local_orin_torch2.7_cuda12.6_2026-09-01
results/community/rtx4090_torch2.11_cuda12.8_2026-08-30
```

Rules:

- `<gpu>` is a short lowercase token: `l40s`, `orin`, `rtx4090`, `a100`.
- `<pytorch>` / `<cuda>` match the stack, not the Layout ABI version
  (`torch2.11_cuda12.8`, not `v0.9.0`).
- A later re-run of the same GPU and stack gets a new date. The index treats it
  as a **replicate**, not a new device.
- Extra software stacks are extra bundles, not extra devices.

```bash
layoutabi reproduce --output results/local_orin_torch2.7_cuda12.6_2026-09-01
layoutabi validate results/local_orin_torch2.7_cuda12.6_2026-09-01
layoutabi prepare-submission \
  results/local_orin_torch2.7_cuda12.6_2026-09-01 \
  --name orin_torch2.7_cuda12.6_2026-09-01
```

`SUMMARY.md` is the human page. JSON files are the provenance.

## What is already published

- `reference_l40s/software_stack_matrix/torch2.10_cuda12.8` (and the 2.11
  CUDA 12.6 / 12.8 siblings): frozen L40S matrix. These folders omit the GPU
  token because the whole tree is L40S-only.
- `reference_l40s/v0_1_bundle/`: same GPU and stack as
  `software_stack_matrix/torch2.11_cuda12.8`; indexed as a replicate.
- `community/orin_torch2.7_cuda12.8_2026-08-31/`: eager FP16 evidence from a
  second GPU architecture. Repair is slower at resolution 128 and compiled
  results are unavailable.
- `community/orin_pointer_alignment/torch2.7_cuda12.8/`: the matching Orin
  100-cell pointer audit. It is a standalone mechanism control, not an indexed
  full-workload bundle.

The `reference_l40s/three_level_sweep/torch2.11_cuda12.8/` bundle closes the
three-level FP16 mechanism gate. It spans `N%8==0`, even non-multiples, and odd
N with isolated KTV profiler names. The older 128/256 reference rows remain
valid profitability records but contain only the intermediate residue class.

The generated index is `RESULTS_INDEX.md` and `results/index.json`.

A container matrix on one GPU writes one dated bundle per stack:

```bash
python containers/run_matrix.py \
  --matrix containers/matrix.json \
  --gpu-tag l40s \
  --gpu-device 0
```

That produces `results/local_l40s_torch2.11_cuda12.8_<today>/` (and the other
enabled stacks). Compiled-audit directories are also `local_*` and stay
gitignored; they are not community bundles.

For the L40S three-level boundary sweep, use the dedicated single-stack matrix so
the frozen three-stack reference protocol remains unchanged:

```bash
python containers/run_matrix.py \
  --matrix containers/matrix_three_level_l40s.json \
  --output-root results/local_nmod8_sweep \
  --gpu-tag l40s \
  --gpu-device 0
```

It measures `126/127/128` and `254/255/256`. For this workload,
`consumer_n = resolution^2 + 4`: 126/254 produce the fastest-class residue 0,
128/256 the intermediate even residue 4, and 127/255 the odd residue 5. Every
resolution records isolated KTV profiler names as well as chain/module latency.

The standalone pointer mechanism audit lives outside the ordinary result index:

```bash
layoutabi audit-pointer \
  --output results/local_l40s_pointer_audit_2026-09-01 \
  --ns 65536,65538,65540,65543 \
  --offsets 0,2,4,8,16 \
  --cycles 12 \
  --iterations 20
layoutabi validate-pointer-audit results/local_l40s_pointer_audit_2026-09-01
```

The published L40S audit is in
`reference_l40s/pointer_alignment/torch2.11_cuda12.8/`. It contains a complete
K-pointer × V-pointer grid and is not counted as another workload or device.
The matching Orin audit is in
`community/orin_pointer_alignment/torch2.7_cuda12.8/`; both pass the dedicated
validator and preserve all 100 cells.

The matching six-shape compiled mechanism evidence is in
`reference_l40s/compile_audit/torch2.11_cuda12.8/`. Compiled audits use
`validate-audit`, not the ordinary bundle validator, and are intentionally not
counted as another device or workload in `RESULTS_INDEX.md`.
