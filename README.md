# Layout ABI Lab

Layout ABI Lab is a reproducibility project for a counter-intuitive GPU performance
effect: removing an intermediate tensor copy is not always faster.

In the studied normalize/view-to-GEMM chains, two numerically equivalent layouts can
select different opaque vendor GEMM kernel families. A deliberate materialization may
therefore pay for itself by unlocking a faster downstream kernel:

```text
producer output layout -> vendor kernel family -> discrete latency cliff
```

Across three pinned NVIDIA L40S software stacks, explicitly repairing both K and V
before compiling a public diffusion LinearAttention module reduced 256x256 steady-state
module latency by **13.95% to 15.19% (1.162x to 1.179x)**. The kernel transition and
eager graph boundary reproduced in every stack. This is a scoped result, not a claim
that repair is universally profitable. See the
[software-stack matrix](results/reference_l40s/SOFTWARE_STACK_MATRIX.md).

## Research question

When should a compiler preserve a zero-copy producer layout, and when should it
intentionally materialize that layout to improve a downstream vendor-library call?

The project is designed to collect positive and negative results across GPUs, PyTorch
versions, CUDA software stacks, data types, shapes, and graph contexts.

## What is included

- A dependency-free reconstruction of an audited public diffusion LinearAttention
  module, with source provenance pinned in the code.
- Direct, repair-K, and repair-KV execution policies with identical numerical semantics.
- Order-balanced, multi-seed eager measurements.
- Isolated `torch.compile` measurements for direct and repair-KV policies.
- CUDA kernel-name capture when `torch.profiler` supports it.
- Machine-readable environment and result records.
- A validator for community-submitted result bundles, with JSON Schema contracts and
  forward schema migration.
- A `prepare-submission` command that copies a local bundle, recomputes checksums, and
  reports possible private metadata without uploading.
- An experimental `layoutabi.optimize()` API that captures a public FX graph, matches
  the frozen LinearAttention KTV pattern, and conservatively chooses direct or repair.
- A compiled mechanism audit that records FX/export graphs, Inductor IR, and profiler
  kernel names so compiled kernel-family claims are not inferred from timing.
- A container matrix runner for testing multiple pinned software stacks.

The repository does **not** contain a general TorchInductor pass. Version 0.3 adds an
experimental external optimizer for one frozen LinearAttention pattern. Version 0.4
adds a compiled mechanism audit so kernel-family claims come from graphs and profiler
names, not from timing. See [Pattern contract](docs/PATTERN_CONTRACT.md),
[Compiled audit](docs/COMPILE_AUDIT.md), and [Roadmap](docs/ROADMAP.md).

The generated [result index](RESULTS_INDEX.md) summarizes all checksum-validated
reference and accepted community bundles. Its JSON counterpart is `results/index.json`.

## Requirements

- Linux
- Python 3.8 or newer
- NVIDIA GPU
- A CUDA-enabled PyTorch installation with `torch.compile` for compiled controls
- Enough GPU memory for the selected resolution

Install PyTorch using the method appropriate for the target GPU first. This project
does not declare PyTorch as a package dependency because an automatic `pip` choice can
silently install an incompatible CUDA build.

## Quick start

```bash
git clone https://github.com/KaiKai0714/layout-abi-lab.git
cd layout-abi-lab

# Install a CUDA-enabled PyTorch build first, then install this project.
python -m pip install -e .

layoutabi check
layoutabi reproduce --output results/local_my_gpu
layoutabi validate results/local_my_gpu
```

For a shorter smoke test:

```bash
layoutabi reproduce \
  --output results/local_smoke \
  --resolutions 128 \
  --seeds 1701 \
  --cycles 2 \
  --iterations 3 \
  --skip-compile
```

The default protocol tests resolutions 256 and 128 with three seeds. It may take
several minutes because compiled policies run in isolated processes and caches.

## Interpreting the output

Each run produces:

```text
results/local_my_gpu/
  environment.json
  eager_results.json
  compile_results.json
  SUMMARY.md
  manifest.json
```

A ratio `direct_ms / repair_ms > 1` means repair was faster. Always inspect the full
module result, correctness gate, software stack, and per-seed consistency. An isolated
GEMM or chain win must not be presented as an end-to-end model win.

## Container software-stack matrix

Edit [containers/matrix.json](containers/matrix.json) with image tags that are valid
for the local NVIDIA driver, then run:

```bash
python containers/run_matrix.py \
  --matrix containers/matrix.json \
  --gpu-device 0 \
  --keep-going
```

This compares complete software stacks, not CUDA in isolation. A container image may
change PyTorch, CUDA runtime, cuBLAS/cuBLASLt, Triton, and TorchInductor together.
The matrix runner processes one image at a time. By default, a newly pulled image is
removed immediately after its result bundle passes validation. Source files are mounted
read-only, containers use `--rm`, and compilation caches disappear with the container.
Images that existed before the run are never removed automatically.

## Submit a result

Both wins and losses are useful. Prepare a community bundle without uploading it:

```bash
layoutabi prepare-submission \
  results/local_my_gpu \
  --name rtx4090_torch2.11_cuda12.8_2026-08-30
```

The command copies the bundle to `results/community/`, recomputes checksums, and prints
possible hostname, username, private-path, and extra-metadata findings. Re-runs of the
same graph, device, software stack, and protocol are kept as replicates, not as a new
device. Bundles without compiled controls are accepted and indexed as compiled
unavailable rather than as a loss. Then follow [CONTRIBUTING.md](CONTRIBUTING.md). Do
not edit measured JSON values by hand.

Maintainers regenerate the public index after accepting a bundle with:

```bash
layoutabi aggregate
```

Continuous integration runs `layoutabi aggregate --check` so stale or manually edited
indexes cannot be merged.

## Experimental optimizer

```python
import torch
from layoutabi import optimize
from layoutabi.workload import PublicDiffusionLinearAttention

model = PublicDiffusionLinearAttention().eval().half().cuda()
x = torch.randn(1, 64, 128, 128, device="cuda", dtype=torch.float16)
result = optimize(model, (x,), policy="autotune")
compiled = torch.compile(result.module)
```

`policy` may be `off`, `direct`, `repair_k`, `repair_kv`, or `autotune`. Autotune
measures full-module CUDA-event latency and caches the decision. If the graph is not
supported, guards fail, or a candidate is incorrect, the original module is returned.

```bash
layoutabi inspect-model --resolution 128
layoutabi optimize-model --resolution 128 --policy repair_kv
layoutabi audit-compile --output results/local_compile_audit
```

## Scope and limitations

- The optimizer supports one frozen inference pattern, fixed shapes, and FP16.
- Autotune requires CUDA. Unsupported graphs are a safe no-op, not a guaranteed speedup.
- Current reference experiments focus on inference, fixed shapes, and FP16.
- Profitability is device-, version-, dtype-, shape-, and graph-dependent.
- `N % 8` is an observed FP16 feature in a bounded regime, not a universal law.
- BF16, INT8, FP8, larger batches, and some devices provide important negative cases.
- The public module is a source-equivalent audited reconstruction, not a vendored copy
  of the complete upstream application.
- A module-level speedup is not a full diffusion-model speedup.

See [Scientific claims and boundaries](docs/CLAIMS_AND_LIMITATIONS.md) before citing
the results.

## License and citation

The project is released under the Apache License 2.0. Upstream workload provenance and
licenses remain separately documented. Citation metadata is available in
[CITATION.cff](CITATION.cff).

Maintainer: Sheng-Kai Ku (`ethankai0714@gmail.com`).
