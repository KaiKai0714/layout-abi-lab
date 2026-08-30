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
- A validator for community-submitted result bundles.
- A container matrix runner for testing multiple pinned software stacks.

The repository does **not** yet contain a general TorchInductor pass. The first public
release is a reproducibility artifact and data-collection platform. See
[Roadmap](docs/ROADMAP.md).

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

Both wins and losses are useful. Before opening a pull request:

```bash
layoutabi validate results/local_my_gpu --strict
```

Then follow [CONTRIBUTING.md](CONTRIBUTING.md). Do not edit measured JSON values by
hand. Remove hostnames or other identifying metadata if required by local policy.

## Scope and limitations

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
