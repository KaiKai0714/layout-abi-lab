# Layout ABI Lab

This project studies a hidden GPU contract:

```text
producer output layout -> vendor GEMM kernel family -> discrete latency cliff
```

Numerically equivalent layouts can select different opaque GEMM families. The
independent variables include the environment (GPU, PyTorch/CUDA stack, dtype,
eager vs compile), physical strides, reduction length, and operand-pointer
alignment. **Linear Attention is a public real-world witness, not the limit of
the theory.** Other graphs can be added; the current automatic optimizer only
rewrites one frozen KTV pattern and otherwise no-ops.

A conservative helper can then choose to keep the producer layout or to
materialize it. Materialization is an experimental contrast and a possible
fix, not the default answer.

Version 1.0.0 freezes the first public API, matcher, and measurement contract.
Public evidence includes three L40S software stacks, a six-shape residue sweep,
a compiled mechanism audit, matching 100-cell two-operand pointer audits on
L40S and Jetson Orin, and an Orin eager-128 boundary where repair is slower.
See the [result index](RESULTS_INDEX.md) and
[v1.0.0 release notes](docs/RELEASE_1.0.0.md).

## What we need from the community

The useful contribution is **measured evidence on machines and lengths we do
not have**, including cases where repair loses or no kernel-family change
appears.

1. **Run the protocol on a GPU that is not already in the index**, or span all
   three FP16 residue classes: divisible by 8, even non-multiple, and odd.
2. **Keep isolated KTV profiler kernel names** in the bundle. The mechanism prior
   is `align8/ldg8` → `align2` → `align1`; latency says whether repair paid off.
3. **Score the binary safety planner with `layoutabi evaluate-planner`.** It
   deliberately merges the intermediate and slowest tiers into repair.
4. **Run `audit-pointer` on another GPU architecture.** This tests whether the
   least-aligned tier among N and both GEMM operands transfers.
5. **Report optimizer failures** (crash, wrong rewrite, unexpected no-op)
   with the [RC issue template](https://github.com/KaiKai0714/layout-abi-lab/issues/new?template=rc_feedback.yml).

The repository includes a dedicated paired L40S sweep for the alignment
contrast: `containers/matrix_three_level_l40s.json`. It measures neighboring
resolutions around both 128 and 256 without changing the frozen reference
matrix; see [results/README.md](results/README.md) for the command.

Do not send hand-edited JSON, only-favorable cells, or a new matcher pattern
during this RC. Details: [CONTRIBUTING.md](CONTRIBUTING.md).

## What we already measured

On three pinned L40S stacks, compiled repair-KV reduced 256×256 module latency
by **13.95% to 15.19%**. Eager 128 on L40S and Orin is a **direct-win**: always-
repair would be a false repair. The original 128/256 matrix contains the
intermediate residue class; the newer 126–128 and 254–256 sweep covers all three
classes and observes `align8`, `align2`, and `align1` in the isolated eager KTV
consumer.

The controlled pointer audits contain 100/100 valid cells on each device. On
both L40S and Orin, the observed family always equals the least-aligned tier
among N, the K pointer, and the V pointer. At aligned N, L40S median `align2`
and `align1` latency were **1.23×** and **2.37×** its `align8` baseline; Orin
measured **1.83×** and **4.10×** relative to `ldg8`. The compiled audit also
shows that Inductor may materialize or rewrite a layout, so eager family rules
cannot be copied blindly into compiled execution.

Evidence: [software-stack matrix](results/reference_l40s/SOFTWARE_STACK_MATRIX.md),
[three-level sweep](results/reference_l40s/three_level_sweep/torch2.11_cuda12.8/SUMMARY.md),
[L40S pointer audit](results/reference_l40s/pointer_alignment/torch2.11_cuda12.8/SUMMARY.md),
[Orin pointer audit](results/community/orin_pointer_alignment/torch2.7_cuda12.8/SUMMARY.md),
and [compiled audit](results/reference_l40s/compile_audit/torch2.11_cuda12.8/SUMMARY.md).

## Quick start

GPU reproduction is documented for Linux + NVIDIA. Install a CUDA-enabled
PyTorch build that matches the GPU first. This package does **not** install
PyTorch. CPU tools (`supported`, `validate`, `evaluate-planner`, `rc-status`)
work without a GPU.

```bash
git clone https://github.com/KaiKai0714/layout-abi-lab.git
cd layout-abi-lab
python -m pip install -e .

layoutabi check
layoutabi reproduce --output results/local_rtx4090_torch2.11_cuda12.8_2026-08-30
layoutabi validate results/local_rtx4090_torch2.11_cuda12.8_2026-08-30
layoutabi evaluate-planner --results-root results
```

Name local runs `<gpu>_torch<pytorch>_cuda<cuda>_<yyyy-mm-dd>` so a later
submission can drop the `local_` prefix. See [results/README.md](results/README.md).
Smoke tests can use `/tmp`; do not keep them under `results/`.

```bash
layoutabi reproduce \
  --output /tmp/layoutabi_smoke \
  --resolutions 128 \
  --seeds 1701 \
  --cycles 2 \
  --iterations 3 \
  --skip-compile
```

The default protocol is resolutions 256 and 128 with three seeds. Compiled
controls take several minutes (isolated processes and caches).

`direct_ms / repair_ms > 1` means repair was faster at that scope. Always
read kernel families, `N % 8`, the full-module row, and the software stack.

### Controlled operand-pointer audit

The normal reproduction uses allocator-aligned tensors. To separate logical
length alignment from base-pointer alignment, run the full K-pointer × V-pointer
grid:

```bash
layoutabi audit-pointer \
  --output results/local_pointer_audit \
  --ns 65536,65538,65540,65543 \
  --offsets 0,2,4,8,16 \
  --cycles 12 \
  --iterations 20
layoutabi validate-pointer-audit results/local_pointer_audit
```

See [docs/POINTER_AUDIT.md](docs/POINTER_AUDIT.md) for controlled variables and
the interpretation boundary.

### Submit a bundle

```bash
layoutabi prepare-submission \
  results/local_rtx4090_torch2.11_cuda12.8_2026-08-30 \
  --name rtx4090_torch2.11_cuda12.8_2026-08-30
```

Then open a pull request with `results/community/<name>/`. The command
recomputes checksums and prints possible hostname, username, and private-path
findings. It does not upload.

## Optimizer

The automatic rewrite currently matches one frozen KTV graph
(`linear_attention_ktv_v1`). Unmatched graphs are left unchanged.

```python
import torch
from layoutabi import explain, inspect, optimize, supported
from layoutabi.workload import PublicDiffusionLinearAttention

print(supported()["pattern_id"])

model = PublicDiffusionLinearAttention().eval().half().cuda()
x = torch.randn(1, 64, 128, 128, device="cuda", dtype=torch.float16)
result = optimize(model, (x,), policy="autotune")
print(explain(result))
compiled = torch.compile(result.module)
```

Missing PyTorch raises `MissingPyTorchError`. See
[docs/SUPPORTED.md](docs/SUPPORTED.md) and
[docs/PATTERN_CONTRACT.md](docs/PATTERN_CONTRACT.md).

```bash
layoutabi supported
layoutabi inspect-model --resolution 128
layoutabi optimize-model --workload scaled_dot_product --resolution 16 --policy repair_kv
layoutabi audit-compile --output results/local_l40s_compile_audit_2026-08-31
```

## What is in the repo

- Public LinearAttention and Efficient Attention reconstructions, plus an SDPA
  no-op control (witnesses and a negative graph, not the whole theory)
- Direct / repair-K / repair-KV as a numerical contrast, plus eager and
  isolated compiled protocol
- Community result schema, checksums, and `prepare-submission`
- Conservative `layoutabi.optimize()` for the currently matched KTV pattern
- Compiled mechanism audit (graphs, Inductor IR, profiler names)
- Three-level residue reporting plus scoring of the conservative binary planner
- Controlled full-factorial K-pointer × V-pointer mechanism audit

Longer notes: [architecture](docs/ARCHITECTURE.md),
[claims](docs/CLAIMS_AND_LIMITATIONS.md),
[roadmap](docs/ROADMAP.md).

## Scope

- Theory: producer layout vs vendor GEMM family, including feature-map length
- Current automatic optimizer: one frozen inference KTV pattern, fixed shapes, FP16
- Matching L40S and Orin pointer grids complete the bounded cross-architecture
  FP16 mechanism test; Orin also provides an eager-128 no-repair boundary
- FP16 three-tier family names are backend observations; the binary safety policy is not an oracle
- BF16, INT8, FP8, larger batches, and unmatched graphs are boundaries
- A module-level result is not a full diffusion-model win

## License and citation

Apache License 2.0. Upstream workload licenses are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Cite [CITATION.cff](CITATION.cff).

Maintainer: Sheng-Kai Ku (`ethankai0714@gmail.com`).
