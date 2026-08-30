# Workload provenance

The benchmark module is a dependency-free, source-equivalent reconstruction of the
LinearAttention implementation from:

- Repository: `lucidrains/denoising-diffusion-pytorch`
- Audited commit: `faed4db28e724735323fa91c70aa9b28a6e1cbac`
- Source file: `denoising_diffusion_pytorch/denoising_diffusion_pytorch.py`
- Audited regions: helper normalization and LinearAttention implementation
- Upstream license: MIT

The 256x256 motivating configuration comes from the public CMB foreground diffusion
pipeline:

- Repository: `AlexBM173/cmb_foregrounds_diffusion`
- Audited commit: `04668963f0b349cfb1d85366380086d07e77b662`
- Public configuration: 256x256, dimension 64, dimension multipliers `(1, 2, 4, 8)`,
  flash attention enabled
- Audited dependency version: `denoising-diffusion-pytorch` tag `2.2.5`, commit
  `8fac8e52126f5d8dbf93c4a8c3af4e9924000369`

No explicit license was visible in the audited CMB repository. This project therefore
does not copy, package, or redistribute CMB source code or data. It cites the public
repository only as provenance for the naturally occurring configuration. The
source-equivalent module is derived solely from the separately MIT-licensed
`denoising-diffusion-pytorch` implementation documented above.

The harness preserves the relevant equations and physical layout construction while
adding an explicit policy switch after K Softmax. It does not claim to be the complete
upstream training or inference pipeline.

## Second public graph: Efficient Attention

The second independent source-equivalent module reconstructs Efficient Attention from:

- Repository: `cmsflash/efficient-attention`
- Audited commit: `46a5f9eaf09470affb0ab30932b7748cc3c871ef`
- Source file: `efficient_attention.py`
- Upstream license: MIT
- Paper: Shen et al., Efficient Attention: Attention with Linear Complexities, WACV 2021

This is not the lucidrains LinearAttention module under another repository. The
published implementation loops over heads on rank-3 tensors. The reconstruction
keeps the per-head softmax-K, `K @ V^T`, and `context^T @ Q` equations and expresses
them as a batched rank-4 tensor so the bounded matcher can see the GEMM.

v0.5 ships the graph and matcher coverage. It does not claim a measured speedup on
this module. L40S remains the only complete public device reference (graph 1). Orin
and other architectures are community/held-out slots, not implied by extra software
stacks.

New optimizer graphs are registered under `layoutabi/workloads/cases/`. See that
directory's README for the drop-in spec. Adding a case should not require editing
the CLI.

## Public negative graph

Scaled dot-product attention follows the Vaswani et al. 2017 equations
(`softmax(QK^T) V`). Softmax sits after the first GEMM, so `layoutabi.optimize()`
must no-op rather than rewrite.
