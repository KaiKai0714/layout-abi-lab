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
