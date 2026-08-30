"""Named public and synthetic workloads for optimizer generalization tests."""

from __future__ import annotations

from typing import Any

from ..identity import DEFAULT_GRAPH_FINGERPRINT
from .synthetic import synthetic_cells

EFFICIENT_FINGERPRINT = (
    "efficient_attention:cmsflash/efficient-attention@"
    "46a5f9eaf09470affb0ab30932b7748cc3c871ef"
)
SDPA_FINGERPRINT = "scaled_dot_product_attention:vaswani2017"

CATALOG: dict[str, dict[str, Any]] = {
    "diffusion_linear_attention": {
        "id": "diffusion_linear_attention",
        "role": "positive_reference",
        "title": "Diffusion LinearAttention (lucidrains reconstruction)",
        "repository": "lucidrains/denoising-diffusion-pytorch",
        "commit": "faed4db28e724735323fa91c70aa9b28a6e1cbac",
        "license": "MIT",
        "graph_fingerprint": DEFAULT_GRAPH_FINGERPRINT,
        "expected_optimizer": "match",
        "independent_of": None,
    },
    "efficient_attention": {
        "id": "efficient_attention",
        "role": "second_public",
        "title": "Efficient Attention (Shen et al. WACV 2021 reconstruction)",
        "repository": "cmsflash/efficient-attention",
        "commit": "46a5f9eaf09470affb0ab30932b7748cc3c871ef",
        "license": "MIT",
        "graph_fingerprint": EFFICIENT_FINGERPRINT,
        "expected_optimizer": "match",
        "independent_of": "lucidrains/denoising-diffusion-pytorch",
    },
    "scaled_dot_product": {
        "id": "scaled_dot_product",
        "role": "negative",
        "title": "Scaled dot-product attention (Vaswani et al. 2017 equations)",
        "repository": None,
        "commit": None,
        "license": "public equations; no third-party source vendored",
        "graph_fingerprint": SDPA_FINGERPRINT,
        "expected_optimizer": "noop",
        "independent_of": "lucidrains/denoising-diffusion-pytorch",
    },
}


def list_workloads() -> list[dict[str, Any]]:
    return [dict(spec) for spec in CATALOG.values()]


def _dtype(name: str) -> Any:
    import torch

    mapping = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dtype {name!r}") from exc


def _place(module: Any, tensors: tuple[Any, ...], device: str) -> tuple[Any, tuple[Any, ...]]:
    if device == "cuda":
        module = module.cuda()
        tensors = tuple(tensor.cuda() for tensor in tensors)
    return module, tensors


def make_workload(
    workload_id: str,
    *,
    resolution: int = 128,
    batch: int = 1,
    dtype: str = "fp16",
    device: str = "cpu",
) -> tuple[Any, tuple[Any, ...]]:
    """Build an eval module and example inputs for a named workload."""

    import torch

    if workload_id not in CATALOG:
        known = ", ".join(sorted(CATALOG))
        raise ValueError(f"Unknown workload {workload_id!r}; expected one of {known}")
    if resolution <= 0 or batch <= 0:
        raise ValueError("resolution and batch must be positive")
    torch_dtype = _dtype(dtype)
    if workload_id == "diffusion_linear_attention":
        from ..workload import PublicDiffusionLinearAttention

        module = PublicDiffusionLinearAttention(policy="direct").eval()
        if torch_dtype == torch.float16:
            module = module.half()
        elif torch_dtype == torch.bfloat16:
            module = module.to(dtype=torch_dtype)
        sample = torch.randn(batch, 64, resolution, resolution, dtype=torch_dtype)
        return _place(module, (sample,), device)
    if workload_id == "efficient_attention":
        from .efficient_attention import PublicEfficientAttention

        module = PublicEfficientAttention().eval()
        if torch_dtype == torch.float16:
            module = module.half()
        elif torch_dtype == torch.bfloat16:
            module = module.to(dtype=torch_dtype)
        sample = torch.randn(batch, 64, resolution, resolution, dtype=torch_dtype)
        return _place(module, (sample,), device)
    from .scaled_dot_product import PublicScaledDotProductAttention

    module = PublicScaledDotProductAttention().eval()
    qkv = tuple(
        torch.randn(batch, 4, resolution, 32, dtype=torch_dtype) for _ in range(3)
    )
    return _place(module, qkv, device)
