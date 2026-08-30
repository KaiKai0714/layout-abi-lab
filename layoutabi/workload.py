"""Audited source-equivalent diffusion LinearAttention workload.

Provenance:
  Repository: https://github.com/lucidrains/denoising-diffusion-pytorch
  Commit: faed4db28e724735323fa91c70aa9b28a6e1cbac
  Source: denoising_diffusion_pytorch/denoising_diffusion_pytorch.py

The upstream project is MIT licensed. This dependency-free reconstruction preserves
the relevant tensor equations and layouts while expressing fixed einops patterns as
reshape and expand operations. The only experimental addition is a policy switch after
K Softmax. See docs/PROVENANCE.md for the motivating public CMB configuration.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

POLICIES = ("direct", "repair_k", "repair_kv")


def bhnd_backed_view(x: Tensor) -> Tensor:
    """Return logical BHDN dimensions backed by contiguous BHND storage."""

    return x.transpose(-2, -1).contiguous().transpose(-2, -1)


def context_from_kv(k: Tensor, v: Tensor, policy: str) -> Tensor:
    """Apply a layout policy before the K-transpose-V matrix multiplication."""

    if policy not in POLICIES:
        raise ValueError(f"Unknown policy {policy!r}; expected one of {POLICIES}")
    if policy in ("repair_k", "repair_kv"):
        k = bhnd_backed_view(k)
    if policy == "repair_kv":
        v = bhnd_backed_view(v)
    return k @ v.transpose(-2, -1)


def public_chain(q: Tensor, k: Tensor, v: Tensor, policy: str = "direct") -> Tensor:
    """Run the normalize-to-KTV chain starting after QKV projection and memory concat."""

    q = q.softmax(dim=-2) * (q.shape[-2] ** -0.5)
    k = k.softmax(dim=-1)
    context = context_from_kv(k, v, policy)
    return context.transpose(-2, -1) @ q


class RMSNorm(nn.Module):
    """Source-equivalent spatial RMS normalization used by the public module."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, dim=1) * self.g * self.scale


class PublicDiffusionLinearAttention(nn.Module):
    """Public LinearAttention equations with an explicit post-Softmax layout policy."""

    def __init__(
        self,
        dim: int = 64,
        heads: int = 4,
        dim_head: int = 32,
        num_mem_kv: int = 4,
        policy: str = "direct",
    ) -> None:
        super().__init__()
        if policy not in POLICIES:
            raise ValueError(f"Unknown policy {policy!r}")
        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        self.num_mem_kv = num_mem_kv
        self.policy = policy
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), RMSNorm(dim))

    def forward(self, x: Tensor) -> Tensor:
        batch, _channels, height, width = x.shape
        q, k, v = self.to_qkv(self.norm(x)).chunk(3, dim=1)
        spatial_n = height * width
        q = q.reshape(batch, self.heads, self.dim_head, spatial_n)
        k = k.reshape(batch, self.heads, self.dim_head, spatial_n)
        v = v.reshape(batch, self.heads, self.dim_head, spatial_n)

        memory_k = self.mem_kv[0].unsqueeze(0).expand(batch, -1, -1, -1)
        memory_v = self.mem_kv[1].unsqueeze(0).expand(batch, -1, -1, -1)
        k = torch.cat((memory_k, k), dim=-1)
        v = torch.cat((memory_v, v), dim=-1)

        q = q.softmax(dim=-2) * self.scale
        k = k.softmax(dim=-1)
        context = context_from_kv(k, v, self.policy)
        out = context.transpose(-2, -1) @ q
        out = out.reshape(batch, self.heads * self.dim_head, height, width)
        return self.to_out(out)


def make_chain_inputs(
    resolution: int,
    *,
    batch: int = 1,
    heads: int = 4,
    dim_head: int = 32,
    num_mem_kv: int = 4,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> tuple[Tensor, Tensor, Tensor]:
    """Create chain inputs with the natural memory-token residue of the public module."""

    spatial_n = resolution * resolution
    consumer_n = spatial_n + num_mem_kv
    q = torch.randn(batch, heads, dim_head, spatial_n, device=device, dtype=dtype)
    k = torch.randn(batch, heads, dim_head, consumer_n, device=device, dtype=dtype)
    v = torch.randn(batch, heads, dim_head, consumer_n, device=device, dtype=dtype)
    return q, k, v
