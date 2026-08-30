"""Public negative graph: scaled dot-product attention.

Equations follow Vaswani et al., Attention Is All You Need (2017),
NeurIPS. This is not a vendored copy of any application repository.

The softmax sits after QK^T, so the bounded K-softmax → KTV matcher must
reject it. Repair is not applied.
"""

from __future__ import annotations

from torch import Tensor, nn

GRAPH_FINGERPRINT = "scaled_dot_product_attention:vaswani2017"


class PublicScaledDotProductAttention(nn.Module):
    """Standard QK^T softmax V attention used as a public no-op control."""

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        scale = query.shape[-1] ** -0.5
        scores = (query @ key.transpose(-2, -1)) * scale
        return scores.softmax(dim=-1) @ value
