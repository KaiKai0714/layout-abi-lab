"""Explicit shape buckets. Unseen sizes are not silently repaired."""

from __future__ import annotations

from typing import Any

# Inclusive upper bounds for a last-dimension (or spatial) size.
SHAPE_BUCKETS = (32, 64, 96, 128, 160, 192, 224, 256, 384, 512)
UNSEEN_SHAPE_ACTIONS = ("direct", "noop", "autotune")


def bucket_dim(size: int) -> int | None:
    """Map a size onto the smallest bucket that covers it, or None if out of range."""

    if size <= 0:
        return None
    for bound in SHAPE_BUCKETS:
        if size <= bound:
            return bound
    return None


def bucket_shape(shape: tuple[int, ...] | list[int]) -> list[int] | None:
    bucketed = []
    for size in shape:
        mapped = bucket_dim(int(size))
        if mapped is None:
            return None
        bucketed.append(mapped)
    return bucketed


def pointer_class(data_ptr: int) -> str:
    if data_ptr % 256 == 0:
        return "align256"
    if data_ptr % 128 == 0:
        return "align128"
    if data_ptr % 64 == 0:
        return "align64"
    if data_ptr % 32 == 0:
        return "align32"
    if data_ptr % 16 == 0:
        return "align16"
    if data_ptr % 8 == 0:
        return "align8"
    return "unaligned"


def tensor_layout_fields(tensor: Any) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "pointer_class": pointer_class(int(tensor.data_ptr())),
    }
