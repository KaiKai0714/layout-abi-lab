"""Public package for Layout ABI Lab."""

from __future__ import annotations

from typing import Any

__version__ = "0.3.0"


def optimize(*args: Any, **kwargs: Any) -> Any:
    """Lazy wrapper so importing layoutabi does not require PyTorch."""

    try:
        from .optimizer.api import optimize as impl
    except ImportError as exc:
        raise RuntimeError(
            "layoutabi.optimize requires PyTorch. Install a CUDA-enabled build "
            "appropriate for this GPU; this package does not install PyTorch."
        ) from exc
    return impl(*args, **kwargs)


def inspect(*args: Any, **kwargs: Any) -> Any:
    """Lazy wrapper so importing layoutabi does not require PyTorch."""

    try:
        from .optimizer.api import inspect as impl
    except ImportError as exc:
        raise RuntimeError(
            "layoutabi.inspect requires PyTorch. Install a CUDA-enabled build "
            "appropriate for this GPU; this package does not install PyTorch."
        ) from exc
    return impl(*args, **kwargs)
