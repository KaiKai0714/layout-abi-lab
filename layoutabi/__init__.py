"""Public package for Layout ABI Lab."""

from __future__ import annotations

from typing import Any

from .errors import (
    InvalidArgumentError,
    LayoutABIError,
    MissingPyTorchError,
    UnsupportedCUDAError,
)
from .explain import explain
from .supported import supported

__version__ = "0.8.0"

__all__ = [
    "InvalidArgumentError",
    "LayoutABIError",
    "MissingPyTorchError",
    "UnsupportedCUDAError",
    "__version__",
    "cache_info",
    "clear_cache",
    "explain",
    "inspect",
    "optimize",
    "supported",
]


def optimize(*args: Any, **kwargs: Any) -> Any:
    """Rewrite a supported inference graph, or return the original module.

    Importing ``layoutabi`` does not require PyTorch. Calling ``optimize`` does.
    This package does not install a PyTorch wheel.
    """

    try:
        from .optimizer.api import optimize as impl
    except ImportError as exc:
        raise MissingPyTorchError("layoutabi.optimize") from exc
    return impl(*args, **kwargs)


def inspect(*args: Any, **kwargs: Any) -> Any:
    """Capture and match without rewriting.

    Importing ``layoutabi`` does not require PyTorch. Calling ``inspect`` does.
    """

    try:
        from .optimizer.api import inspect as impl
    except ImportError as exc:
        raise MissingPyTorchError("layoutabi.inspect") from exc
    return impl(*args, **kwargs)


def cache_info(*args: Any, **kwargs: Any) -> Any:
    """Return optimizer decision-cache status. Does not require a GPU."""

    from .optimizer.cache import cache_info as impl

    return impl(*args, **kwargs)


def clear_cache(*args: Any, **kwargs: Any) -> Any:
    """Erase optimizer decision-cache entries."""

    from .optimizer.cache import clear_cache as impl

    return impl(*args, **kwargs)
