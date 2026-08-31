"""Structured public exceptions. Importing this module does not require PyTorch."""

from __future__ import annotations


class LayoutABIError(Exception):
    """Base error for Layout ABI Lab."""


class MissingPyTorchError(LayoutABIError, ImportError):
    """Raised when a PyTorch-backed API is called without PyTorch installed."""

    def __init__(self, api: str = "layoutabi.optimize") -> None:
        super().__init__(
            f"{api} requires PyTorch. Install a CUDA-enabled build appropriate "
            "for this GPU; this package does not install PyTorch."
        )
        self.api = api


class InvalidArgumentError(LayoutABIError, ValueError):
    """Raised when a public argument is not in the supported set."""


class UnsupportedCUDAError(LayoutABIError):
    """Raised only by explicit CUDA helpers; optimize itself no-ops instead."""
