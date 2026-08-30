"""Synthetic shape, batch, and dtype cells. These do not replace public graphs."""

from __future__ import annotations

from typing import Any

# Nearby resolutions around the 128/256 reference pair, plus natural residues.
SYNTHETIC_RESOLUTIONS = (64, 96, 128, 160, 192, 224, 256, 384)
SYNTHETIC_BATCHES = (1, 2)
SYNTHETIC_DTYPES = ("fp16", "bf16")


def synthetic_cells() -> list[dict[str, Any]]:
    """Return boundary cells. fp16/batch-1 is the mainline; others are boundaries."""

    cells = []
    for resolution in SYNTHETIC_RESOLUTIONS:
        for batch in SYNTHETIC_BATCHES:
            for dtype in SYNTHETIC_DTYPES:
                role = "mainline" if batch == 1 and dtype == "fp16" else "boundary"
                if dtype == "bf16":
                    role = "boundary_unsupported_dtype"
                elif batch > 1:
                    role = "boundary_batch"
                spatial = resolution * resolution
                cells.append(
                    {
                        "resolution": resolution,
                        "batch": batch,
                        "dtype": dtype,
                        "n_mod_8": spatial % 8,
                        "consumer_n_mod_8": (spatial + 4) % 8,
                        "role": role,
                    }
                )
    return cells
