"""Device floating-point mode via ``DUNE_DEVICE_FLOAT`` (float32 | float64). Fail closed.

Resolution is lazy: call :func:`resolve` after ``site.env`` is loaded. Importing this module never reads the
environment.
"""

from __future__ import annotations

import os

import numpy as np

ENV = "DUNE_DEVICE_FLOAT"
DEFAULT = "float64"
ALLOWED = ("float32", "float64")


def resolve(name: str | None = None) -> str:
    """Return ``float32`` or ``float64``; unset -> :data:`DEFAULT`; any other value raises."""
    raw = name if name is not None else (os.environ.get(ENV) or DEFAULT)
    if raw not in ALLOWED:
        raise ValueError(f"{ENV} must be float32 or float64, not {raw!r}")
    return raw


def compute_dtype(name: str | None = None) -> np.dtype:
    """The device float dtype; ``name`` overrides the environment for one call."""
    return np.dtype(resolve(name))


def apply_jax_x64(name: str | None = None) -> str:
    """Set ``jax_enable_x64`` to match the resolved mode when JAX is importable; return the mode."""
    resolved = resolve(name)
    try:
        import jax
    except ImportError:
        return resolved
    jax.config.update("jax_enable_x64", resolved == "float64")
    return resolved
