"""Charge readout through JAXTPC (DeepLearnPhysics/JAXTPC). Not vendored: JAXTPC has no licence yet.

Install: clone JAXTPC at :data:`JAXTPC_PIN`, ``pip install -e`` it (the only install it supports: its config and
response files resolve relative to the checkout), and set ``JAXTPC_ROOT`` to the checkout.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

JAXTPC_REPO = "https://github.com/DeepLearnPhysics/JAXTPC"
JAXTPC_PIN = "50ed555ec69adf1c5a7fdf67af81c544a751dd2e"  # main, 2026-10-17 (v0.9.0)
ENV = "JAXTPC_ROOT"


def root() -> Path | None:
    """The JAXTPC checkout from ``JAXTPC_ROOT``, or None."""
    value = os.environ.get(ENV)
    return Path(value) if value else None


def status() -> dict:
    """Where JAXTPC is, its commit, whether it imports; never raises."""
    out = {"pin": JAXTPC_PIN, "root": None, "commit": None, "pinned": False, "importable": False}
    r = root()
    if r is None or not (r / "tools" / "simulation.py").is_file():
        out["error"] = f"set {ENV} to a JAXTPC checkout"
        return out
    out["root"] = str(r)
    try:
        out["commit"] = subprocess.run(["git", "-C", str(r), "rev-parse", "HEAD"], capture_output=True, text=True,
                                       check=True).stdout.strip()
        out["pinned"] = out["commit"] == JAXTPC_PIN
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        _import()
        out["importable"] = True
    except Exception as exc:  # noqa: BLE001 - report, do not raise
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _import():
    """Import JAXTPC's ``tools`` package from the checkout (its import root is the generic name ``tools``)."""
    r = root()
    if r is None:
        raise ImportError(f"{ENV} is not set")
    if str(r) not in sys.path:
        sys.path.insert(0, str(r))
    from tools import simulation  # noqa: F401

    return simulation


def ndlar_template() -> Path:
    """JAXTPC's own ND-LAr config in the checkout (template for physics/readout blocks; its volumes are replaced)."""
    r = root()
    if r is None:
        raise FileNotFoundError(f"{ENV} is not set")
    return r / "config" / "dune_ndlar_config.yaml"
