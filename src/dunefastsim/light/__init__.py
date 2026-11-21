"""Light readout through lumen 0.2 + lumen-tpc (OmarAlterkait, Apache-2.0). Not vendored.

``pip install 'dunefastsim[light]'`` on Linux (CUDA 12 for full detectors). lumen-tpc's ``dune_ndlar`` recipe
converts ND-LAr from the TDR production GDML (2x2_sim v1.0.3) with 1400 ArCLight tiles; its frame differs from
DUNE's v2026.03.06 hall geometry used for charge until the plan's phase 5 maps them.
"""

from __future__ import annotations

DETECTOR = "lumen_tpc:dune_ndlar"
VALUES = "lumen_tpc:dune_ndlar_arclight"
LUMEN_TPC_PIN = "66032a0a71b4369f07661427f8d2fc4d2d81267d"
LUMEN_VERSION = "0.2.0"


def status() -> dict:
    """Whether lumen and lumen-tpc import, their versions and JAX devices; never raises."""
    out = {"detector": DETECTOR, "lumen": None, "lumen_tpc": None, "importable": False}
    try:
        import lumen
        import lumen_tpc

        out["lumen"] = getattr(lumen, "__version__", "?")
        out["lumen_tpc"] = getattr(lumen_tpc, "__version__", "?")
        out["importable"] = True
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out
