"""Thin orchestration over bdnmc-dune (BdNMC cards -> HEPEVT -> edep-sim job). Not vendored.

The generator and the edep-sim job stay in bdnmc-dune (``pip install 'dunefastsim[ldm]'``); this package only hands
their edep-sim output to :func:`dunefastsim.steps.edepsim.convert`.
"""

from __future__ import annotations

BDNMC_DUNE_PIN = "165be9ebd7cd8a6dbb1038e7ed3c62ea36235955"


def status() -> dict:
    """Whether bdnmc-dune imports; never raises."""
    out = {"pin": BDNMC_DUNE_PIN, "importable": False}
    try:
        import bdnmc_dune  # noqa: F401

        out["importable"] = True
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out
