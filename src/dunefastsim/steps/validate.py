"""Checks of a step record against a detector configuration and against its source.

- :func:`containment`: how many steps (and how much deposited energy) fall in exactly one drift volume of a JAXTPC
  config, in none, or in more than one. Steps written in the detector frame from the segment detector
  ``volTPCActive`` should all be in exactly one volume when the config describes the same geometry.
- :func:`energy` : summed deposits against the primaries' kinetic energy (bdnmc-dune's ``check_edepsim.py`` reports
  the same ratio from the ROOT file).
"""

from __future__ import annotations

import h5py
import numpy as np


def _volume_boxes_mm(config: dict):
    r = np.array([v["geometry"]["ranges"] for v in config["volumes"]], dtype=np.float64) * 10.0  # cm -> mm
    return r[:, :, 0], r[:, :, 1]


def containment(path: str, config: dict, max_events: int | None = None, tol_mm: float = 1e-3) -> dict:
    """Counts of steps (and MeV) in 0, 1 or more volumes of ``config`` (a JAXTPC detector config dict)."""
    lo, hi = _volume_boxes_mm(config)
    lo, hi = lo - tol_mm, hi + tol_mm
    out = {"events": 0, "steps": [0, 0, 0], "de_MeV": [0.0, 0.0, 0.0], "outside_examples_mm": []}
    with h5py.File(path, "r") as f:
        n = f["pstep/lar_vol"].shape[0] if max_events is None else min(max_events, f["pstep/lar_vol"].shape[0])
        for i in range(n):
            s = f["pstep/lar_vol"][i]
            if not len(s):
                continue
            pos = np.column_stack([s["x"], s["y"], s["z"]]).astype(np.float64)
            k = np.zeros(len(pos), dtype=np.int64)
            for start in range(0, len(lo), 10):  # 10 volumes at a time keeps memory small
                inside = np.all((pos[:, None, :] >= lo[None, start:start + 10]) &
                                (pos[:, None, :] <= hi[None, start:start + 10]), axis=2)
                k += inside.sum(axis=1)
            cls = np.minimum(k, 2)
            for c in range(3):
                out["steps"][c] += int(np.sum(cls == c))
                out["de_MeV"][c] += float(np.sum(s["de"][cls == c], dtype=np.float64))
            if len(out["outside_examples_mm"]) < 5 and np.any(cls == 0):
                out["outside_examples_mm"].append(pos[cls == 0][0].round(1).tolist())
            out["events"] += 1
    tot, tde = sum(out["steps"]), sum(out["de_MeV"])
    out["fraction_steps"] = [x / tot if tot else 0.0 for x in out["steps"]]
    out["fraction_de"] = [x / tde if tde else 0.0 for x in out["de_MeV"]]
    out["legend"] = "index 0: in no volume, 1: in exactly one, 2: in more than one"
    return out


def energy(path: str, max_events: int | None = None) -> dict:
    """Summed deposits and primaries' kinetic energy (MeV), and their ratio."""
    de = ke = 0.0
    with h5py.File(path, "r") as f:
        n = f["pstep/lar_vol"].shape[0] if max_events is None else min(max_events, f["pstep/lar_vol"].shape[0])
        for i in range(n):
            de += float(np.sum(f["pstep/lar_vol"][i]["de"], dtype=np.float64))
            ke += float(np.sum(f["primary/geant4"][i]["ke"], dtype=np.float64))
    return {"events": int(n), "de_MeV": de, "primary_ke_MeV": ke, "ratio": de / ke if ke else float("nan")}
