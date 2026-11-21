"""One ND-LAr frame for charge and light, and JAXTPC's volume configuration generated from the geometry.

The frame is detector-centred: the origin is the centre of the union box of ND-LAr's active-argon placements
(``volTPCActive``) in the GDML world frame, axes unchanged (DUNE's hall GDML: +y up, +z downstream, drift along x).
For DUNE's hall geometry ``nd_hall_with_lar_tms_sand_stt1_v2026.03.06.gdml`` that centre is (0, -0.668713, 6.657559) m
(bdnmc-dune ``data/frames/nd_hall_v2026.03.06.toml``); any other GDML needs its own reading (:func:`boxes_from_gdml`).

JAXTPC describes a detector as drift volumes, each an axis-aligned box (ranges in cm) with a drift direction (+-1
along x) and a pixel readout. :func:`jaxtpc_volumes` builds that list from the active boxes; :func:`layout` counts
modules along each axis, the check that exposed JAXTPC's shipped ``dune_ndlar_config.yaml`` (5 modules across x, 7
along z) against DUNE's geometry (7 across, 5 along).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ND_HALL_V2026_03_06_OFFSET_M = (0.0, -0.668713, 6.657559)
LARPIX_PITCH_CM = 0.372


@dataclass(frozen=True)
class Boxes:
    """Axis-aligned active boxes in the world frame: centres (n, 3) and half-sizes (n, 3), mm."""

    centres_mm: np.ndarray
    half_mm: np.ndarray

    def union(self):
        lo = (self.centres_mm - self.half_mm).min(axis=0)
        hi = (self.centres_mm + self.half_mm).max(axis=0)
        return lo, hi

    def offset_mm(self) -> np.ndarray:
        """The detector-frame origin in the world frame: the union box's centre."""
        lo, hi = self.union()
        return 0.5 * (lo + hi)


def boxes_from_gdml(path: str, volume: str = "volTPCActive") -> Boxes:
    """Active boxes from a GDML file, via bdnmc-dune's reader (``pip install dunefastsim[ldm]``).

    Rotated placements are allowed only if they leave the box axis-aligned (the 180-degree turns of drift volumes).
    """
    from bdnmc_dune import gdml  # dependency, not vendored

    geom = gdml.load(path)
    half = geom.box_half(volume)
    centres, halves = [], []
    for _, rot, pos in geom.instances(volume):
        r = np.abs(np.round(rot, 9))
        if not np.allclose(r, np.eye(3)):
            raise ValueError(f"{volume}: a placement is not axis-aligned")
        centres.append(pos)
        halves.append(np.abs(rot) @ half)
    return Boxes(np.array(centres), np.array(halves))


def _groups(values, tol):
    """Sorted distinct values (within tol)."""
    out = []
    for v in np.sort(values):
        if not out or v - out[-1] > tol:
            out.append(v)
    return np.array(out)


def layout(boxes: Boxes, tol_mm: float = 1.0) -> dict:
    """Count drift volumes and modules along each axis (a module = two drift volumes sharing a cathode in x)."""
    nx = len(_groups(boxes.centres_mm[:, 0], tol_mm))
    ny = len(_groups(boxes.centres_mm[:, 1], tol_mm))
    nz = len(_groups(boxes.centres_mm[:, 2], tol_mm))
    lo, hi = boxes.union()
    return {"drift_volumes": len(boxes.centres_mm), "x_columns": nx, "modules_x": nx // 2, "y_rows": ny,
            "modules_z": nz, "union_size_m": ((hi - lo) / 1e3).round(4).tolist()}


def drift_directions(centres_x_mm, tol_mm: float = 1.0) -> np.ndarray:
    """Drift direction (+-1 along x) of each box: columns pair up into modules from low x; the lower column of a
    pair drifts towards -x (away from the shared cathode), the upper one towards +x. Reproduces JAXTPC's
    ``dune_ndlar_config.yaml``; to be checked against the GDML's placement rotations (plan, phase 3)."""
    cols = _groups(np.asarray(centres_x_mm), tol_mm)
    if len(cols) % 2:
        raise ValueError(f"{len(cols)} drift columns along x: not pairs of drift volumes")
    k = np.searchsorted(cols, np.asarray(centres_x_mm) - tol_mm)
    return np.where(k % 2 == 0, -1, 1)


def jaxtpc_volumes(boxes: Boxes, offset_mm=None, pitch_cm: float = LARPIX_PITCH_CM) -> list[dict]:
    """JAXTPC ``volumes:`` entries (ranges in cm, detector frame) for the active boxes, ordered by z, then x.

    ``pixel_shape`` is (y, z) pixels: floor(extent / pitch). JAXTPC's shipped config uses 800 x 256 for a
    297.6 x 95.232 cm face; DUNE's v2026.03.06 boxes are 299.6 x 95.7 cm (805 x 257 by this rule): which active face
    the pixel planes cover is an open question of the plan.
    """
    off = boxes.offset_mm() if offset_mm is None else np.asarray(offset_mm, dtype=np.float64)
    c = boxes.centres_mm - off
    lo, hi = (c - boxes.half_mm) / 10.0, (c + boxes.half_mm) / 10.0
    drift = drift_directions(boxes.centres_mm[:, 0])
    order = np.lexsort((c[:, 0], c[:, 2]))
    vols = []
    for i, j in enumerate(order):
        size = hi[j] - lo[j]
        vols.append({
            "id": i,
            "description": f"drift volume at ({c[j, 0] / 1e3:.3f}, {c[j, 1] / 1e3:.3f}, {c[j, 2] / 1e3:.3f}) m",
            "geometry": {"ranges": [[round(float(lo[j, a]), 4), round(float(hi[j, a]), 4)] for a in range(3)],
                         "drift_direction": int(drift[j])},
            "readout": {"type": "pixel", "pixel_pitch": pitch_cm,
                        "pixel_shape": [int(size[1] // pitch_cm), int(size[2] // pitch_cm)]},
        })
    return vols


def boxes_from_jaxtpc(config: dict) -> Boxes:
    """Active boxes of a JAXTPC detector config (detector frame, cm -> mm), for comparing layouts."""
    r = np.array([v["geometry"]["ranges"] for v in config["volumes"]], dtype=np.float64) * 10.0
    return Boxes(r.mean(axis=2), 0.5 * (r[:, :, 1] - r[:, :, 0]))


def jaxtpc_config(template: dict, boxes: Boxes, offset_mm=None, pitch_cm: float = LARPIX_PITCH_CM) -> dict:
    """A JAXTPC detector config: the template's physics/readout blocks with ``volumes`` generated from ``boxes``."""
    out = {k: v for k, v in template.items() if k != "volumes"}
    out["volumes"] = jaxtpc_volumes(boxes, offset_mm, pitch_cm)
    return out
