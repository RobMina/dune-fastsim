"""Write and check step-record files (the layout in :mod:`dunefastsim.steps.schema`)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from dunefastsim.steps import schema


@dataclass
class StepEvent:
    """One event of the step record. ``steps`` need not be sorted; the writer sorts them by ``track_id``."""

    event_id: int
    steps: np.ndarray
    particles: np.ndarray = field(default_factory=lambda: schema.empty(schema.PARTICLE))
    primaries: np.ndarray = field(default_factory=lambda: schema.empty(schema.PRIMARY))
    vertices: np.ndarray = field(default_factory=lambda: schema.empty(schema.VERTEX))


def associate(steps: np.ndarray, particles: np.ndarray) -> np.ndarray:
    """[start, end) of each particle's steps in ``steps`` (sorted by track_id); row k is ``particles[k]``.

    Particles must be indexed by track id (row k has ``track_id`` k), as edep-sim numbers its trajectories.
    """
    tids = particles["track_id"].astype(np.int64)
    if len(tids) and not np.array_equal(tids, np.arange(len(tids))):
        raise ValueError("particles must be ordered with track_id == row index")
    st = steps["track_id"].astype(np.int64)
    if len(st) and np.any(np.diff(st) < 0):
        raise ValueError("steps must be sorted by track_id")
    out = np.zeros(len(tids), dtype=schema.ASS)
    out["start"] = np.searchsorted(st, tids, side="left")
    out["end"] = np.searchsorted(st, tids, side="right")
    return out


def write_events(path: str | Path, events, run_id: int = 0, attrs: dict | None = None) -> Path:
    """Write ``events`` (iterable of :class:`StepEvent`) to ``path``; return the path."""
    events = list(events)
    path = Path(path)
    with h5py.File(path, "w") as f:
        dsets = {name: f.create_dataset(name, (len(events),), dtype=h5py.vlen_dtype(dt))
                 for name, dt in schema.DATASETS.items()}
        table = np.zeros(len(events), dtype=schema.EVENT)
        for i, ev in enumerate(events):
            steps = np.asarray(ev.steps, dtype=schema.PSTEP)
            steps = steps[np.argsort(steps["track_id"], kind="stable")]
            particles = np.asarray(ev.particles, dtype=schema.PARTICLE)
            rows = {
                "pstep/lar_vol": steps,
                "particle/geant4": particles,
                "primary/geant4": np.asarray(ev.primaries, dtype=schema.PRIMARY),
                "vertex/geant4": np.asarray(ev.vertices, dtype=schema.VERTEX),
                "ass/particle_pstep_lar_vol": associate(steps, particles),
            }
            for name, arr in rows.items():
                dsets[name][i] = arr
            table[i] = (run_id, ev.event_id, len(rows["vertex/geant4"]), len(rows["primary/geant4"]),
                        len(particles), len(steps))
        f.create_dataset("event/geant4", data=table)
        f.attrs["writer"] = "dunefastsim.steps"
        for k, v in (attrs or {}).items():
            f.attrs[k] = v
    return path


def check(path: str | Path) -> dict:
    """Check a step-record file against the schema; return a summary. Raises ValueError on a mismatch."""
    out = {"path": str(path), "events": 0, "steps": 0, "de_MeV": 0.0}
    with h5py.File(path, "r") as f:
        for name, fields in schema.REQUIRED.items():
            if name not in f:
                raise ValueError(f"{path}: missing dataset {name}")
            base = h5py.check_vlen_dtype(f[name].dtype)
            missing = [x for x in fields if base is None or x not in base.names]
            if missing:
                raise ValueError(f"{path}: {name} lacks fields {missing}")
        n = f["pstep/lar_vol"].shape[0]
        out["events"] = int(n)
        for i in range(n):
            s = f["pstep/lar_vol"][i]
            out["steps"] += int(len(s))
            out["de_MeV"] += float(np.sum(s["de"], dtype=np.float64))
    return out
