"""edep-sim ROOT (stock lbl-neutrino/edep-sim, TG4Event) -> step record.

Two layers:

- :func:`segments_to_steps` and :func:`trajectories_to_particles` are pure numpy and tested on synthetic input.
- :func:`convert` reads an edep-sim ROOT file with PyROOT (run it where ROOT and edep-sim's I/O library are, e.g. the
  bdnmc-dune image). **Not run yet** (phase 2 of the plan): check it on a real file before trusting it.

Conventions, matched to the sample file of JAXTPC (``muon.h5``, 2026-11-21): a step's position and time are the
midpoints of the segment; ``dx`` is the segment's track length (the chord if that is zero); ``theta`` / ``phi`` are
the chord direction's polar angle from +z and azimuth from +x. ``p`` is written as 0: in the sample it holds the
particle's mass, not a momentum, and JAXTPC does not read it.

Frame: edep-sim writes the GDML world frame (mm). Subtract ``offset_mm`` (the ND-LAr active-volume centre, see
:mod:`dunefastsim.geometry`) to get the detector-centred frame that JAXTPC's volume ranges use.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dunefastsim.steps import schema
from dunefastsim.steps.write import StepEvent, write_events

DEFAULT_DETECTOR = "volTPCActive"


def segments_to_steps(start, stop, de, track_length, track_id, primary_id, pdg_of_track,
                      offset_mm=(0.0, 0.0, 0.0)) -> np.ndarray:
    """PSTEP rows from edep-sim hit segments.

    start, stop   : (n, 4) x, y, z (mm), t (ns) of each segment's ends (edep-sim's GetStart / GetStop)
    de            : (n,) energy deposit (MeV)
    track_length  : (n,) GetTrackLength (mm)
    track_id      : (n,) the segment's first contributor (Contrib[0])
    primary_id    : (n,) GetPrimaryId (the primary ancestor's track id)
    pdg_of_track  : (n_tracks,) PDG code indexed by track id
    offset_mm     : subtracted from positions (world -> detector frame)
    """
    start = np.asarray(start, dtype=np.float64).reshape(-1, 4)
    stop = np.asarray(stop, dtype=np.float64).reshape(-1, 4)
    n = len(start)
    out = schema.empty(schema.PSTEP, n)
    mid = 0.5 * (start + stop)
    pos = mid[:, :3] - np.asarray(offset_mm, dtype=np.float64)
    out["x"], out["y"], out["z"] = pos[:, 0], pos[:, 1], pos[:, 2]
    out["t"] = mid[:, 3]
    d = stop[:, :3] - start[:, :3]
    chord = np.linalg.norm(d, axis=1)
    safe = np.where(chord > 0, chord, 1.0)
    out["theta"] = np.where(chord > 0, np.arccos(np.clip(d[:, 2] / safe, -1.0, 1.0)), 0.0)
    out["phi"] = np.where(chord > 0, np.arctan2(d[:, 1], d[:, 0]), 0.0)
    tl = np.asarray(track_length, dtype=np.float64)
    out["dx"] = np.where(tl > 0, tl, chord)
    out["de"] = de
    tid = np.asarray(track_id, dtype=np.int64)
    out["track_id"] = tid
    out["ancestor_track_id"] = primary_id
    pdg = np.asarray(pdg_of_track, dtype=np.int64)
    ok = (tid >= 0) & (tid < len(pdg))
    out["pdg"] = np.where(ok, pdg[np.clip(tid, 0, max(len(pdg) - 1, 0))] if len(pdg) else 0, 0)
    out["proc_start"] = out["proc_stop"] = -1
    out["subproc_start"] = out["subproc_stop"] = -1
    return out


def trajectories_to_particles(track_id, parent_id, pdg, mass, first, last, p_first,
                              offset_mm=(0.0, 0.0, 0.0)) -> np.ndarray:
    """PARTICLE rows from edep-sim trajectories, ordered by track id.

    first, last : (n, 4) x, y, z (mm), t (ns) of the first and last trajectory points
    p_first     : (n, 3) initial momentum (MeV)
    mass        : (n,) MeV
    The ancestor of each particle is found by walking parents up to a particle whose parent is -1.
    """
    order = np.argsort(np.asarray(track_id))
    tid = np.asarray(track_id, dtype=np.int64)[order]
    par = np.asarray(parent_id, dtype=np.int64)[order]
    n = len(tid)
    out = schema.empty(schema.PARTICLE, n)
    off = np.asarray(offset_mm, dtype=np.float64)
    f = np.asarray(first, dtype=np.float64).reshape(-1, 4)[order]
    lst = np.asarray(last, dtype=np.float64).reshape(-1, 4)[order]
    p = np.asarray(p_first, dtype=np.float64).reshape(-1, 3)[order]
    m = np.asarray(mass, dtype=np.float64)[order]
    out["x"], out["y"], out["z"] = (f[:, :3] - off).T
    out["t"] = f[:, 3]
    out["end_x"], out["end_y"], out["end_z"] = (lst[:, :3] - off).T
    out["end_t"] = lst[:, 3]
    out["px"], out["py"], out["pz"] = p.T
    out["ke"] = np.sqrt(np.sum(p * p, axis=1) + m * m) - m
    out["track_id"], out["parent_track_id"] = tid, par
    out["pdg"] = np.asarray(pdg)[order]
    out["mass"] = m
    # ancestor: iterate the parent map to a fixed point (depth is small)
    index = {t: i for i, t in enumerate(tid)}
    anc = tid.copy()
    for i in range(n):
        a = tid[i]
        seen = 0
        while par[index[a]] >= 0 and par[index[a]] in index and seen < n:
            a = par[index[a]]
            seen += 1
        anc[i] = a
    out["ancestor_track_id"] = anc
    out["proc_start"] = out["subproc_start"] = out["proc_end"] = out["subproc_end"] = -1
    return out


def convert(root_path: str | Path, out_path: str | Path, detector: str = DEFAULT_DETECTOR,
            offset_mm=(0.0, 0.0, 0.0), max_events: int | None = None, run_id: int = 0) -> Path:
    """Convert an edep-sim ROOT file to a step record (PyROOT; untested until phase 2)."""
    import ROOT  # noqa: N811 - available only where ROOT is

    ROOT.gROOT.SetBatch(True)
    if ROOT.gSystem.Load("libedepsim_io") < 0:
        raise RuntimeError("cannot load libedepsim_io (edep-sim's I/O library) into ROOT")
    f = ROOT.TFile.Open(str(root_path))
    tree = f.Get("EDepSimEvents")
    if not tree:
        raise ValueError(f"{root_path}: no EDepSimEvents tree")
    n = tree.GetEntries() if max_events is None else min(max_events, tree.GetEntries())
    events = []
    for i in range(n):
        tree.GetEntry(i)
        ev = tree.Event
        trajs = list(ev.Trajectories)
        tids = np.array([t.GetTrackId() for t in trajs], dtype=np.int64)
        pdg_by_tid = np.zeros(tids.max() + 1 if len(tids) else 0, dtype=np.int64)
        pdg_by_tid[tids] = [t.GetPDGCode() for t in trajs]

        def _xyzt(pt):
            v = pt.GetPosition()
            return (v.X(), v.Y(), v.Z(), v.T())

        particles = trajectories_to_particles(
            tids, [t.GetParentId() for t in trajs], pdg_by_tid[tids],
            [t.GetInitialMomentum().M() for t in trajs],
            [_xyzt(t.Points[0]) for t in trajs], [_xyzt(t.Points[t.Points.size() - 1]) for t in trajs],
            [(t.GetInitialMomentum().X(), t.GetInitialMomentum().Y(), t.GetInitialMomentum().Z()) for t in trajs],
            offset_mm)
        segs = []
        for name, vec in ev.SegmentDetectors:
            if str(name) == detector:
                segs = list(vec)
        steps = segments_to_steps(
            [(s.GetStart().X(), s.GetStart().Y(), s.GetStart().Z(), s.GetStart().T()) for s in segs],
            [(s.GetStop().X(), s.GetStop().Y(), s.GetStop().Z(), s.GetStop().T()) for s in segs],
            [s.GetEnergyDeposit() for s in segs], [s.GetTrackLength() for s in segs],
            [s.Contrib[0] for s in segs], [s.GetPrimaryId() for s in segs], pdg_by_tid, offset_mm)
        prims, verts = [], []
        for iid, vtx in enumerate(ev.Primaries):
            v = vtx.GetPosition()
            for p in vtx.Particles:
                mom = p.GetMomentum()
                prims.append((mom.X(), mom.Y(), mom.Z(), mom.E() - mom.M(), iid, p.GetTrackId(), p.GetPDGCode(),
                              mom.M(), str(p.GetName())))
            row = schema.empty(schema.VERTEX, 1)
            row["x"], row["y"], row["z"] = np.array([v.X(), v.Y(), v.Z()]) - np.asarray(offset_mm)
            row["t"], row["num_particles"], row["interaction_id"] = v.T(), vtx.Particles.size(), iid
            verts.append(row)
        events.append(StepEvent(
            event_id=int(ev.EventId), steps=steps, particles=particles,
            primaries=np.array(prims, dtype=schema.PRIMARY) if prims else schema.empty(schema.PRIMARY),
            vertices=np.concatenate(verts) if verts else schema.empty(schema.VERTEX)))
    return write_events(out_path, events, run_id=run_id,
                        attrs={"source": str(Path(root_path).name), "detector": detector,
                               "offset_mm": np.asarray(offset_mm, dtype=np.float64)})
