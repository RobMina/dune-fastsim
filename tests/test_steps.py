import sys

import h5py
import numpy as np
import pytest

from dunefastsim.steps import StepEvent, check, schema, write_events
from dunefastsim.steps.edepsim import segments_to_steps, trajectories_to_particles


def _muon_event(offset=(0.0, 0.0, 0.0)):
    """A +y muon (track 0) and a delta ray (track 1) crossing the detector centre, edep-sim style segments."""
    n = 50
    y = np.linspace(-500.0, 500.0, n + 1)
    start = np.column_stack([np.full(n, 10.0), y[:-1], np.zeros(n), y[:-1] / 250.0]) + [*offset, 0.0]
    stop = np.column_stack([np.full(n, 10.0), y[1:], np.zeros(n), y[1:] / 250.0]) + [*offset, 0.0]
    d_start = np.array([[10.0, 0.0, 0.0, 0.0]]) + [*offset, 0.0]
    d_stop = np.array([[13.0, 0.0, 4.0, 0.01]]) + [*offset, 0.0]
    steps = segments_to_steps(
        np.vstack([d_start, start]), np.vstack([d_stop, stop]),
        de=np.r_[0.5, np.full(n, 4.2)], track_length=np.r_[0.0, np.full(n, 20.0)],
        track_id=np.r_[1, np.zeros(n, int)], primary_id=np.zeros(n + 1, int), pdg_of_track=[13, 11], offset_mm=offset)
    particles = trajectories_to_particles(
        track_id=[1, 0], parent_id=[0, -1], pdg=[11, 13], mass=[0.511, 105.658],
        first=[d_start[0], start[0]], last=[d_stop[0], stop[-1]], p_first=[[1.0, 0, 1.0], [0, 261.6, 0]],
        offset_mm=offset)
    prim = np.array([(0, 261.6, 0, 176.4, 0, 0, 13, 105.658, "mu-")], dtype=schema.PRIMARY)
    return StepEvent(event_id=7, steps=steps, particles=particles, primaries=prim)


def test_segment_conventions():
    ev = _muon_event()
    s = ev.steps
    mu = s[s["track_id"] == 0]
    assert np.allclose(mu["theta"], np.pi / 2) and np.allclose(mu["phi"], np.pi / 2)
    assert np.allclose(mu["dx"], 20.0) and np.allclose(mu["y"][0], -490.0) and np.allclose(mu["t"][0], -1.96)
    delta = s[s["track_id"] == 1]
    assert np.isclose(delta["dx"][0], 5.0) and delta["pdg"][0] == 11  # chord when track length is 0
    assert list(ev.particles["track_id"]) == [0, 1] and list(ev.particles["ancestor_track_id"]) == [0, 0]


def test_offset_moves_to_detector_frame():
    a, b = _muon_event(), _muon_event(offset=(0.0, -668.713, 6657.559))
    for k in ("x", "y", "z"):
        assert np.allclose(a.steps[k], b.steps[k], atol=1e-3)


def test_write_and_check(tmp_path):
    path = write_events(tmp_path / "ev.h5", [_muon_event(), _muon_event()], run_id=3)
    summary = check(path)
    assert summary["events"] == 2 and summary["steps"] == 102
    assert np.isclose(summary["de_MeV"], 2 * (0.5 + 50 * 4.2), rtol=1e-6)
    with h5py.File(path) as f:
        ass = f["ass/particle_pstep_lar_vol"][0]
        steps = f["pstep/lar_vol"][0]
        assert (steps["track_id"][ass["start"][0]:ass["end"][0]] == 0).all() and ass["end"][1] - ass["start"][1] == 1
        assert f["event/geant4"][1]["num_steps"] == 51 and f["event/geant4"][0]["run_id"] == 3


def test_jaxtpc_loader_reads_our_file(tmp_path, jaxtpc_root, monkeypatch):
    """The real check: JAXTPC's own loader extracts our steps unchanged (needs JAXTPC_ROOT)."""
    monkeypatch.syspath_prepend(str(jaxtpc_root))
    pytest.importorskip("jax")
    from tools.loader import load_particle_step_data

    path = write_events(tmp_path / "ev.h5", [_muon_event()])
    raw = load_particle_step_data(str(path), event_idx=0)
    assert raw["positions_mm"].shape == (51, 3)
    assert np.isclose(raw["de"].sum(), 0.5 + 50 * 4.2, rtol=1e-6)
    assert set(np.unique(raw["interaction_ids"])) == {0}
