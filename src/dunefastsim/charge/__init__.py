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


def _tools():
    """JAXTPC's tools.{geometry, loader, simulation} modules from the checkout."""
    _import()
    from tools import geometry, loader, simulation

    return geometry, loader, simulation


def jaxtpc_precision():
    """Context in which JAXTPC (and BdNMC's port in float32) runs: 64-bit mode off. JAXTPC is float32 throughout (its
    kernel tables fail with jax_enable_x64), while option B may run float64 and hepsimkernels switches 64-bit mode on
    when imported. ``jax.experimental.disable_x64`` up to JAX 0.6-0.10, ``jax.enable_x64(False)`` from 0.11 (which
    removed the former); raises if neither exists rather than silently leaving 64-bit mode on."""
    import jax
    import jax.experimental

    f = getattr(jax.experimental, "disable_x64", None)
    if f is not None:
        return f()
    g = getattr(jax, "enable_x64", None)
    if callable(g):
        return g(False)
    raise RuntimeError(f"JAX {jax.__version__}: no context to turn 64-bit mode off (disable_x64 / enable_x64)")


def make_simulator(config_yaml, total_pad: int = 50_000, **kwargs):
    """A JAXTPC DetectorSimulator for a detector YAML (e.g. ``dunefastsim ndlar-config``'s ND-LAr volumes). Pixel
    readout needs track hits, so they are on by default; other DetectorSimulator options pass through."""
    geometry, _, simulation = _tools()
    kwargs.setdefault("include_track_hits", True)
    kwargs.setdefault("response_chunk_size", min(total_pad, 50_000))  # JAXTPC: total_pad must be a multiple of it
    with jaxtpc_precision():
        return simulation.DetectorSimulator(geometry.generate_detector(str(config_yaml)), total_pad=total_pad,
                                            **kwargs)


def light(sim, deposits):
    """JAXTPC charge and scintillation per deposit (no readout response), in JAXTPC's precision."""
    with jaxtpc_precision():
        return sim.process_event_light(deposits)


def pixels(sim, deposits, key=None):
    """JAXTPC's full event (response signals, track hits, filled deposits), in JAXTPC's precision."""
    import jax

    with jaxtpc_precision():
        return sim.process_event(deposits, key=key if key is not None else jax.random.PRNGKey(0))


def option_b_deposits(ev: dict, sim_config, interaction_id: int = 0, group_size: int = 5,
                      gap_threshold_mm: float = 5.0):
    """JAXTPC DepositData from one option B event (``transport.cascade.split_events`` output: detector-frame mm,
    MeV, ns). Directions become the step record's (theta, phi) = (acos d_z, atan2(d_y, d_x)); times ns -> us; every
    deposit's root track is the primary (0). JAXTPC drops deposits outside its volumes."""
    _, loader, _ = _tools()
    import numpy as np

    pos = np.asarray(ev["positions_mm"], np.float32)
    d = np.asarray(ev["dir"], np.float64)
    n = len(pos)
    with jaxtpc_precision():
        return loader.build_deposit_data(
            pos, np.asarray(ev["de"], np.float32), np.asarray(ev["dx"], np.float32), sim_config,
            theta=np.arccos(np.clip(d[:, 2], -1.0, 1.0)).astype(np.float32),
            phi=np.arctan2(d[:, 1], d[:, 0]).astype(np.float32),
            track_ids=np.asarray(ev["tid"], np.int32), t0_us=np.asarray(ev["t"], np.float32) / 1000.0,
            interaction_ids=np.full(n, interaction_id, np.int32), root_track_ids=np.zeros(n, np.int32),
            pdg=np.asarray(ev["pdg"], np.int32), group_size=group_size, gap_threshold_mm=gap_threshold_mm)


def step_file_deposits(path, event_idx: int, sim_config, group_size: int = 5, gap_threshold_mm: float = 5.0):
    """JAXTPC DepositData from one event of a step-record file, as JAXTPC's production/run_batch.py loads it."""
    _, loader, _ = _tools()
    import numpy as np

    with loader.ParticleStepExtractor(str(path)) as ex:
        sd = ex.extract_step_arrays(event_idx)
        pdata = getattr(ex, "_last_particle_data", None) or {}
        iid = loader.compute_interaction_ids(ex.file, event_idx, root_track_ids=sd.get("root_track_id"),
                                             particle_track_ids=pdata.get("track_id"),
                                             particle_parent_ids=pdata.get("parent_track_id"))
    pos = np.asarray(sd.get("position", np.empty((0, 3))), np.float32)
    n = len(pos)
    get = lambda k, dflt, dt: np.asarray(sd.get(k, dflt), dt)  # noqa: E731
    with jaxtpc_precision():
        return loader.build_deposit_data(
            pos, get("de", np.zeros(n), np.float32), get("dx", np.zeros(n), np.float32), sim_config,
            theta=get("theta", np.zeros(n), np.float32), phi=get("phi", np.zeros(n), np.float32),
            track_ids=get("track_id", np.ones(n), np.int32), t0_us=get("t", np.zeros(n), np.float32) / 1000.0,
            interaction_ids=iid, root_track_ids=get("root_track_id", np.zeros(n), np.int32),
            pdg=get("pdg", np.zeros(n), np.int32), group_size=group_size, gap_threshold_mm=gap_threshold_mm)
