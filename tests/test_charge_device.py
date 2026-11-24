"""On-device volume split (charge.device) against JAXTPC's host-side build_deposit_data. Needs JAXTPC_ROOT; run on
UVA. Uses JAXTPC's own ND-LAr config (70 pixel volumes)."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from dunefastsim import charge as ch  # noqa: E402
from dunefastsim.charge import device as dv  # noqa: E402


@pytest.fixture(scope="module")
def ndlar(request):
    import os
    from pathlib import Path

    r = os.environ.get("JAXTPC_ROOT")
    if not r or not (Path(r) / "tools" / "loader.py").is_file():
        pytest.skip("JAXTPC_ROOT not set")
    geometry, loader, _ = ch._tools()
    from tools.config import create_sim_config

    with ch.jaxtpc_precision():
        cfg = create_sim_config(geometry.generate_detector(str(ch.ndlar_template())), total_pad=5000,
                                response_chunk_size=5000)
    return cfg, loader


def _tracks(rng, cfg, n_tracks=60):
    """Random-walk tracks inside and around the volumes, interleaved as in a wavefront buffer (each track's own
    order kept), with occasional jumps above the 5 mm grouping gap."""
    r = np.array([v.ranges_cm for v in cfg.volumes]) * 10.0  # (V, 3, 2) mm
    lo, hi = r[:, :, 0].min(0), r[:, :, 1].max(0)
    pos, tid, key = [], [], []
    for t in range(n_tracks):
        n = int(rng.integers(5, 200))
        start = lo + (hi - lo) * rng.uniform(0.02, 0.98, 3)
        steps = rng.normal(0, 1.5, (n, 3))
        steps[rng.uniform(size=n) < 0.05] *= 8.0
        pos.append(start + np.cumsum(steps, 0))
        tid.append(np.full(n, t))
        key.append(np.sort(rng.uniform(size=n)))
    pos, tid, key = np.concatenate(pos), np.concatenate(tid), np.concatenate(key)
    o = np.argsort(key, kind="stable")
    n = len(o)
    f = {"positions_mm": pos[o].astype(np.float32), "track_ids": tid[o].astype(np.int32),
         "de": rng.uniform(0.01, 2.0, n).astype(np.float32), "dx": rng.uniform(0.05, 10.0, n).astype(np.float32),
         "theta": rng.uniform(0, np.pi, n).astype(np.float32), "phi": rng.uniform(-np.pi, np.pi, n).astype(np.float32),
         "t0_us": rng.uniform(0, 1, n).astype(np.float32), "interaction_ids": rng.integers(0, 3, n).astype(np.int16),
         "root_track_ids": rng.integers(0, 4, n).astype(np.int32), "pdg": rng.choice([11, -11, 22], n).astype(np.int32)}
    return f


def _split(cfg, f, total_pad, valid=None):
    vt = dv.volume_table(cfg)
    n = len(f["de"])
    valid = np.ones(n, bool) if valid is None else valid
    with ch.jaxtpc_precision():
        fn = jax.jit(lambda *a: dv.split_volumes(vt, total_pad, *a))
        return fn(*(jnp.asarray(f[k]) for k in ("positions_mm", "de", "dx", "theta", "phi", "track_ids", "t0_us",
                                                 "interaction_ids", "root_track_ids", "pdg")), jnp.asarray(valid))


def test_split_matches_build_deposit_data(ndlar):
    cfg, loader = ndlar
    f = _tracks(np.random.default_rng(1), cfg)
    stacked, info = _split(cfg, f, cfg.total_pad)
    with ch.jaxtpc_precision():
        host = loader.build_deposit_data(f["positions_mm"], f["de"], f["dx"], cfg, theta=f["theta"], phi=f["phi"],
                                         track_ids=f["track_ids"], t0_us=f["t0_us"],
                                         interaction_ids=f["interaction_ids"], root_track_ids=f["root_track_ids"],
                                         pdg=f["pdg"])
    assert int(info.multi) == 0
    assert float(info.overflow_mev) == 0.0
    n_host = np.array([int(v.n_actual) for v in host.volumes])
    assert n_host.sum() > 1000 and (n_host > 0).sum() > 10  # the tracks cross many volumes
    np.testing.assert_array_equal(np.asarray(stacked.n_actual), n_host)
    assert int(info.n_in_volumes) == n_host.sum()
    assert np.isclose(float(info.outside_mev), f["de"].sum() - sum(float(np.asarray(v.de).sum()) for v in host.volumes),
                      rtol=1e-4)
    for v, hv in enumerate(host.volumes):
        for field in hv._fields:
            if field == "n_actual":
                continue
            a, b = np.asarray(getattr(stacked, field)[v]), np.asarray(getattr(hv, field))
            assert a.dtype == b.dtype, (field, a.dtype, b.dtype)
            np.testing.assert_array_equal(a, b, err_msg=f"volume {v} field {field}")
        g2t = np.asarray(host.group_to_track[v])
        np.testing.assert_array_equal(np.asarray(info.group_to_track[v])[:len(g2t)], g2t)
        np.testing.assert_array_equal(np.asarray(info.original_index[v])[:n_host[v]], host.original_indices[v])


def test_overflow_is_reported_not_raised(ndlar):
    cfg, _ = ndlar
    f = _tracks(np.random.default_rng(2), cfg)
    big, _ = _split(cfg, f, cfg.total_pad)
    small, info = _split(cfg, f, 16)
    n = np.asarray(big.n_actual)
    assert (n > 16).any()
    np.testing.assert_array_equal(np.asarray(small.n_actual), np.minimum(n, 16))
    dropped = sum(float(np.asarray(big.de[v])[16:n[v]].sum()) for v in range(len(n)))
    assert np.isclose(float(info.overflow_mev), dropped, rtol=1e-5)
    np.testing.assert_array_equal(np.asarray(small.de), np.asarray(big.de)[:, :16])


def test_invalid_deposits_ignored(ndlar):
    cfg, _ = ndlar
    f = _tracks(np.random.default_rng(3), cfg)
    valid = np.random.default_rng(4).uniform(size=len(f["de"])) < 0.5
    a, _ = _split(cfg, f, cfg.total_pad, valid)
    g = {k: v[valid] for k, v in f.items()}
    b, _ = _split(cfg, g, cfg.total_pad)
    for field in ("positions_mm", "de", "dx", "track_ids", "n_actual"):
        np.testing.assert_array_equal(np.asarray(getattr(a, field)), np.asarray(getattr(b, field)))


@pytest.fixture(scope="module")
def light_sim(ndlar):
    cfg, _ = ndlar
    return ch.make_simulator(ch.ndlar_template(), total_pad=cfg.total_pad, include_track_hits=False)


def test_option_b_event_matches_host_path_and_light(ndlar, light_sim):
    """Option B showers: the device split of each primary equals the host path (split_events +
    option_b_deposits), and JAXTPC's light calculator gives the same charge from both."""
    pytest.importorskip("hepsimkernels")
    from dunefastsim.transport import cascade as cs
    from dunefastsim.transport import emphysics as em

    cfg, _ = ndlar
    v0 = np.array(cfg.volumes[20].ranges_cm) * 10.0
    vtx = np.stack([v0.mean(1) + [0, 0, -100.0], v0.mean(1) + [0, 50.0, -100.0]])
    tb = cs.device_tables(em.build_tables(nt=121, nk=121, nc=65, nx=65), jnp.float32)
    dep = cs.make_showers_fn(tb, cs.Config(n_cap=4096, m_cap=1 << 16, fill=0.5))(
        jax.random.PRNGKey(5), np.array([150.0, 80.0]), vtx, np.array([[0, 0, 1.0], [0.3, 0, 0.954]]))
    evs, _ = cs.split_events(dep, 2)
    vt = dv.volume_table(cfg)
    split = dv.make_event_split(vt, cfg.total_pad)
    for b in range(2):
        stacked, info = split(dep, b)
        host = ch.option_b_deposits(evs[b], cfg)
        n_host = np.array([int(v.n_actual) for v in host.volumes])
        np.testing.assert_array_equal(np.asarray(stacked.n_actual), n_host)
        assert n_host.sum() > 50
        for v in np.nonzero(n_host)[0]:
            for field in ("positions_mm", "de", "dx", "track_ids", "group_ids", "pdg", "t0_us"):
                np.testing.assert_allclose(np.asarray(getattr(stacked, field)[v]),
                                           np.asarray(getattr(host.volumes[v], field)), rtol=1e-6, atol=1e-6,
                                           err_msg=f"event {b} volume {v} {field}")
            for field in ("theta", "phi"):  # float32 arccos/atan2 on the device vs float64 on the host
                np.testing.assert_allclose(np.asarray(getattr(stacked, field)[v]),
                                           np.asarray(getattr(host.volumes[v], field)), atol=2e-3)
        with ch.jaxtpc_precision():
            q_dev, l_dev = dv.light_stacked(light_sim, stacked)
            lit = ch.light(light_sim, host)
        q_host = sum(float(np.asarray(v.charge).sum()) for v in lit.volumes)
        assert np.isclose(float(q_dev.sum()), q_host, rtol=1e-3), (float(q_dev.sum()), q_host)


def test_charge_is_differentiable_through_the_split(ndlar, light_sim):
    """Gradients of JAXTPC's total charge reach the deposit energies and positions through the device split, and
    agree with finite differences."""
    cfg, _ = ndlar
    f = _tracks(np.random.default_rng(6), cfg, n_tracks=20)
    vt = dv.volume_table(cfg)
    args = {k: jnp.asarray(f[k]) for k in f}
    valid = jnp.ones(len(f["de"]), bool)

    def total_q(scale, shift_x):
        pos = args["positions_mm"] + jnp.array([1.0, 0.0, 0.0], jnp.float32) * shift_x
        st, _ = dv.split_volumes(vt, cfg.total_pad, pos, args["de"] * scale, args["dx"], args["theta"],
                                 args["phi"], args["track_ids"], args["t0_us"], args["interaction_ids"],
                                 args["root_track_ids"], args["pdg"], valid)
        q, _ = dv.light_stacked(light_sim, st)
        return q.sum()

    with ch.jaxtpc_precision():
        g = jax.jit(jax.grad(total_q, argnums=(0, 1)))(jnp.float32(1.0), jnp.float32(0.0))
        h = 0.01
        q = jax.jit(total_q)
        fd_s = (q(jnp.float32(1 + h), jnp.float32(0.0)) - q(jnp.float32(1 - h), jnp.float32(0.0))) / (2 * h)
        q0 = q(jnp.float32(1.0), jnp.float32(0.0))
    gs, gx = float(g[0]), float(g[1])
    assert np.isfinite(gs) and np.isfinite(gx)
    assert 0 < gs < float(q0)  # more energy, more charge, sublinear through recombination
    assert np.isclose(gs, float(fd_s), rtol=0.02), (gs, float(fd_s))
