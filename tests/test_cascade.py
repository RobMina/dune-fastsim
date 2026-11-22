"""Option B (B0): physics anchors and cascade bookkeeping. Needs hep-simkernels (the [transport] extra); run on UVA."""

import numpy as np
import pytest

pytest.importorskip("hepsimkernels")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from dunefastsim.transport import cascade as cs  # noqa: E402
from dunefastsim.transport import emphysics as em  # noqa: E402


@pytest.fixture(scope="module")
def host():
    return em.build_tables(nt=121, nk=121, nc=65, nx=65)


def test_radiation_length():
    assert np.isclose(em.radiation_length_g_cm2(), 19.55, rtol=5e-3)  # PDG argon 19.55 g/cm^2


def test_klein_nishina_1mev():
    assert np.isclose(em.klein_nishina_xsec_per_electron(1.0), 0.2112e-22, rtol=2e-3)  # 0.2112 b


def test_pair_high_energy_limit(host):
    """Pair cross section times X0 tends to 7/9 (minus a small electron-field term)."""
    sp = np.interp(np.log(1e4), np.log(host.k_nodes), host.sig_pair)
    assert 0.74 < sp * host.x0_mm < 0.80


def test_photon_mean_free_path_10gev(host):
    s = np.interp(np.log(1e4), np.log(host.k_nodes), host.sig_pair + host.sig_compton)
    assert np.isclose(1.0 / s / host.x0_mm, 9.0 / 7.0, rtol=0.05)


def test_compton_dominates_low_energy(host):
    i = np.searchsorted(host.k_nodes, 2.0)
    assert host.sig_compton[i] > 5 * host.sig_pair[i]


def test_tsai_matches_seltzer_berger_at_1gev():
    from hepsimkernels import ebrem
    mat = em.lar_material()
    sb_sig = ebrem.xsec_per_mm(mat, [1000.0], 0.1)[0]
    ts_sig = em.tsai_xsec_per_mm([1000.0], 0.1)[0]
    sb_dedx = ebrem.dedx_restricted(mat, [1000.0], 1000.0)[0]
    ts_dedx = em.tsai_dedx_restricted([1000.0], 1000.0)[0]
    assert np.isclose(ts_sig, sb_sig, rtol=0.1), (ts_sig, sb_sig)
    assert np.isclose(ts_dedx, sb_dedx, rtol=0.1), (ts_dedx, sb_dedx)


def test_inverse_cdf_tables_monotone(host):
    for q in (host.brem_q, host.delta_q):
        assert np.all(np.diff(q, axis=-1) >= -1e-12) and q.min() >= -1e-12 and q.max() <= 1 + 1e-12
    assert np.all(np.diff(host.compton_q, axis=-1) >= -1e-12)


@pytest.mark.parametrize("dtype,tol", [(jnp.float64, 1e-8), (jnp.float32, 2e-4)])
@pytest.mark.parametrize("energy", [50.0, 1000.0])
def test_energy_conserved(host, dtype, tol, energy):
    tb = cs.device_tables(host, dtype)
    cfg = cs.Config(n_cap=4096, m_cap=1 << 16)
    dep = cs.to_numpy(cs.make_shower_fn(tb, cfg)(jax.random.PRNGKey(1), energy, jnp.zeros(3), jnp.array([0.0, 0, 1])))
    assert dep["lost_mev"] == 0.0 and dep["peak"] < cfg.n_cap
    assert np.isclose(dep["de"].sum(dtype=np.float64), energy, rtol=tol), dep["de"].sum()
    assert np.all(dep["de"] >= 0) and np.all(np.isfinite(dep["positions_mm"]))


def test_shower_shape_1gev(host):
    """Energy-weighted depth of a few 1 GeV showers: 3-5 X0 (GFLASH / Geant4 give about 3.6-4)."""
    tb = cs.device_tables(host)
    f = cs.make_shower_fn(tb, cs.Config(n_cap=4096, m_cap=1 << 16))
    depth = []
    for i in range(8):
        dep = cs.to_numpy(f(jax.random.PRNGKey(10 + i), 1000.0, jnp.zeros(3), jnp.array([0.0, 0, 1])))
        depth.append(np.average(dep["positions_mm"][:, 2], weights=dep["de"]) / host.x0_mm)
    assert 3.0 < np.mean(depth) < 5.0, depth


def test_photon_and_positron_primaries(host):
    tb = cs.device_tables(host)
    cfg = cs.Config(n_cap=4096, m_cap=1 << 16)
    for prim, extra in ((cs.GAMMA, 0.0), (cs.E_PLUS, 2 * em.ME)):
        dep = cs.to_numpy(jax.jit(lambda k: cs.shower(tb, cfg, k, 200.0, jnp.zeros(3), jnp.array([1.0, 0, 0]),
                                                          primary=prim))(jax.random.PRNGKey(3)))
        assert np.isclose(dep["de"].sum(), 200.0 + extra, rtol=1e-8)


@pytest.mark.parametrize("q", [1e-9, 1e-6, 1e-3, 0.05, 0.5, 0.95])
def test_wentzel_screening_matches_mean(q):
    """The table's screening reproduces the target mean (closed form), and the sampler follows it (moderate q: a
    uniform grid cannot resolve the heavy tail at tiny q)."""
    lnq, lna = em.wentzel_table()
    a = np.exp(np.interp(np.log(q), lnq, lna))
    assert np.isclose(em.wentzel_mean(a), q, rtol=1e-3)
    if q >= 0.05:
        u = (np.arange(400000) + 0.5) / 400000
        assert np.isclose((2 * a * u / (1 - u + a)).mean(), q, rtol=0.01)


def test_transport_mfp_vs_highland(host):
    """Small-angle check: 2 s / lambda_1 against the space-angle variance 2 theta0^2 of Highland, 10 MeV, 1 mm."""
    T, s = 10.0, 1.0
    lam = np.exp(np.interp(np.log(T), np.log(host.t_nodes), np.log(host.lam1[0])))
    pc = np.sqrt(T * (T + 2 * em.ME))
    beta = pc / (T + em.ME)
    xs = s / host.x0_mm
    th0 = 13.6 / (beta * pc) * np.sqrt(xs) * (1 + 0.038 * np.log(xs / beta ** 2))
    assert 0.5 < (2 * s / lam) / (2 * th0 ** 2) < 2.0


def _dedx_q(dep):
    dedx = 10.0 * dep["de"] / dep["dx"]
    o = np.argsort(dedx)
    c = np.cumsum(dep["de"][o]) / dep["de"].sum()
    return [dedx[o][np.searchsorted(c, p)] for p in (0.1, 0.5, 0.9)]


def test_fluctuations_widen_dedx(host):
    """Energy-weighted dE/dx of a 1 GeV shower: B1 (Urban fluctuations) wider than B0 (mean loss)."""
    tb = cs.device_tables(host)
    b0 = cs.Config(n_cap=4096, m_cap=1 << 16, fluctuate=False, msc="highland", tsai_angles=False)
    b1 = cs.Config(n_cap=4096, m_cap=1 << 16)
    q0 = _dedx_q(cs.to_numpy(cs.make_shower_fn(tb, b0)(jax.random.PRNGKey(5), 1000.0, jnp.zeros(3),
                                                       jnp.array([0.0, 0, 1]))))
    dep1 = cs.to_numpy(cs.make_shower_fn(tb, b1)(jax.random.PRNGKey(5), 1000.0, jnp.zeros(3), jnp.array([0.0, 0, 1])))
    q1 = _dedx_q(dep1)
    assert q1[2] - q1[0] > 1.3 * (q0[2] - q0[0]), (q0, q1)
    assert np.isclose(dep1["de"].sum(), 1000.0, rtol=1e-8)


def test_b0_config_still_runs(host):
    tb = cs.device_tables(host)
    cfg = cs.Config(n_cap=4096, m_cap=1 << 16, fluctuate=False, msc="highland", tsai_angles=False)
    dep = cs.to_numpy(cs.make_shower_fn(tb, cfg)(jax.random.PRNGKey(2), 300.0, jnp.zeros(3), jnp.array([0.0, 1, 0])))
    assert np.isclose(dep["de"].sum(), 300.0, rtol=1e-8)


def test_many_showers_one_buffer(host):
    """B2: primaries of mixed energies (and zero-energy padding) share one buffer; a small fill target makes them
    queue. Each primary's deposits add up to its energy, nothing is lost, and each primary's track renumbers to 0."""
    tb = cs.device_tables(host)
    cfg = cs.Config(n_cap=4096, m_cap=1 << 18, fill=0.05)
    e = np.array([50.0, 300.0, 0.0, 1000.0, 120.0, 0.0, 2000.0, 75.0])
    v = np.arange(24.0).reshape(8, 3)
    d = np.tile([0.0, 0.6, 0.8], (8, 1))
    out = cs.make_showers_fn(tb, cfg)(jax.random.PRNGKey(4), e, v, d)
    evs, info = cs.split_events(out, len(e))
    assert info["deposits"] < cfg.m_cap and info["peak"] < cfg.n_cap
    for ei, ev in zip(e, evs):
        assert ev["lost_mev"] == 0.0
        assert np.isclose(ev["de"].sum(), ei, rtol=1e-8, atol=1e-12), (ei, ev["de"].sum())
        if ei > 0:
            assert ev["tid"].min() == 0 and np.any(ev["tid"] == 0)
    # each shower develops from its own vertex along its direction: 1 GeV centroid 3-5 X0 deep, near the axis
    ev = evs[3]
    c = np.average(ev["positions_mm"], axis=0, weights=ev["de"]) - v[3]
    depth = c @ d[3]
    assert 3.0 < depth / host.x0_mm < 5.0 and np.linalg.norm(c - depth * d[3]) < 30.0, c


def test_single_shower_is_batch_of_one(host):
    tb = cs.device_tables(host)
    cfg = cs.Config(n_cap=4096, m_cap=1 << 16)
    k = jax.random.PRNGKey(8)
    a = cs.to_numpy(cs.make_shower_fn(tb, cfg)(k, 200.0, jnp.zeros(3), jnp.array([0.0, 0, 1])))
    evs, _ = cs.split_events(cs.make_showers_fn(tb, cfg)(k, np.array([200.0]), np.zeros((1, 3)),
                                                         np.array([[0.0, 0, 1]])), 1)
    assert np.array_equal(a["de"], evs[0]["de"]) and np.array_equal(a["tid"], evs[0]["tid"])
