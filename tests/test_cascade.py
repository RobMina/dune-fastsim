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
    total = dep["de"].sum(dtype=np.float64) + dep["nuclear_mev"]  # invisible photonuclear energy included
    assert np.isclose(total, energy, rtol=tol), total
    assert np.all(dep["de"] >= 0) and np.all(np.isfinite(dep["positions_mm"]))


def test_shower_shape_1gev(host):
    """Energy-weighted depth of a few 1 GeV showers: 3-5 X0 (GFLASH / Geant4 give about 3.6-4)."""
    tb = cs.device_tables(host)
    # 32 showers in one buffer: one shower's depth scatters by ~1.5 X0, so 8 showers left +-0.5 X0 of noise
    nb = 32
    f = cs.make_showers_fn(tb, cs.Config(n_cap=8192, m_cap=1 << 20, fill=0.5))
    out = f(jax.random.PRNGKey(10), np.full(nb, 1000.0), np.zeros((nb, 3)), np.tile([0.0, 0.0, 1.0], (nb, 1)))
    evs, info = cs.split_events(out, nb)
    assert info["deposits"] < (1 << 20) and all(ev["lost_mev"] == 0.0 for ev in evs)
    depth = [np.average(ev["positions_mm"][:, 2], weights=ev["de"]) / host.x0_mm for ev in evs]
    assert 3.0 < np.mean(depth) < 5.0, depth


def test_photon_and_positron_primaries(host):
    tb = cs.device_tables(host)
    cfg = cs.Config(n_cap=4096, m_cap=1 << 16)
    for prim, extra in ((cs.GAMMA, 0.0), (cs.E_PLUS, 2 * em.ME)):
        dep = cs.to_numpy(jax.jit(lambda k: cs.shower(tb, cfg, k, 200.0, jnp.zeros(3), jnp.array([1.0, 0, 0]),
                                                          primary=prim))(jax.random.PRNGKey(3)))
        assert np.isclose(dep["de"].sum() + dep["nuclear_mev"], 200.0 + extra, rtol=1e-8)


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
    assert np.isclose(dep1["de"].sum() + dep1["nuclear_mev"], 1000.0, rtol=1e-8)


def test_b0_config_still_runs(host):
    tb = cs.device_tables(host)
    cfg = cs.Config(n_cap=4096, m_cap=1 << 16, fluctuate=False, msc="highland", tsai_angles=False)
    dep = cs.to_numpy(cs.make_shower_fn(tb, cfg)(jax.random.PRNGKey(2), 300.0, jnp.zeros(3), jnp.array([0.0, 1, 0])))
    assert np.isclose(dep["de"].sum() + dep["nuclear_mev"], 300.0, rtol=1e-8)


def test_many_showers_one_buffer(host):
    """B2: primaries of mixed energies (and zero-energy padding) share one buffer; a small fill target makes them
    queue. Each primary's deposits add up to its energy, nothing is lost, and each primary's track renumbers to 0."""
    tb = cs.device_tables(host)
    cfg = cs.Config(n_cap=4096, m_cap=1 << 18, fill=0.05)
    e = np.array([50.0, 300.0, 0.0, 1000.0, 120.0, 0.0, 2000.0, 75.0])
    v = np.column_stack([10000.0 * np.arange(8), np.zeros(8), np.zeros(8)])  # vertices 10 m apart
    d = np.tile([0.0, 0.6, 0.8], (8, 1))
    out = cs.make_showers_fn(tb, cfg)(jax.random.PRNGKey(4), e, v, d)
    evs, info = cs.split_events(out, len(e))
    assert info["deposits"] < cfg.m_cap and info["peak"] < cfg.n_cap
    for ei, ev in zip(e, evs):
        assert ev["lost_mev"] == 0.0
        assert np.isclose(ev["de"].sum() + ev["nuclear_mev"], ei, rtol=1e-8, atol=1e-12), (ei, ev["de"].sum())
        if ei > 0:
            assert ev["tid"].min() == 0 and np.any(ev["tid"] == 0)
    # each primary's deposits belong to its own shower: the primary's first step starts at its vertex, every deposit
    # lies within 5 m of it (the next vertex is 10 m away), and the shower goes along its direction (one 1 GeV
    # shower's depth scatters by ~1.5 X0, so only a wide window)
    for k, (ei, ev) in enumerate(zip(e, evs)):
        if ei == 0:
            continue
        p0 = ev["positions_mm"][ev["tid"] == 0][np.argmin(ev["t"][ev["tid"] == 0])]
        assert np.linalg.norm(p0 - v[k]) <= 0.5 * cfg.s_max + 1e-6, (k, p0)
        assert np.linalg.norm(ev["positions_mm"] - v[k], axis=1).max() < 5000.0, k
    c = np.average(evs[3]["positions_mm"], axis=0, weights=evs[3]["de"]) - v[3]
    depth = c @ d[3]
    assert 2.0 < depth / host.x0_mm < 8.0 and np.linalg.norm(c - depth * d[3]) < 50.0, c


def test_single_shower_is_batch_of_one(host):
    tb = cs.device_tables(host)
    cfg = cs.Config(n_cap=4096, m_cap=1 << 16)
    k = jax.random.PRNGKey(8)
    a = cs.to_numpy(cs.make_shower_fn(tb, cfg)(k, 200.0, jnp.zeros(3), jnp.array([0.0, 0, 1])))
    evs, _ = cs.split_events(cs.make_showers_fn(tb, cfg)(k, np.array([200.0]), np.zeros((1, 3)),
                                                         np.array([[0.0, 0, 1]])), 1)
    assert np.array_equal(a["de"], evs[0]["de"]) and np.array_equal(a["tid"], evs[0]["tid"])


def test_photoelectric_argon():
    """Livermore (EPICS2014) argon photoelectric cross section: NIST XCOM magnitudes, smooth at the joins, dominant
    below ~60 keV and negligible against Compton at 1 MeV."""
    per_cm2_g = em.N_A / em.A_AR * 1e-24  # barn per atom -> cm^2/g
    s10 = float(em.photoelectric_xsec_per_atom_barn(0.010)) * per_cm2_g
    s50 = float(em.photoelectric_xsec_per_atom_barn(0.050)) * per_cm2_g
    assert 55.0 < s10 < 66.0 and 0.40 < s50 < 0.58, (s10, s50)  # XCOM: ~61 and ~0.48 cm2/g
    for e in (em.PE_AR_LOW[0], em.PE_AR_HIGH[0]):
        lo, hi = em.photoelectric_xsec_per_atom_barn(np.array([e * 0.999, e * 1.001]))
        assert abs(hi / lo - 1) < 0.03, (e, lo, hi)


def test_photon_tables_photoelectric(host):
    k = host.k_nodes
    i30, i1 = np.searchsorted(k, 0.03), np.searchsorted(k, 1.0)
    assert host.sig_photo[i30] > host.sig_compton[i30]
    assert host.sig_photo[i1] < 1e-2 * host.sig_compton[i1]  # argon at 1 MeV: ~0.1-0.3 %
    assert k[0] <= em.KCUT_REFERENCE  # the tables reach the reference's photon cut


def test_low_energy_photon_is_absorbed_with_its_energy(host):
    """A 60 keV photon: deposited in full (photoelectric, or Compton and then photoelectric), centimetres away."""
    tb = cs.device_tables(host)
    cfg = cs.Config(n_cap=512, m_cap=1 << 14, k_track=0.004)  # tracked below the default 0.1 MeV cut
    f = jax.jit(lambda k, e, v, d: cs.showers(tb, cfg, k, e, v, d, primary=cs.GAMMA))
    nb = 200
    out = f(jax.random.PRNGKey(5), np.full(nb, 0.06), np.zeros((nb, 3)), np.tile([0.0, 0.0, 1.0], (nb, 1)))
    evs, _ = cs.split_events(out, nb)
    z = []
    for ev in evs:
        assert np.isclose(ev["de"].sum(), 0.06, rtol=1e-8) and ev["lost_mev"] == 0.0
        z.append(np.average(ev["positions_mm"][:, 2], weights=ev["de"]))
    # mean free path at 60 keV: 1 / (photoelectric + Compton) ~ 1.5 cm in LAr; absorption takes longer
    assert 5.0 < np.mean(z) < 60.0, np.mean(z)


def _emsc_mixed_reference(T, s, n, positron, seed=3):
    """hepsimkernels.emsc's own mixed scheme (NumPy): soft deflection with u_c from C1 lambda_1, then a Poisson number
    of hard trials (sample_hard: Mott factor and nuclear form factor)."""
    from hepsimkernels import emsc

    mat = em.lar_material()
    rng = np.random.default_rng(seed)
    Ta = np.full(n, T)
    uc = emsc.u_cut(mat, Ta, 0.22, positron)
    d = np.tile([0.0, 0.0, 1.0], (n, 1))
    d = emsc.deflect(rng, d, emsc.sample_soft(rng, mat, Ta, s, uc, 0.22, positron))
    nh = rng.poisson(s * emsc.hard_xsec(mat, Ta, uc, 0.22, positron))
    for j in range(int(nh.max())):
        u, acc = emsc.sample_hard(rng, mat, Ta, uc, 0.22, positron)
        m = (nh > j) & acc
        d[m] = emsc.deflect(rng, d[m], u[m])
    return np.arccos(np.clip(d[:, 2], -1.0, 1.0))


@pytest.mark.parametrize("T,s", [(1.0, 0.4), (10.0, 5.0), (1000.0, 10.0)])
def test_mixed_msc_matches_emsc_and_highland_core(host, T, s):
    """Mixed scattering over one step against hepsimkernels.emsc's mixed sampler (median and 90 % angles; the mean
    of this heavy-tailed angle is dominated by rare hard collisions) and against Highland's Gaussian core (median
    near the Rayleigh median 1.177 theta0; the matched-mean Wentzel draw of v0.0.6 put it several times lower)."""
    tb = cs.device_tables(host)
    n = 200000
    d0 = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (n, 1))
    for positron in (False, True):
        out = cs._msc_mixed(tb, cs.Config(), jnp.full(n, positron), jnp.full(n, T), jnp.full(n, s), d0,
                            jax.random.PRNGKey(11))
        th = np.arccos(np.clip(np.asarray(out[:, 2]), -1.0, 1.0))
        ref = _emsc_mixed_reference(T, s, n, positron)
        for q, tol in ((0.5, 0.06), (0.9, 0.10)):
            a, b = np.quantile(th, q), np.quantile(ref, q)
            assert abs(a / b - 1) < tol, (positron, q, a, b)
        pc = np.sqrt(T * (T + 2 * em.ME))
        beta = pc / (T + em.ME)
        xs = s / host.x0_mm
        th0 = 13.6 / (beta * pc) * np.sqrt(xs) * (1 + 0.038 * np.log(xs / beta ** 2))
        assert 0.7 < np.median(th) / (1.1774 * th0) < 1.3, (positron, np.median(th), th0)


def test_annihilation_in_flight_kinematics():
    """Two photons carry T + 2 m_e and the positron's momentum; their energies span Geant4's range."""
    n = 50000
    key = jax.random.PRNGKey(2)
    T = jnp.asarray(np.geomspace(0.2, 2000.0, n))
    d = jnp.tile(jnp.array([0.6, 0.0, 0.8]), (n, 1))
    u = jax.random.uniform(key, (n, 2))
    iso = cs._isotropic(u[:, 0], u[:, 1])
    e1, d1, e2, d2 = cs._annihilate_in_flight(T, d, iso)
    e1, e2, d1, d2, T, d = (np.asarray(a) for a in (e1, e2, d1, d2, T, d))
    assert np.allclose(e1 + e2, T + 2 * em.ME, rtol=1e-12)
    p = np.sqrt(T * (T + 2 * em.ME))
    assert np.allclose(e1[:, None] * d1 + e2[:, None] * d2, p[:, None] * d, atol=1e-8 * (T[:, None] + 1))
    assert e1.min() > 0 and e2.min() > 0


def test_annihilation_in_flight_table(host):
    """Heitler cross section in the e+ row only; a 30 MeV positron slowing to 1 MeV in LAr annihilates in flight with
    a probability of order 10 % (integral of sigma_annih over the continuous dE/dx, which ignores discrete losses and
    so overestimates the path: a loose band)."""
    from hepsimkernels import eannihil

    assert np.all(host.sig_annih[0] == 0.0)
    i = np.searchsorted(host.t_nodes, 10.0)
    assert np.isclose(host.sig_annih[1, i], em.N_E_MM3 * eannihil.xsec_per_electron(host.t_nodes[i]), rtol=1e-12)
    m = (host.t_nodes > 1.0) & (host.t_nodes < 30.0)
    t = host.t_nodes[m]
    trap = getattr(np, "trapezoid", None) or np.trapz
    prob = 1 - np.exp(-trap(host.sig_annih[1, m] / host.dedx[1, m], t))
    assert 0.02 < prob < 0.5, prob


def test_positrons_conserve_energy_with_annihilation(host):
    """Many positrons: total deposits = kinetic + 2 m_e each, whether they annihilate in flight or at rest."""
    tb = cs.device_tables(host)
    cfg = cs.Config(n_cap=8192, m_cap=1 << 18, fill=0.25)
    nb = 64
    e = np.geomspace(5.0, 500.0, nb)
    f = jax.jit(lambda k, e_, v, d: cs.showers(tb, cfg, k, e_, v, d, primary=cs.E_PLUS))
    out = f(jax.random.PRNGKey(9), e, np.zeros((nb, 3)), np.tile([0.0, 0.0, 1.0], (nb, 1)))
    evs, _ = cs.split_events(out, nb)
    for ei, ev in zip(e, evs):
        assert ev["lost_mev"] == 0.0
        assert np.isclose(ev["de"].sum() + ev["nuclear_mev"], ei + 2 * em.ME, rtol=1e-8), (ei, ev["de"].sum())


def test_pair_cross_section_is_geant4s(host):
    """Below 30 GeV the table is Geant4's parametrisation (G4PairProductionRelModel, option 0); option B's own
    Bethe-Heitler integral stays within a few per cent of it above 5 MeV and joins it smoothly at 30 GeV."""
    k = np.array([2.0, 10.0, 50.0, 300.0, 3000.0])
    g4 = em.N_ATOM_MM3 * 1e-22 * em.pair_xsec_g4_barn(k)
    assert np.allclose(em.pair_xsec_table_per_mm(k), g4, rtol=1e-12)
    # Geant4 numbers (barn per argon atom) at 10 MeV and 1 GeV from the same fit
    assert np.isclose(em.pair_xsec_g4_barn(10.0), 0.7063, rtol=2e-3)
    assert np.isclose(em.pair_xsec_g4_barn(1000.0), 2.5011, rtol=2e-3)
    bh = em.pair_xsec_per_mm(k[1:])
    assert np.all(np.abs(bh / g4[1:] - 1) < 0.05)
    e = em.PAIR_G4_EMAX
    lo, hi = em.pair_xsec_table_per_mm(np.array([e * 0.999, e * 1.001]))
    assert abs(hi / lo - 1) < 0.01


def test_photonuclear_argon_table(host):
    """Giant dipole resonance of argon (G4PARTICLEXS gamma/inel18): ~37 mb near 17 MeV, ~1-2 % of a 20 MeV photon's
    interactions; nothing below the neutron threshold region."""
    from dunefastsim.transport import photonuclear_ar as pa

    assert 30.0 < pa.xsec_mb(17.0) < 45.0 and pa.xsec_mb(5.0) < 1.0
    k = host.k_nodes
    i = np.searchsorted(k, 20.0)
    frac = host.sig_pnuc[i] / (host.sig_pnuc[i] + host.sig_compton[i] + host.sig_pair[i] + host.sig_photo[i])
    assert 0.005 < frac < 0.03, frac


def test_photonuclear_sink_books_invisible_energy(host):
    """20 MeV photons: some are absorbed by nuclei; deposits + nuclear_mev = energy, the nuclear deposits (pdg 2212)
    are the visible share, and nuclear_mev the rest."""
    tb = cs.device_tables(host)
    cfg = cs.Config(n_cap=16384, m_cap=1 << 19, fill=0.25)  # 2000 photons at once: peak ~15800 live particles
    nb = 2000
    f = jax.jit(lambda k, e, v, d: cs.showers(tb, cfg, k, e, v, d, primary=cs.GAMMA))
    out = f(jax.random.PRNGKey(4), np.full(nb, 20.0), np.zeros((nb, 3)), np.tile([0.0, 0.0, 1.0], (nb, 1)))
    evs, _ = cs.split_events(out, nb)
    nuc = np.array([ev["nuclear_mev"] for ev in evs])
    vis = np.array([ev["de"][ev["pdg"] == 2212].sum() for ev in evs])
    for ev in evs:
        assert ev["lost_mev"] == 0.0 and np.isclose(ev["de"].sum() + ev["nuclear_mev"], 20.0, rtol=1e-8)
    assert 0 < (nuc > 0).sum() < 0.2 * nb  # a few per cent of the photons
    assert np.allclose(vis[nuc > 0] / (vis[nuc > 0] + nuc[nuc > 0]), cfg.pnuc_visible, atol=0.02)
