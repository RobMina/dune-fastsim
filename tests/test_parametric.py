"""Option A parametric showers: energy, profiles, gradients, both dtypes. Run on UVA (CPU is enough)."""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from dunefastsim.transport import parametric as pm  # noqa: E402

DTYPES = [jnp.float32, jnp.float64]


@pytest.mark.parametrize("alpha", [1.5, 3.0, 6.0])
def test_gamma_quantile_inverts_cdf(alpha):
    u = jnp.linspace(0.001, 0.999, 999)
    x = pm.gamma_quantile(jnp.float64(alpha), u)
    assert np.allclose(np.asarray(jax.scipy.special.gammainc(alpha, x)), np.asarray(u), atol=1e-8)


def test_material_constants():
    assert np.isclose(pm.X0_LAR_CM, 14.0, atol=0.01)
    assert 28.0 < pm.EC_MEV < 30.0 and 9.5 < pm.RM_CM < 10.5  # GFLASH definitions (PDG: 32.84 MeV, 9.04 cm)


@pytest.mark.parametrize("dtype", DTYPES)
def test_energy_conserved_and_axis(dtype):
    out = pm.shower(jax.random.PRNGKey(1), 1000.0, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], n_pairs=4096, dtype=dtype)
    assert out["positions_mm"].dtype == dtype
    assert np.isclose(float(out["de"].sum()), 1000.0, rtol=1e-5)
    pos = np.asarray(out["positions_mm"], np.float64)
    w = np.asarray(out["de"], np.float64)
    assert abs(np.average(pos[:, 0], weights=w)) < 10.0 and abs(np.average(pos[:, 1], weights=w)) < 10.0
    assert np.all(np.isfinite(pos))


def test_mean_depth_matches_gamma_mean():
    """Without fluctuations, the energy-weighted depth is alpha / beta (in X0), to the Monte Carlo error."""
    out = pm.shower(jax.random.PRNGKey(2), 2000.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], n_pairs=20000, fluctuate=False,
                    leading_track=False, photon_delay=False)
    sh = np.asarray(out["kind"]) > 0
    mean_t = np.average(np.asarray(out["depth_x0"])[sh], weights=np.asarray(out["de"])[sh])
    expect = float(out["alpha"] / out["beta"])
    assert np.isclose(mean_t, expect, rtol=0.02)
    tmax = float(out["tmax"])  # GFLASH: T = ln y - 0.812 at the mean
    assert np.isclose(tmax, np.log(2000.0 / pm.EC_MEV) - 0.812, rtol=1e-9)


def test_radial_containment():
    """Most of the energy within 2 Moliere radii of the axis (PDG: about 95 %)."""
    out = pm.shower(jax.random.PRNGKey(3), 1000.0, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], n_pairs=20000)
    pos, w = np.asarray(out["positions_mm"]), np.asarray(out["de"])
    r_cm = np.hypot(pos[:, 0], pos[:, 2]) / 10.0
    f2 = w[r_cm < 2 * pm.RM_CM].sum() / w.sum()
    assert 0.85 < f2 < 0.99, f2


def test_low_energy_clamped_finite():
    out = pm.shower(jax.random.PRNGKey(4), 40.0, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], n_pairs=256)
    assert np.all(np.isfinite(np.asarray(out["positions_mm"]))) and np.isclose(float(out["de"].sum()), 40.0)


def test_gradient_matches_finite_difference():
    """d(energy-weighted mean depth)/dE with a fixed key, against a central difference."""
    key = jax.random.PRNGKey(5)

    def mean_depth(e):
        out = pm.shower(key, e, jnp.zeros(3), jnp.array([0.0, 0.0, 1.0]), n_pairs=2048)
        return jnp.sum(out["positions_mm"][:, 2] * out["de"]) / jnp.sum(out["de"])

    g = float(jax.grad(mean_depth)(1500.0))
    h = 1e-3
    fd = float((mean_depth(1500.0 + h) - mean_depth(1500.0 - h)) / (2 * h))
    assert np.isfinite(g) and g > 0 and np.isclose(g, fd, rtol=1e-4)


def test_gradient_wrt_vertex_and_direction_finite():
    key = jax.random.PRNGKey(6)

    def loss(v, d):
        out = pm.shower(key, 800.0, v, d, n_pairs=512)
        return jnp.sum(out["de"] * jnp.sum(out["positions_mm"] ** 2, axis=1))

    gv, gd = jax.grad(loss, argnums=(0, 1))(jnp.array([10.0, -5.0, 3.0]), jnp.array([0.1, 0.2, 1.0]))
    assert np.all(np.isfinite(np.asarray(gv))) and np.all(np.isfinite(np.asarray(gd)))


@pytest.mark.parametrize("dtype", DTYPES)
def test_batch_jit(dtype):
    f = jax.jit(lambda k, e, v, d: pm.showers(k, e, v, d, n_pairs=256, dtype=dtype))
    e = np.array([50.0, 300.0, 5000.0])
    out = f(jax.random.PRNGKey(7), e, np.zeros((3, 3)), np.tile([0.0, 0.0, 1.0], (3, 1)))
    assert out["positions_mm"].shape == (3, 128 + 512, 3)
    assert np.allclose(np.asarray(out["de"], np.float64).sum(axis=1), e, rtol=1e-4)


def test_float32_float64_agree():
    """Same key: the two dtypes give the same showers to float32 precision (statistically identical draws)."""
    args = (jax.random.PRNGKey(8), 1000.0, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    a = pm.shower(*args, n_pairs=1024, fluctuate=False, dtype=jnp.float32)
    b = pm.shower(*args, n_pairs=1024, fluctuate=False, dtype=jnp.float64)
    za = np.average(np.asarray(a["positions_mm"][:, 2], np.float64), weights=np.asarray(a["de"], np.float64))
    zb = np.average(np.asarray(b["positions_mm"][:, 2]), weights=np.asarray(b["de"]))
    assert np.isclose(za, zb, rtol=0.05)


def test_track_split():
    e = jnp.array([35.0, 100.0, 1000.0, 20000.0])
    e_trk, length = pm.track_split(e)
    frac = np.asarray(e_trk / e)
    assert np.all(frac > 0) and np.all(frac < 1) and np.all(np.diff(frac) < 0)  # track-like at low energy
    assert 0.05 < frac[2] < 0.2  # about 10 % at 1 GeV
    assert np.allclose(np.asarray(e_trk / length), pm.DEDX_MIP_MEV_CM)


def test_track_spots_on_axis_and_energy_split():
    out = pm.shower(jax.random.PRNGKey(9), 60.0, [5.0, 5.0, 5.0], [0.0, 1.0, 0.0], n_pairs=256)
    kind = np.asarray(out["kind"])
    trk = np.asarray(out["positions_mm"])[kind == 0]
    assert np.allclose(trk[:, 0], 5.0) and np.allclose(trk[:, 2], 5.0) and np.all(trk[:, 1] > 5.0)
    assert np.isclose(float(np.asarray(out["de"])[kind == 0].sum()), float(out["e_track"]), rtol=1e-9)
    assert np.isclose(float(out["de"].sum()), 60.0, rtol=1e-9)


def test_fluctuation_ramp():
    w = np.asarray(pm.fluctuation_weight(jnp.array([100.0, 300.0, 1000.0])))
    assert w[0] < 0.02 and np.isclose(w[1], 0.5) and w[2] > 0.98


def test_dedx_spread_matches_reference_quantiles():
    """Energy-weighted dE/dx of the spots: the reference's 10/50/90 % = 1.642 / 1.922 / 3.325 MeV/cm."""
    out = pm.shower(jax.random.PRNGKey(10), 1000.0, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], n_pairs=50000, n_track=4096)
    de, dx = np.asarray(out["de"]), np.asarray(out["dx"])
    dedx = 10.0 * de / dx
    o = np.argsort(dedx)
    c = np.cumsum(de[o]) / de.sum()
    q = [dedx[o][np.searchsorted(c, p)] for p in (0.1, 0.5, 0.9)]
    assert np.allclose(q, [1.642, 1.922, 3.325], rtol=0.03), q


def test_photon_delay():
    d = np.asarray(pm.photon_delay_x0(jnp.array([10.0, 300.0, 3000.0])))
    assert np.all(np.diff(d) < 0) and np.isclose(d[1], 9.0 / 14.0) and d[0] < 9.0 / 7.0
    kw = dict(n_pairs=20000, fluctuate=False, leading_track=False)
    a = pm.shower(jax.random.PRNGKey(11), 200.0, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], photon_delay=False, **kw)
    b = pm.shower(jax.random.PRNGKey(11), 200.0, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], photon_delay=True, **kw)
    ta = np.average(np.asarray(a["depth_x0"]), weights=np.asarray(a["de"]))
    tb = np.average(np.asarray(b["depth_x0"]), weights=np.asarray(b["de"]))
    assert np.isclose(tb - ta, float(pm.photon_delay_x0(200.0)), rtol=0.05)
