"""G4UniversalFluctuation (EM option 0) sampler for e+-: JAX against the literal NumPy transcription, mean kept."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("hepsimkernels")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from dunefastsim.transport import fluct  # noqa: E402


@pytest.fixture(scope="module")
def lar():
    from hepsimkernels import eloss_jax

    from dunefastsim.transport.emphysics import lar_material

    c = eloss_jax.consts(lar_material())
    assert 1e-4 < c.I < 3e-4 and 0 < c.e0 < 1e-4  # MeV: I(LAr) = 188 eV, e0 = 10 eV
    return c


# (mean loss MeV, cut MeV): LAr, ~0.21 MeV/mm restricted loss, delta cut 0.22 MeV, steps 0.1 / 0.5 / 2 / 3.72 mm;
# a low-energy electron whose Tmax = T/2 sets the cut; a tiny loss in the Poisson-only regime
CASES = [(0.021, 0.22), (0.105, 0.22), (0.42, 0.22), (0.78, 0.22), (0.3, 0.05), (2e-4, 0.22)]


@pytest.mark.parametrize("mean,cut", CASES)
def test_jax_matches_numpy_transcription(lar, mean, cut):
    rng = np.random.default_rng(7)
    ref = fluct.sample_universal_np(rng, np.full(20000, mean), cut, lar.I, lar.e0)
    m = jnp.full((400000,), mean)
    got = np.asarray(fluct.sample_universal(jax.random.PRNGKey(3), m, jnp.full_like(m, cut), lar.I, lar.e0))
    assert np.all(np.isfinite(got)) and got.min() >= 0.0
    # mean kept (the model is unbiased by construction), within 5 standard errors: for tiny means the 1/E^2
    # ionisation tail out to the cut makes the spread several times the mean
    for x in (got, ref):
        assert abs(x.mean() - mean) < 5 * x.std() / np.sqrt(x.size), (x.mean(), mean)
    for q in (0.1, 0.5, 0.9):
        a, b = np.quantile(ref, q), np.quantile(got, q)
        assert abs(a - b) < 0.02 * mean + 3e-6, (q, a, b)


def test_below_thresholds_returns_mean(lar):
    m = jnp.array([5e-6, 0.1, 0.1])
    c = jnp.array([0.22, 0.5 * lar.e0, lar.e0])
    out = np.asarray(fluct.sample_universal(jax.random.PRNGKey(0), m, c, lar.I, lar.e0))
    assert np.array_equal(out, np.asarray(m))


def test_float32_runs(lar):
    m = jnp.full((100000,), 0.78, jnp.float32)
    out = fluct.sample_universal(jax.random.PRNGKey(1), m, jnp.full_like(m, 0.22), jnp.float32(lar.I),
                                 jnp.float32(lar.e0))
    assert out.dtype == jnp.float32 and abs(float(out.mean()) / 0.78 - 1) < 0.01
