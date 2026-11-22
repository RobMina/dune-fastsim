"""Energy-loss fluctuations of e+- in Geant4's standard EM physics (option 0): G4UniversalFluctuation, Geant4 11.

The Geant4 reference (edep-sim with QGSP_BERT, the DUNE ND production setting) uses G4EmStandardPhysics, whose
default fluctuation model since Geant4 11.0 is G4UniversalFluctuation with a one-level ``SampleGlandz``: one
excitation level at the mean excitation energy I, width factor ``0.1 + (fw - 0.1) sqrt(a1 / a0)`` for few
excitations, plus the 1/E^2 ionisation part. hepsimkernels' sampler is the two-level G4UrbanFluctuation variant
(EM options 3 and 4, with f1, f2, e1, e2 from G4IonisParamMat). Transcribed from
``source/processes/electromagnetic/standard/src/G4UniversalFluctuation.cc`` at v11.4.0 (identical to v11.3.2).

For e+- Geant4 never takes the Gaussian/Gamma (Bohr) branch (it requires a particle heavier than the electron), so
the sampler here has only the Glandz branch. ``cut`` is min(production cut, Tmax) as G4VEnergyLossProcess passes it.

``sample_universal`` (JAX, jit-able, fixed shapes) and ``sample_universal_np`` (NumPy, the literal transcription with
rejection loops, the reference for the tests).
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

# G4UniversalFluctuation constants (G4UniversalFluctuation.hh, v11.4.0)
RATE, FW, A0, NMAXCONT, MINLOSS = 0.56, 4.0, 42.0, 8.0, 10e-6  # MINLOSS 10 eV in MeV
K_POISSON = 32  # Poisson means here are <= 8 (excitation: Gaussian above NMAXCONT; ionisation: 8 a3 / (8 + a3))


def _params(ml, tc, I, e0, xp=np):
    """One-level Glandz parameters: (a1, e1, a3, w1) for mean loss ml (already divided by the scaling) and cut tc."""
    ex = tc > I
    a1 = xp.where(ex, ml * (1.0 - RATE) / I, 0.0)
    fwnow = xp.where(a1 < A0, 0.1 + (FW - 0.1) * xp.sqrt(xp.clip(a1, 0.0, None) / A0), FW)
    a1 = xp.where(ex, a1 / fwnow, 0.0)
    e1 = xp.where(ex, I * fwnow, I)
    w1 = tc / e0
    lw1 = xp.log(xp.where(w1 > 1.0, w1, 2.0))
    a3 = RATE * ml * (tc - e0) / (e0 * tc * lw1)
    a3 = xp.where(a1 <= 0.0, a3 / RATE, a3)
    return a1, e1, a3, w1


def _ion_split(a3, w1, xp=np):
    """Continuous/Poisson split of the ionisation part for a3 > NMAXCONT: (alfa, alfa1, namean, p3).

    Geant4 writes namean = a3 w1 (alfa - 1) / ((w1 - 1) alfa) and p3 = a3 - namean, algebraically
    p3 = NMAXCONT a3 / (NMAXCONT + a3); the closed form avoids the cancellation for w1 -> 1 (as hepsimkernels)."""
    cont = a3 > NMAXCONT
    alfa = xp.where(cont, w1 * (NMAXCONT + a3) / (w1 * NMAXCONT + a3), 1.0)
    am1 = xp.where(cont, a3 * (w1 - 1.0) / (w1 * NMAXCONT + a3), 1.0)  # alfa - 1
    am1 = xp.where(am1 > 0.0, am1, 1.0)
    alfa1 = xp.where(cont, xp.where(am1 < 1e-8, 1.0 + 0.5 * am1, alfa * xp.log1p(am1) / am1), 0.0)
    p3 = xp.where(cont, NMAXCONT * a3 / (NMAXCONT + a3), a3)
    namean = xp.where(cont, a3 - p3, 0.0)
    return alfa, alfa1, namean, p3


# ---------------------------------------------------------------------------------------------------- NumPy reference
def _gauss_np(rng, eav, sig2):
    """G4UniversalFluctuation::SampleGauss (scalar)."""
    sig = np.sqrt(sig2)
    if eav < 0.25 * sig:
        return eav + (2.0 * rng.random() - 1.0) * eav
    while True:
        x = rng.normal(eav, sig)
        if 0.0 <= x <= 2.0 * eav:
            return x


def sample_universal_np(rng, mean, cut, I, e0):
    """Literal scalar-loop transcription of SampleFluctuations + SampleGlandz for e+- (arrays in, array out)."""
    mean, cut = np.broadcast_arrays(np.asarray(mean, float), np.asarray(cut, float))
    out = np.empty(mean.shape)
    for i, (m, tc) in enumerate(zip(mean.ravel(), cut.ravel())):
        if m < MINLOSS or tc <= e0:
            out.flat[i] = m
            continue
        scaling = min(1.0 + 0.5e-3 / tc, 1.5)
        ml = m / scaling
        a1, e1 = 0.0, I
        if tc > e1:
            a1 = ml * (1.0 - RATE) / e1
            if a1 < A0:
                fwnow = 0.1 + (FW - 0.1) * np.sqrt(a1 / A0)
                a1 /= fwnow
                e1 *= fwnow
            else:
                a1 /= FW
                e1 *= FW
        w1 = tc / e0
        a3 = RATE * ml * (tc - e0) / (e0 * tc * np.log(w1))
        if a1 <= 0.0:
            a3 /= RATE
        loss, emean, sig2e = 0.0, 0.0, 0.0
        if a1 > 0.0:
            if a1 > NMAXCONT:
                emean += a1 * e1
                sig2e += a1 * e1 * e1
            else:
                p = rng.poisson(a1)
                if p > 0:
                    loss += ((p + 1) - 2.0 * rng.random()) * e1
        if sig2e > 0.0:
            loss += _gauss_np(rng, emean, sig2e)
        if a3 > 0.0:
            emean, sig2e, p3, alfa = 0.0, 0.0, a3, 1.0
            if a3 > NMAXCONT:
                alfa = w1 * (NMAXCONT + a3) / (w1 * NMAXCONT + a3)
                alfa1 = alfa * np.log(alfa) / (alfa - 1.0)
                namean = a3 * w1 * (alfa - 1.0) / ((w1 - 1.0) * alfa)
                emean += namean * e0 * alfa1
                sig2e += e0 * e0 * namean * (alfa - alfa1 * alfa1)
                p3 = a3 - namean
            w3 = alfa * e0
            if tc > w3:
                w = (tc - w3) / tc
                nnb = rng.poisson(p3)
                if nnb > 0:
                    loss += np.sum(w3 / (1.0 - w * rng.random(nnb)))
            if sig2e > 0.0:
                loss += _gauss_np(rng, emean, sig2e)
        out.flat[i] = loss * scaling
    return out


# ---------------------------------------------------------------------------------------------------- JAX
def _gauss_clipped(u, eav, sig2):
    """SampleGauss from one uniform: uniform in [0, 2 eav] if eav < sig/4, else the Gauss truncated to [0, 2 eav]
    by inverse CDF (replaces the rejection loop, same distribution)."""
    from jax.scipy.special import ndtr, ndtri

    sig = jnp.sqrt(jnp.maximum(sig2, 0.0))
    flat = eav < 0.25 * sig
    b = jnp.where(sig2 > 0, eav / jnp.where(sig2 > 0, sig, 1.0), 1.0)
    pa = ndtr(-b)
    z = jnp.clip(ndtri(pa + u * (1 - 2 * pa)), -b, b)
    return jnp.where(flat, eav + (2 * u - 1) * eav, jnp.where(sig2 > 0, eav + sig * z, 0.0))


def sample_universal(key, mean, cut, I, e0, K: int = K_POISSON):
    """Sampled e+- continuous loss [MeV] for mean loss ``mean`` and cut = min(tcut, Tmax) [MeV] (arrays of one shape;
    I and e0 [MeV] scalars). Inverse-CDF Poisson in a fixed K-round loop; the result has mean ``mean``."""
    dt = mean.dtype
    sh = mean.shape
    u = (mean >= MINLOSS) & (cut > e0)
    tc = jnp.where(u, cut, 1.0)
    scaling = jnp.minimum(1.0 + 0.5e-3 / tc, 1.5)
    ml = jnp.where(u, mean, 0.0) / scaling
    a1, e1, a3, w1 = _params(ml, tc, I, e0, jnp)
    cont1 = a1 > NMAXCONT
    eav = jnp.where(cont1, a1 * e1, 0.0)
    s2 = jnp.where(cont1, a1 * e1 * e1, 0.0)
    lam1 = jnp.where(cont1 | (a1 <= 0.0), 0.0, a1)
    alfa, alfa1, namean, p3 = _ion_split(a3, w1, jnp)
    eav3 = namean * e0 * alfa1
    s23 = e0 * e0 * namean * (alfa - alfa1 * alfa1)
    w3 = alfa * e0
    has = (tc > w3) & (a3 > 0.0)
    w = jnp.where(has, (tc - w3) / tc, 0.0)
    lam3 = jnp.where(has, jnp.clip(p3, 0.0, None), 0.0)
    ks = jax.random.split(key, 5)
    uu = jax.random.uniform(ks[0], (2,) + sh, dt)
    lams = jnp.stack([lam1, lam3])

    def body(j, carry):
        pmf, cdf, k, ion = carry
        above = uu > cdf
        r = jax.random.uniform(jax.random.fold_in(ks[1], j), sh, dt)
        ion = ion + jnp.where(above[1], w3 / (1.0 - w * r), 0.0)
        pmf = pmf * lams / (j + 1)
        return pmf, cdf + pmf, k + above, ion

    pmf0 = jnp.exp(-lams)
    _, _, kk, ion = lax.fori_loop(0, K, body, (pmf0, pmf0, jnp.zeros((2,) + sh, jnp.int32), jnp.zeros(sh, dt)))
    rx = jax.random.uniform(ks[2], sh, dt)
    loss = jnp.where(kk[0] > 0, ((kk[0] + 1) - 2.0 * rx) * e1, 0.0)
    loss = loss + jnp.where(s2 > 0, _gauss_clipped(jax.random.uniform(ks[3], sh, dt), eav, s2), 0.0)
    loss = loss + ion
    loss = loss + jnp.where(s23 > 0, _gauss_clipped(jax.random.uniform(ks[4], sh, dt), eav3, s23), 0.0)
    return jnp.where(u, loss * scaling, mean).astype(dt)
