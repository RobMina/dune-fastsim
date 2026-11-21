"""Option A: parametric electromagnetic showers in liquid argon, on the device and differentiable.

The GFLASH parametrisation of showers in homogeneous media (G. Grindhammer, M. Rudowicz, S. Peters, NIM A290 (1990)
469; G. Grindhammer, S. Peters, hep-ex/0001020), reimplemented in JAX from the published formulas. The constants were
checked against Geant4's GFlash (``GVFlashHomoShowerTuning.hh``, ``GFlashHomoShowerParameterisation.cc``) on
2026-11-21. Geant4 is not used at run time.

Per shower (energy E, vertex, direction):

- longitudinal energy profile: a gamma distribution in depth t (radiation lengths) with shape alpha and
  rate beta = (alpha - 1) / T, where ln T and ln alpha are correlated normal draws whose means and widths depend on
  ln y, y = E / E_c (``longitudinal``);
- radial profile at each depth: two components (core and tail) of the form 2 r R^2 / (r^2 + R^2)^2, radii R_C and
  R_T in Moliere radii and core weight p, all functions of tau = t / T and ln E (``radial``).

Deposits are a fixed number of energy spots per shower (fixed shapes for ``jit``/``vmap``). Spots come in pairs at a
common depth, one drawn from the core and one from the tail, carrying p and 1 - p of the pair's energy, so the core
weight enters through energies instead of a discrete choice and stays differentiable. Depths are gamma quantiles of
fixed uniforms (``gamma_quantile``: Wilson-Hilferty start, Newton steps on the regularised incomplete gamma), not
``jax.random.gamma``: that is a rejection sampler, whose samples for a fixed key are not smooth in the shape. With the
random key fixed, every output is a smooth function of E, the vertex, the direction and the material constants.

Additions to GFLASH (2026-11-21, from the comparison with the Geant4 reference, see the wiki plan):

- **leading track** (``track_split``): the primary electron ionises along the axis before its energy is radiated
  away. Its straight track has length L = X0 ln(1 + E/E_c) at MIP dE/dx, and carries
  E_trk = E (1 - exp(-L dE/dx_MIP / E)) (smooth, below E). The rest, E - E_trk, is the GFLASH shower, with its shape
  evaluated at that energy. At low energy most of the energy is track-like; at 1 GeV the track carries about 10 %;
- **fluctuations ramp in** (``fluctuation_weight``): GFLASH's widths diverge near threshold (sigma(ln T) is 3.6 at
  110 MeV in argon), so they are scaled by a smooth logistic step in ln E centred at ``E_FLUCT_MEV``;
- **dE/dx spread**: each spot's dx = de / (dE/dx), with dE/dx = floor + scale exp(sigma z) for a fixed normal z
  per spot. The constants fit the energy-weighted 10/50/90 % quantiles of the reference's steps (1.64 / 1.92 /
  3.3 MeV/cm, nearly the same at every electron energy), so recombination sees a Landau-like spread.

- **photon conversion delay** (``photon_delay_x0``): at low energy the shower part is mostly bremsstrahlung photons,
  which travel a mean 9/7 X0 before converting; GFLASH's fitted profile already contains that at high energy, and the
  clamped shape below ``E_MIN_MEV`` does not. Shower spots get an extra depth lambda Exp(1) (fixed uniforms) with
  lambda = (9/7) X0 E_gamma / (E_gamma + E_shower), E_gamma = ``E_DELAY_MEV``, so the delay fades with energy.

Known limits (the reason option B exists): the track is straight (no multiple scattering), photons convert at the
GFLASH profile's depths rather than after a sampled conversion length, and the medium is infinite argon
(boundaries, inactive material and leakage are applied downstream by the geometry). Below ``E_MIN_MEV`` the shower
shape is evaluated at ``E_MIN_MEV``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

# Liquid argon (PDG): Z, A, density, radiation length.
Z_AR = 18.0
A_AR = 39.948
RHO_LAR = 1.396  # g/cm^3
X0_LAR_G_CM2 = 19.55
X0_LAR_CM = X0_LAR_G_CM2 / RHO_LAR  # 14.00 cm
# GFLASH's own material definitions (GFlashHomoShowerParameterisation::SetMaterial), kept so that the fitted
# constants see the E_c they were fitted with: E_c = 2.66 (X0 rho Z/A)^1.1 MeV = 29.1 MeV, R_M = X0 * 21.2 MeV / E_c.
# (PDG's values for liquid argon are E_c = 32.84 MeV and R_M = 9.04 cm.)
EC_MEV = 2.66 * (X0_LAR_G_CM2 * Z_AR / A_AR) ** 1.1
RM_CM = X0_LAR_CM * 21.2 / EC_MEV
DEDX_MIP_MEV_CM = 2.12  # minimum ionisation in liquid argon, 1.519 MeV cm^2/g x 1.396 g/cm^3
E_MIN_MEV = 100.0
E_FLUCT_MEV = 300.0  # centre of the fluctuation ramp
FLUCT_WIDTH_LN = 0.25  # width of the ramp in ln E
# Energy-weighted dE/dx of the reference's steps (bdnmc-dune edep-sim, 2000 electrons, 2026-11-21):
# q10/q50/q90 = 1.642 / 1.922 / 3.325 MeV/cm, fitted exactly by floor + scale exp(sigma z).
E_DELAY_MEV = 300.0  # energy scale over which the photon conversion delay fades
DEDX_FLOOR_MEV_CM = 1.572
DEDX_SCALE_MEV_CM = 0.350
DEDX_SIGMA_LN = 1.258


@dataclass(frozen=True)
class GFlashConstants:
    """GFLASH homogeneous-media constants (Grindhammer et al. 1990; Geant4 GVFlashHomoShowerTuning)."""

    ave_t1: float = -0.812
    ave_a1: float = 0.81
    ave_a2: float = 0.458
    ave_a3: float = 2.26
    sig_t1: float = -1.4
    sig_t2: float = 1.26
    sig_a1: float = -0.58
    sig_a2: float = 0.86
    rho1: float = 0.705
    rho2: float = -0.023
    rc1: float = 0.0251
    rc2: float = 0.00319
    rc3: float = 0.1162
    rc4: float = -0.000381
    rt1: float = 0.659
    rt2: float = -0.00309
    rt3: float = 0.645
    rt4: float = -2.59
    rt5: float = 0.3585
    rt6: float = 0.0412
    wc1: float = 2.632
    wc2: float = -0.00094
    wc3: float = 0.401
    wc4: float = 0.00187
    wc5: float = 1.313
    wc6: float = -0.0686


GF = GFlashConstants()


def longitudinal(energy_mev, z_normal, fluctuate=True, c: GFlashConstants = GF, z=Z_AR, ec_mev=EC_MEV):
    """Shape (alpha), rate (beta, per X0) and T of the longitudinal profile, and <ln alpha> (needed for tau).

    ``z_normal``: (..., 2) standard normal draws (fixed for pathwise gradients). Energies below ``E_MIN_MEV`` use
    the shape at ``E_MIN_MEV``; fluctuations are scaled by ``fluctuation_weight``.
    """
    e = jnp.asarray(energy_mev)
    ln_y = jnp.log(jnp.maximum(e, E_MIN_MEV) / ec_mev)
    ave_ln_t = jnp.log(c.ave_t1 + ln_y)
    ave_ln_a = jnp.log(c.ave_a1 + (c.ave_a2 + c.ave_a3 / z) * ln_y)
    sig_ln_t = 1.0 / (c.sig_t1 + c.sig_t2 * ln_y)
    sig_ln_a = 1.0 / (c.sig_a1 + c.sig_a2 * ln_y)
    rho = c.rho1 + c.rho2 * ln_y
    c1, c2 = jnp.sqrt((1.0 + rho) / 2.0), jnp.sqrt((1.0 - rho) / 2.0)
    on = fluctuation_weight(e) if fluctuate else jnp.zeros_like(e)
    z1, z2 = z_normal[..., 0], z_normal[..., 1]
    tmax = jnp.exp(ave_ln_t + on * sig_ln_t * (c1 * z1 + c2 * z2))
    alpha = jnp.exp(ave_ln_a + on * sig_ln_a * (c1 * z1 - c2 * z2))
    beta = (alpha - 1.0) / tmax
    return alpha, beta, tmax, ave_ln_a


def fluctuation_weight(energy_mev):
    """Smooth 0 -> 1 step in ln E centred at ``E_FLUCT_MEV`` (scales GFLASH's fluctuation widths)."""
    # math.log, not np.log: a NumPy float64 scalar is strongly typed in JAX and would promote float32 to float64
    return jax.nn.sigmoid((jnp.log(jnp.maximum(energy_mev, 1e-3)) - math.log(E_FLUCT_MEV)) / FLUCT_WIDTH_LN)


def track_split(energy_mev, ec_mev=EC_MEV):
    """(energy on the leading track, its length in cm): L = X0 ln(1 + E/E_c) at MIP dE/dx, capped smoothly at E."""
    e = jnp.asarray(energy_mev)
    reach_cm = X0_LAR_CM * jnp.log1p(e / ec_mev)
    e_trk = e * (1.0 - jnp.exp(-DEDX_MIP_MEV_CM * reach_cm / e))
    return e_trk, e_trk / DEDX_MIP_MEV_CM


def photon_delay_x0(shower_energy_mev):
    """Mean extra depth (X0) of the shower part: 9/7 X0 scaled by E_gamma / (E_gamma + E_shower)."""
    return (9.0 / 7.0) * E_DELAY_MEV / (E_DELAY_MEV + shower_energy_mev)


def radial(energy_mev, tau, c: GFlashConstants = GF, z=Z_AR):
    """Core radius, tail radius (Moliere radii) and core weight at reduced depth ``tau``."""
    ln_e = jnp.log(jnp.maximum(energy_mev, E_MIN_MEV) / 1000.0)
    r_core = (c.rc1 + c.rc2 * ln_e) + (c.rc3 + c.rc4 * z) * tau
    p1, p2, p3 = c.wc1 + c.wc2 * z, c.wc3 + c.wc4 * z, c.wc5 + c.wc6 * ln_e
    x = (p2 - tau) / p3
    w_core = p1 * jnp.exp(x - jnp.exp(x))
    k1, k2, k3, k4 = c.rt1 + c.rt2 * z, c.rt3, c.rt4, c.rt5 + c.rt6 * ln_e
    r_tail = k1 * (jnp.exp(k3 * (tau - k2)) + jnp.exp(k4 * (tau - k2)))
    return r_core, r_tail, jnp.clip(w_core, 0.0, 1.0)


def gamma_quantile(alpha, u, newton: int = 4):
    """Quantile of the unit-rate gamma distribution with shape ``alpha`` at probabilities ``u`` (smooth in both).

    Wilson-Hilferty start, then Newton steps on P(alpha, x) - u (``jax.scipy.special.gammainc``, differentiable in
    alpha through ``lax.igamma_grad_a``).
    """
    u = jnp.clip(u, 1e-7, 1.0 - 1e-7)
    z = jax.scipy.special.ndtri(u)
    x = alpha * (1.0 - 1.0 / (9.0 * alpha) + z / (3.0 * jnp.sqrt(alpha))) ** 3
    x = jnp.maximum(x, 1e-3 * alpha)
    for _ in range(newton):
        log_pdf = (alpha - 1.0) * jnp.log(x) - x - jax.scipy.special.gammaln(alpha)
        x = x - (jax.scipy.special.gammainc(alpha, x) - u) / jnp.exp(log_pdf)
        x = jnp.maximum(x, 1e-6 * alpha)
    return x


def _frame(direction):
    """Unit axis and two unit vectors perpendicular to it."""
    u = direction / jnp.linalg.norm(direction)
    helper = jnp.where(jnp.abs(u[2]) < 0.9, jnp.array([0.0, 0.0, 1.0], u.dtype), jnp.array([1.0, 0.0, 0.0], u.dtype))
    e1 = jnp.cross(u, helper)
    e1 = e1 / jnp.linalg.norm(e1)
    return u, e1, jnp.cross(u, e1)


def shower(key, energy_mev, vertex_mm, direction, n_pairs: int = 512, fluctuate: bool = True,
           u_max: float = 0.995, dtype=jnp.float64, n_track: int = 128, leading_track: bool = True,
           dedx_spread: bool = True, photon_delay: bool = True):
    """Spots of one electron: positions (n_track + 2 n_pairs, 3) mm, de (MeV), dx (mm), theta, phi (axis direction).

    The first ``n_track`` spots are the leading track (evenly spaced along the axis), then pairs: spot
    n_track + 2k is the core member and n_track + 2k + 1 the tail member of pair k. ``kind`` is 0 / 1 / 2 for
    track / core / tail. ``u_max`` caps the radial quantile (the tails are heavy: r = R sqrt(u / (1 - u))); the
    energy of a pair is kept, so the cap moves energy inward slightly.
    """
    k_z, k_t, k_r, k_phi, k_dedx, k_dly = jax.random.split(key, 6)
    e0 = jnp.asarray(energy_mev, dtype)
    if leading_track:
        e_trk, len_trk_cm = track_split(e0)
    else:
        e_trk, len_trk_cm = jnp.zeros_like(e0), jnp.zeros_like(e0)
    e = e0 - e_trk  # energy of the shower part
    alpha, beta, tmax, ave_ln_a = longitudinal(e, jax.random.normal(k_z, (2,), dtype), fluctuate)
    t = gamma_quantile(alpha, jax.random.uniform(k_t, (n_pairs,), dtype)) / beta  # depth in X0
    ave_a = jnp.exp(ave_ln_a)
    tau = t / tmax * (alpha - 1.0) / alpha * ave_a / (ave_a - 1.0)  # GFLASH reduced depth
    r_core, r_tail, w_core = radial(e, tau)
    if photon_delay:  # shifts the deposits deeper; GFLASH's tau (radial shape) stays that of the undelayed depth
        u_d = jnp.clip(jax.random.uniform(k_dly, (n_pairs,), dtype), 0.0, 1.0 - 1e-7)
        t = t - photon_delay_x0(e) * jnp.log1p(-u_d)
    u = jnp.minimum(jax.random.uniform(k_r, (n_pairs, 2), dtype), u_max)
    q = jnp.sqrt(u / (1.0 - u))
    r_cm = RM_CM * jnp.stack([r_core * q[:, 0], r_tail * q[:, 1]], axis=1)  # (n_pairs, 2)
    phi_r = 2.0 * jnp.pi * jax.random.uniform(k_phi, (n_pairs, 2), dtype)
    axis, e1, e2 = _frame(jnp.asarray(direction, dtype))
    along = (t * X0_LAR_CM * 10.0)[:, None, None] * axis  # mm
    across = (r_cm * 10.0)[..., None] * (jnp.cos(phi_r)[..., None] * e1 + jnp.sin(phi_r)[..., None] * e2)
    pos = (jnp.asarray(vertex_mm, dtype) + along + across).reshape(-1, 3)
    de_pair = e / n_pairs
    de = jnp.stack([de_pair * w_core, de_pair * (1.0 - w_core)], axis=1).reshape(-1)
    s_cm = (jnp.arange(n_track, dtype=dtype) + 0.5) / n_track * len_trk_cm  # track spots along the axis
    pos = jnp.concatenate([jnp.asarray(vertex_mm, dtype) + (s_cm * 10.0)[:, None] * axis, pos])
    de = jnp.concatenate([jnp.full(n_track, e_trk / n_track, dtype), de])
    n = n_track + 2 * n_pairs
    if dedx_spread:
        z = jax.random.normal(k_dedx, (n,), dtype)
        dedx = DEDX_FLOOR_MEV_CM + DEDX_SCALE_MEV_CM * jnp.exp(DEDX_SIGMA_LN * z)
    else:
        dedx = jnp.full(n, DEDX_MIP_MEV_CM, dtype)
    dx_mm = de / dedx * 10.0
    theta = jnp.arccos(jnp.clip(axis[2], -1.0, 1.0))
    phi = jnp.arctan2(axis[1], axis[0])
    kind = jnp.concatenate([jnp.zeros(n_track, jnp.int32), jnp.tile(jnp.array([1, 2], jnp.int32), n_pairs)])
    return {"positions_mm": pos, "de": de, "dx": dx_mm, "theta": jnp.full(n, theta), "phi": jnp.full(n, phi),
            "depth_x0": jnp.concatenate([s_cm / X0_LAR_CM, jnp.repeat(t, 2)]), "kind": kind,
            "alpha": alpha, "beta": beta, "tmax": tmax, "e_track": e_trk, "track_length_cm": len_trk_cm}


def showers(key, energy_mev, vertex_mm, direction, n_pairs: int = 512, fluctuate: bool = True, dtype=jnp.float64,
            **kw):
    """``shower`` over a batch: energy (n,), vertex (n, 3) mm, direction (n, 3). Leading axis = shower."""
    keys = jax.random.split(key, np.shape(energy_mev)[0])
    f = jax.vmap(lambda k, e, v, d: shower(k, e, v, d, n_pairs, fluctuate, dtype=dtype, **kw))
    return f(keys, jnp.asarray(energy_mev, dtype), jnp.asarray(vertex_mm, dtype), jnp.asarray(direction, dtype))
