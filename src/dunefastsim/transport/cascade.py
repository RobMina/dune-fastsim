"""Option B (increments B0 + B1): an electron/photon cascade in liquid argon, in JAX.

Particles live in a fixed-capacity buffer (``n_cap`` lanes). Each iteration of a ``lax.while_loop`` moves every live
particle one step (a wavefront), then writes the step deposits to a fixed-capacity deposit buffer and rebuilds the
particle buffer: survivors first, then the secondaries of this step (at most two per lane), placed by prefix sums.
Overflow is counted (``lost_mev``), never silent.

Physics per step (tables from ``emphysics``; production cuts tcut for deltas and kcut for photons):

- e-/e+: step = min(distance to the next discrete interaction, a fraction of the residual range clipped to
  [s_min, s_max], the residual range). Restricted loss along it: the range-table mean, fluctuated with Geant4's
  Urban model for e+- (``hepsimkernels.eloss_jax.sample_loss_electron``, B1), deposited at the step midpoint.
  Multiple scattering (B1): the deflection over the step has <1 - cos> = 1 - exp(-s / lambda_1), lambda_1 the
  Mott-corrected first transport mean free path of ``hepsimkernels.emsc``, drawn from a Wentzel (screened-Rutherford)
  distribution with that mean, so the single-scattering tail is kept; the step chord follows the mean of the initial
  and final directions (lateral displacement and path-length correction to first order).
  Discrete interactions at the post-step energy: bremsstrahlung above kcut (photon angle from Geant4's modified
  Tsai distribution, B1) or a delta ray above tcut (two-body kinematics). Below the tracking cut ``t_track`` the remaining kinetic energy is
  deposited; a positron then annihilates at rest into two 511 keV photons, back to back, isotropic.
- photons: fly to the next Compton or pair interaction (no deposits). Compton: Klein-Nishina energy and angle, the
  electron from momentum conservation. Pair: electron/positron energies from Bethe-Heitler, each lepton at a modified-Tsai angle,
  opposite azimuths (B1).
  Photons below ``k_track`` are absorbed where they are (photoelectric, local deposit).

Not yet: the mixed (class II) scattering scheme itself, annihilation in flight, photoelectric cross section above ``k_track``, Rayleigh scattering, boundaries (infinite
argon; the geometry cut is applied downstream as for option A).

Gradients: ``lax.while_loop`` is differentiable in forward mode (``jax.jvp``) only; see the plan (decision 6, B2).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from dunefastsim.transport.emphysics import ME, HostTables

C_MM_NS = 299.792458
E_MINUS, E_PLUS, GAMMA, DEAD = 0, 1, 2, -1


class Tables(NamedTuple):
    lnt0: jax.Array
    dlnt: jax.Array
    t_nodes: jax.Array
    dedx: jax.Array
    rng: jax.Array
    sig_brem: jax.Array
    sig_delta: jax.Array
    brem_q: jax.Array
    delta_q: jax.Array
    lam1: jax.Array
    wentzel_lnq: jax.Array
    wentzel_lna: jax.Array
    lnk0: jax.Array
    dlnk: jax.Array
    sig_compton: jax.Array
    sig_pair: jax.Array
    compton_q: jax.Array
    pair_q: jax.Array
    tcut: jax.Array
    kcut: jax.Array
    x0_mm: jax.Array
    mc: tuple  # hepsimkernels.eloss_jax.MatConsts of liquid argon (fluctuation constants)


def device_tables(h: HostTables, dtype=jnp.float64) -> Tables:
    f = lambda a: jnp.asarray(a, dtype)  # noqa: E731
    from hepsimkernels import eloss_jax

    from dunefastsim.transport.emphysics import lar_material

    mc = eloss_jax.consts(lar_material())
    mc = type(mc)(*(f(v) for v in mc))
    return Tables(f(h.lnt0), f(h.dlnt), f(h.t_nodes), f(h.dedx), f(h.rng), f(h.sig_brem), f(h.sig_delta),
                  f(h.brem_q), f(h.delta_q), f(h.lam1), f(h.wentzel_lnq), f(h.wentzel_lna), f(h.lnk0), f(h.dlnk),
                  f(h.sig_compton), f(h.sig_pair), f(h.compton_q), f(h.pair_q), f(h.tcut), f(h.kcut), f(h.x0_mm), mc)


class Config(NamedTuple):
    n_cap: int = 2048  # B0 probe: at most 679 live particles in a 44 GeV-spectrum sample
    m_cap: int = 1 << 18
    max_iter: int = 20000
    t_track: float = 0.1  # MeV, e+- tracking cut
    k_track: float = 0.1  # MeV, photon tracking cut (photoelectric: local absorption)
    f_range: float = 0.2  # step <= f_range * residual range ...
    s_min: float = 0.1  # ... but at least s_min (mm) ...
    s_max: float = 2.0  # ... and at most s_max (mm)
    fluctuate: bool = True  # Urban loss fluctuations (B1)
    msc: str = "wentzel"  # "wentzel" (B1) or "highland" (B0)
    tsai_angles: bool = True  # brem / pair angles (B1); False: collinear (B0)


class Deposits(NamedTuple):
    pos: jax.Array  # (M, 3) mm
    de: jax.Array  # (M,) MeV
    dx: jax.Array  # (M,) mm
    dirn: jax.Array  # (M, 3)
    t: jax.Array  # (M,) ns
    tid: jax.Array  # (M,) track id
    anc: jax.Array  # (M,) ancestor (always 0 for one primary)
    pdg: jax.Array  # (M,)
    n: jax.Array  # number written
    lost_mev: jax.Array  # energy of dropped particles and deposits (overflow)
    iters: jax.Array
    peak: jax.Array  # largest number of live particles in one iteration


# --------------------------------------------------------------------------------------------------- interpolation
def _lin(tab, x0, dx, x):
    """Linear interpolation of tab (..., N) on a uniform grid at x (lanes); clamped at the ends."""
    n = tab.shape[-1]
    f = jnp.clip((x - x0) / dx, 0.0, n - 1.000001)
    i = jnp.floor(f).astype(jnp.int32)
    w = f - i
    return tab[..., i] * (1 - w) + tab[..., i + 1] * w


def _lin2(q, x0, dx, x, u):
    """Bilinear lookup in an inverse-CDF table q (N, NC) at energy coordinate x and CDF value u (lanes)."""
    n, nc = q.shape
    f = jnp.clip((x - x0) / dx, 0.0, n - 1.000001)
    i = jnp.floor(f).astype(jnp.int32)
    w = f - i
    g = jnp.clip(u, 0.0, 1.0) * (nc - 1.000001)
    j = jnp.floor(g).astype(jnp.int32)
    v = g - j
    a = q[i, j] * (1 - v) + q[i, j + 1] * v
    b = q[i + 1, j] * (1 - v) + q[i + 1, j + 1] * v
    return a * (1 - w) + b * w


def _e_tab(tab_pm, pos, lnt, x0, dx):
    """Electron/positron table value (tab_pm: (2, N)) per lane."""
    return jnp.where(pos, _lin(tab_pm[1], x0, dx, lnt), _lin(tab_pm[0], x0, dx, lnt))


def _range_inverse(tb, pos, r):
    """Kinetic energy with residual range r (inverse of the range table, linear in ln T)."""
    def inv(rt):
        return jnp.exp(jnp.interp(jnp.log(jnp.maximum(r, 1e-30)), jnp.log(rt), jnp.log(tb.t_nodes)))
    return jnp.where(pos, inv(tb.rng[1]), inv(tb.rng[0]))


# ------------------------------------------------------------------------------------------------------- geometry
def _rotate(d, cost, phi):
    """Rotate unit vectors d (lanes, 3) by polar angle acos(cost) and azimuth phi about themselves."""
    sint = jnp.sqrt(jnp.clip(1.0 - cost * cost, 0.0, 1.0))
    helper = jnp.where(jnp.abs(d[:, 2:3]) < 0.9, jnp.array([0.0, 0.0, 1.0], d.dtype), jnp.array([1.0, 0.0, 0.0], d.dtype))
    e1 = jnp.cross(d, helper)
    e1 = e1 / jnp.linalg.norm(e1, axis=1, keepdims=True)
    e2 = jnp.cross(d, e1)
    out = cost[:, None] * d + sint[:, None] * (jnp.cos(phi)[:, None] * e1 + jnp.sin(phi)[:, None] * e2)
    return out / jnp.linalg.norm(out, axis=1, keepdims=True)


def _tsai_cos(T, r1, r2, r3):
    """Geant4's modified Tsai polar angle for a lepton of kinetic energy T (G4ModifiedTsai::SampleCosTheta): u =
    -ln(r1 r2) a, a = 1.6 with probability 0.25 else 1.6/3, capped at u_max = 2 (1 + T/m); cos = 1 - 2 u^2/u_max^2."""
    umax = 2.0 * (1.0 + T / ME)
    u = -jnp.log(r1 * r2) * jnp.where(r3 < 0.25, 1.6, 1.6 / 3.0)
    u = jnp.minimum(u, umax)
    return 1.0 - 2.0 * u * u / (umax * umax)


def _isotropic(u1, u2):
    cost = 2.0 * u1 - 1.0
    sint = jnp.sqrt(jnp.clip(1.0 - cost * cost, 0.0, 1.0))
    phi = 2.0 * jnp.pi * u2
    return jnp.stack([sint * jnp.cos(phi), sint * jnp.sin(phi), cost], axis=1)


# ------------------------------------------------------------------------------------------------------- the step
class State(NamedTuple):
    kind: jax.Array
    T: jax.Array
    x: jax.Array
    d: jax.Array
    t: jax.Array
    tid: jax.Array
    next_tid: jax.Array
    dep: Deposits
    key: jax.Array


def _isum(b):
    return jnp.sum(b).astype(jnp.int32)


def _icum(b):
    return jnp.cumsum(b.astype(jnp.int32)).astype(jnp.int32)


def _step(tb: Tables, cfg: Config, st: State) -> State:
    dt = st.T.dtype
    n = cfg.n_cap
    key, ku, kn, kf1, kf2 = jax.random.split(st.key, 5)
    u = jax.random.uniform(ku, (n, 18), dt, minval=1e-12, maxval=1.0)
    z = jax.random.normal(kn, (n, 2), dt)
    kind, T, x, d = st.kind, st.T, st.x, st.d
    alive = kind >= 0
    charged = (kind == E_MINUS) | (kind == E_PLUS)
    gamma = kind == GAMMA
    pos = kind == E_PLUS
    Tsafe = jnp.maximum(T, 1e-6)
    lnt = jnp.log(Tsafe)

    # ------------------------------------------------------------------ charged particles
    sig_b = _e_tab(tb.sig_brem, pos, lnt, tb.lnt0, tb.dlnt)
    sig_d = _e_tab(tb.sig_delta, pos, lnt, tb.lnt0, tb.dlnt)
    sig = sig_b + sig_d
    r = jnp.exp(_e_tab(jnp.log(jnp.maximum(tb.rng, 1e-30)), pos, lnt, tb.lnt0, tb.dlnt))
    s_int = jnp.where(sig > 0, -jnp.log(u[:, 0]) / jnp.maximum(sig, 1e-30), jnp.inf)
    s_lim = jnp.clip(cfg.f_range * r, cfg.s_min, cfg.s_max)
    s = jnp.minimum(jnp.minimum(s_int, s_lim), r)
    ranged = s >= r
    T_mean = jnp.minimum(jnp.where(ranged, 0.0, _range_inverse(tb, pos, r - s)), T)
    if cfg.fluctuate:
        from hepsimkernels import eloss_jax

        mean = T - T_mean
        fm = eloss_jax.sample_loss_electron(kf1, tb.mc, Tsafe, jnp.maximum(s, 1e-9), tb.tcut, mean, positron=False)
        fp = eloss_jax.sample_loss_electron(kf2, tb.mc, Tsafe, jnp.maximum(s, 1e-9), tb.tcut, mean, positron=True)
        loss = jnp.where(ranged, T, jnp.clip(jnp.where(pos, fp, fm), 0.0, T))
        loss = jnp.where(jnp.isfinite(loss), loss, mean)
        T_post = jnp.where(charged, T - loss, T_mean)
    else:
        T_post = T_mean
    de_c = T - T_post
    hit = charged & (s_int < jnp.minimum(s_lim, r))
    lnp = jnp.log(jnp.maximum(T_post, 1e-6))
    sb_p = _e_tab(tb.sig_brem, pos, lnp, tb.lnt0, tb.dlnt)
    sd_p = _e_tab(tb.sig_delta, pos, lnp, tb.lnt0, tb.dlnt)
    p_brem = sb_p / jnp.maximum(sb_p + sd_p, 1e-30)
    is_brem = hit & (u[:, 1] < p_brem) & (T_post > tb.kcut)
    wmax = jnp.where(pos, T_post, 0.5 * T_post)
    is_delta = hit & ~is_brem & (wmax > tb.tcut)
    # brem photon energy
    xb = jnp.where(pos, _lin2(tb.brem_q[1], tb.lnt0, tb.dlnt, lnp, u[:, 2]),
                   _lin2(tb.brem_q[0], tb.lnt0, tb.dlnt, lnp, u[:, 2]))
    k_b = tb.kcut * jnp.exp(xb * jnp.log(jnp.maximum(T_post, tb.kcut * 1.0001) / tb.kcut))
    k_b = jnp.minimum(k_b, T_post)
    # delta energy and two-body angles
    xd = jnp.where(pos, _lin2(tb.delta_q[1], tb.lnt0, tb.dlnt, lnp, u[:, 2]),
                   _lin2(tb.delta_q[0], tb.lnt0, tb.dlnt, lnp, u[:, 2]))
    w_d = tb.tcut * jnp.exp(xd * jnp.log(jnp.maximum(wmax, tb.tcut * 1.0001) / tb.tcut))
    w_d = jnp.minimum(w_d, wmax)
    Tp = jnp.maximum(T_post, 1e-9)
    cos_d = jnp.sqrt(jnp.clip(w_d * (Tp + 2 * ME) / (Tp * (w_d + 2 * ME)), 0.0, 1.0))
    T_left = jnp.maximum(Tp - w_d, 1e-9)
    cos_p = jnp.sqrt(jnp.clip(T_left * (Tp + 2 * ME) / (Tp * (T_left + 2 * ME)), 0.0, 1.0))
    phi_d = 2.0 * jnp.pi * u[:, 3]
    # multiple scattering over the step, at the mean energy
    Tm = 0.5 * (T + T_post)
    pc = jnp.sqrt(jnp.maximum(Tm * (Tm + 2 * ME), 1e-12))
    beta = pc / (Tm + ME)
    if cfg.msc == "wentzel":
        lam = jnp.exp(_e_tab(jnp.log(tb.lam1), pos, jnp.log(jnp.maximum(Tm, 1e-6)), tb.lnt0, tb.dlnt))
        q = -jnp.expm1(-jnp.maximum(s, 1e-12) / lam)
        lnq = jnp.log(jnp.clip(q, 1e-10, 0.999))
        a_scr = jnp.exp(jnp.interp(lnq, tb.wentzel_lnq, tb.wentzel_lna))
        uw = u[:, 10]
        one_minus = jnp.clip(2.0 * a_scr * uw / (1.0 - uw + a_scr), 0.0, 2.0)
        d_msc = _rotate(d, 1.0 - one_minus, 2.0 * jnp.pi * u[:, 11])
    else:
        xs = jnp.maximum(s, 1e-9) / tb.x0_mm
        theta0 = 13.6 / (beta * pc) * jnp.sqrt(xs) * jnp.maximum(1.0 + 0.038 * jnp.log(xs / beta ** 2), 0.1)
        thx, thy = theta0 * z[:, 0], theta0 * z[:, 1]
        th = jnp.minimum(jnp.sqrt(thx * thx + thy * thy), jnp.pi)
        d_msc = _rotate(d, jnp.cos(th), jnp.arctan2(thy, thx))
    chord = 0.5 * (d + d_msc)  # mean direction: lateral displacement and path-length correction
    T_after = jnp.where(is_brem, T_post - k_b, jnp.where(is_delta, T_post - w_d, T_post))
    d_after = jnp.where(is_delta[:, None], _rotate(d_msc, cos_p, phi_d), d_msc)
    x_after = x + chord * s[:, None]
    # a lane that emitted a secondary this step keeps its particle alive one more step, so a positron never needs
    # three secondary slots (emission + two annihilation photons) at once
    stop = charged & (ranged | ((T_after < cfg.t_track) & ~(is_brem | is_delta)))
    e_local = jnp.where(stop, T_after, 0.0)
    c_de = jnp.where(charged, de_c + e_local, 0.0)
    c_mid = x + chord * (0.5 * s)[:, None]
    t_after_c = st.t + s / (beta * C_MM_NS)

    # ------------------------------------------------------------------ photons
    lnk = jnp.log(Tsafe)
    sc = _lin(tb.sig_compton, tb.lnk0, tb.dlnk, lnk)
    sp = _lin(tb.sig_pair, tb.lnk0, tb.dlnk, lnk)
    absorb = gamma & (T < cfg.k_track)
    s_g = -jnp.log(u[:, 4]) / jnp.maximum(sc + sp, 1e-30)
    s_g = jnp.where(absorb, 0.0, s_g)
    x_g = x + d * s_g[:, None]
    is_pair = gamma & ~absorb & (u[:, 5] < sp / jnp.maximum(sc + sp, 1e-30)) & (T > 2 * ME)
    is_comp = gamma & ~absorb & ~is_pair
    eps_c = _lin2(tb.compton_q, tb.lnk0, tb.dlnk, lnk, u[:, 6])
    k_out = eps_c * T
    T_ce = T - k_out
    kap = T / ME
    cos_g = jnp.clip(1.0 - (1.0 - eps_c) / (kap * jnp.maximum(eps_c, 1e-12)), -1.0, 1.0)
    phi_g = 2.0 * jnp.pi * u[:, 7]
    d_g = _rotate(d, cos_g, phi_g)
    pe = T[:, None] * d - k_out[:, None] * d_g
    d_ce = pe / jnp.maximum(jnp.linalg.norm(pe, axis=1, keepdims=True), 1e-30)
    eps_p = _lin2(tb.pair_q, tb.lnk0, tb.dlnk, lnk, u[:, 6])
    T_pm = jnp.maximum(eps_p * T - ME, 0.0)
    T_pp = jnp.maximum((1.0 - eps_p) * T - ME, 0.0)
    t_after_g = st.t + s_g / C_MM_NS

    # ------------------------------------------------------------------ deposits (at most one per lane)
    g_de = jnp.where(absorb, T, 0.0)
    de = jnp.where(charged, c_de, g_de)
    has_dep = alive & (de > 0)
    dpos = jnp.where(charged[:, None], c_mid, x)
    ddx = jnp.where(charged, jnp.maximum(s, 1e-6), 1e-3)
    dpdg = jnp.where(kind == E_PLUS, -11, jnp.where(kind == E_MINUS, 11, 22))
    dep = st.dep
    off = dep.n + _icum(has_dep) - 1
    idx = jnp.where(has_dep, off, cfg.m_cap)  # out of range: dropped by mode="drop"
    over = has_dep & (off >= cfg.m_cap)
    dep = dep._replace(
        pos=dep.pos.at[idx].set(dpos, mode="drop"), de=dep.de.at[idx].set(de, mode="drop"),
        dx=dep.dx.at[idx].set(ddx, mode="drop"), dirn=dep.dirn.at[idx].set(d, mode="drop"),
        t=dep.t.at[idx].set(st.t, mode="drop"), tid=dep.tid.at[idx].set(st.tid, mode="drop"),
        anc=dep.anc.at[idx].set(jnp.zeros_like(st.tid), mode="drop"), pdg=dep.pdg.at[idx].set(dpdg, mode="drop"),
        n=jnp.minimum(dep.n + _isum(has_dep), cfg.m_cap).astype(jnp.int32),
        lost_mev=dep.lost_mev + jnp.sum(jnp.where(over, de, 0.0)), iters=dep.iters + 1,
        peak=jnp.maximum(dep.peak, _isum(alive)))

    # ------------------------------------------------------------------ next buffer: survivors, then secondaries
    surv = (charged & ~stop) | is_comp
    s_kind = jnp.where(surv, kind, DEAD)
    s_T = jnp.where(charged, T_after, k_out)
    s_x = jnp.where(charged[:, None], x_after, x_g)
    s_d = jnp.where(charged[:, None], d_after, d_g)
    s_t = jnp.where(charged, t_after_c, t_after_g)
    annih = (kind == E_PLUS) & stop
    # secondary 1: brem photon | delta electron | Compton electron | pair electron | annihilation photon
    n1 = is_brem | is_delta | is_comp | is_pair | annih
    k1 = jnp.where(is_brem | annih, GAMMA, E_MINUS)
    T1 = jnp.where(is_brem, k_b, jnp.where(is_delta, w_d, jnp.where(is_comp, T_ce, jnp.where(is_pair, T_pm, ME))))
    iso = _isotropic(u[:, 8], u[:, 9])
    d1 = jnp.where(is_delta[:, None], _rotate(d_msc, cos_d, phi_d + jnp.pi),
                   jnp.where(is_comp[:, None], d_ce, jnp.where(annih[:, None], iso, d)))
    if cfg.tsai_angles:
        d_brem = _rotate(d_msc, _tsai_cos(T_post, u[:, 12], u[:, 13], u[:, 14]), 2.0 * jnp.pi * u[:, 15])
        phi_pair = 2.0 * jnp.pi * u[:, 15]
        d_pm = _rotate(d, _tsai_cos(T_pm, u[:, 12], u[:, 13], u[:, 14]), phi_pair)
        d_pp = _rotate(d, _tsai_cos(T_pp, u[:, 16], u[:, 17], u[:, 10]), phi_pair + jnp.pi)  # photon lanes: u10 free
    else:
        d_brem, d_pm, d_pp = d_msc, d, d
    d1 = jnp.where(is_brem[:, None], d_brem, jnp.where(is_pair[:, None], d_pm, d1))
    x1 = jnp.where(charged[:, None], x_after, x_g)
    t1 = jnp.where(charged, t_after_c, t_after_g)
    # secondary 2: pair positron | second annihilation photon
    n2 = is_pair | annih
    k2 = jnp.where(is_pair, E_PLUS, GAMMA)
    T2 = jnp.where(is_pair, T_pp, ME)
    d2 = jnp.where(annih[:, None], -iso, d_pp)
    # Compton photon below the photon cut keeps flying one more step and is absorbed there (no second deposit now)
    cnt_s, cnt_1, cnt_2 = _isum(surv), _isum(n1), _isum(n2)
    i_s = jnp.where(surv, _icum(surv) - 1, n)
    i_1 = jnp.where(n1, cnt_s + _icum(n1) - 1, n)
    i_2 = jnp.where(n2, cnt_s + cnt_1 + _icum(n2) - 1, n)
    i_1 = jnp.where(i_1 < n, i_1, n)
    i_2 = jnp.where(i_2 < n, i_2, n)
    lost = (jnp.sum(jnp.where(n1 & (i_1 >= n), T1 + jnp.where(k1 == E_PLUS, 2 * ME, 0.0), 0.0))
            + jnp.sum(jnp.where(n2 & (i_2 >= n), T2 + jnp.where(k2 == E_PLUS, 2 * ME, 0.0), 0.0)))
    tid1 = st.next_tid + _icum(n1) - 1
    tid2 = st.next_tid + cnt_1 + _icum(n2) - 1

    def scatter(base, vs, vals):
        out = base
        for v, i in zip(vals, vs):
            out = out.at[i].set(v, mode="drop")
        return out

    nk = scatter(jnp.full((n,), DEAD, kind.dtype), (i_s, i_1, i_2), (s_kind, k1, k2))
    nT = scatter(jnp.zeros((n,), dt), (i_s, i_1, i_2), (s_T, T1, T2))
    nx = scatter(jnp.zeros((n, 3), dt), (i_s, i_1, i_2), (s_x, x1, x1))
    nd = scatter(jnp.zeros((n, 3), dt).at[:, 2].set(1.0), (i_s, i_1, i_2), (s_d, d1, d2))
    ntm = scatter(jnp.zeros((n,), dt), (i_s, i_1, i_2), (s_t, t1, t1))
    ntid = scatter(jnp.full((n,), -1, st.tid.dtype), (i_s, i_1, i_2), (st.tid, tid1, tid2))
    dep = dep._replace(lost_mev=dep.lost_mev + lost)
    return State(nk, nT, nx, nd, ntm, ntid, st.next_tid + cnt_1 + cnt_2, dep, key)


def shower(tb: Tables, cfg: Config, key, energy_mev, vertex_mm, direction, primary=E_MINUS) -> Deposits:
    """Deposits of one primary electron (or positron / photon) in infinite liquid argon."""
    dt = tb.dlnt.dtype
    n, m = cfg.n_cap, cfg.m_cap
    d0 = jnp.asarray(direction, dt)
    d0 = d0 / jnp.linalg.norm(d0)
    kind = jnp.full((n,), DEAD, jnp.int32).at[0].set(primary)
    T = jnp.zeros((n,), dt).at[0].set(jnp.asarray(energy_mev, dt))
    x = jnp.zeros((n, 3), dt).at[0].set(jnp.asarray(vertex_mm, dt))
    d = jnp.zeros((n, 3), dt).at[:, 2].set(1.0).at[0].set(d0)
    dep = Deposits(jnp.zeros((m, 3), dt), jnp.zeros((m,), dt), jnp.zeros((m,), dt), jnp.zeros((m, 3), dt),
                   jnp.zeros((m,), dt), jnp.full((m,), -1, jnp.int32), jnp.full((m,), -1, jnp.int32),
                   jnp.zeros((m,), jnp.int32), jnp.array(0, jnp.int32), jnp.array(0.0, dt), jnp.array(0, jnp.int32),
                   jnp.array(0, jnp.int32))
    st = State(kind, T, x, d, jnp.zeros((n,), dt), jnp.full((n,), -1, jnp.int32).at[0].set(0),
               jnp.array(1, jnp.int32), dep, key)
    st = lax.while_loop(lambda s: jnp.any(s.kind >= 0) & (s.dep.iters < cfg.max_iter),
                        lambda s: _step(tb, cfg, s), st)
    # anything still alive at max_iter counts as lost
    left = jnp.sum(jnp.where(st.kind >= 0, st.T + jnp.where(st.kind == E_PLUS, 2 * ME, 0.0), 0.0))
    return st.dep._replace(lost_mev=st.dep.lost_mev + left)


def make_shower_fn(tb: Tables, cfg: Config):
    """jit-compiled ``(key, energy, vertex, direction) -> Deposits`` for one electron."""
    return jax.jit(lambda key, e, v, d: shower(tb, cfg, key, e, v, d))


def to_numpy(dep: Deposits):
    """Valid deposits as NumPy arrays (positions, de, dx, direction, time, tid, pdg)."""
    n = int(dep.n)
    g = lambda a: np.asarray(a)[:n]  # noqa: E731
    return {"positions_mm": g(dep.pos), "de": g(dep.de), "dx": g(dep.dx), "dir": g(dep.dirn), "t": g(dep.t),
            "tid": g(dep.tid), "pdg": g(dep.pdg), "lost_mev": float(dep.lost_mev), "iters": int(dep.iters),
            "peak": int(dep.peak)}
