"""Host-side (NumPy) electromagnetic physics for option B in liquid argon: cross sections and sampling tables.

Electrons and positrons up to 1 GeV come from ``hepsimkernels`` (Geant4 11.3.2-matched kernels, validated in
mu2e-fastsim against Geant4 and NIST ESTAR): Moller/Bhabha restricted loss and delta cross sections (``eioni``),
Seltzer-Berger restricted radiative loss and photon cross sections (``ebrem``). Written here, from published formulas:

- bremsstrahlung above 1 GeV (``ebrem`` refuses T > 1 GeV): Tsai's complete-screening cross section (Y.-S. Tsai, Rev.
  Mod. Phys. 46 (1974) 815, eq. 3.83) with the Coulomb correction and the Ter-Mikaelian dielectric factor
  1 / (1 + k_p^2 / k^2); LPM suppression is negligible in argon (E_LPM ~ 100 TeV);
- Compton scattering on free electrons: Klein-Nishina;
- pair production in the nuclear and electron fields: Bethe-Heitler with the Butcher-Messel screening functions
  Phi1, Phi2, the Davies-Bethe-Maximon Coulomb correction above 50 MeV and the electron-field term xi(Z) (the form
  of Geant4's G4BetheHeitlerModel sampling, Butcher & Messel, Nucl. Phys. 20 (1960) 15);
- photoelectric absorption is not tabulated: photons below the tracking cut are absorbed where they are.

Units: MeV, mm. Sampling tables are inverse CDFs on (ln E node, CDF node) grids, linear in both directions, so a
sample is a smooth function of the energy and the uniform number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ME = 0.51099895  # MeV
RE_MM = 2.8179403262e-12  # classical electron radius, mm
LAMBDAE_MM = 3.8615926796e-10  # reduced Compton wavelength, mm
ALPHA = 1.0 / 137.035999
N_A = 6.02214076e23
MIGDAL = 4.0 * np.pi * RE_MM * LAMBDAE_MM ** 2  # mm^3: k_p^2 = MIGDAL n_e E_tot^2

# Liquid argon (G4_lAr): density, mean excitation energy, Sternheimer density-effect parameters
# (-C, X0, X1, a, m, delta0) for liquid argon from Sternheimer, Berger & Seltzer, At. Data Nucl. Data Tables 30 (1984).
Z_AR, A_AR = 18, 39.948
RHO_LAR = 1.396
I_LAR_EV = 188.0
STERNHEIMER_LAR = (5.2146, 0.2, 3.0, 0.19559, 3.0, 0.0)

# Photoelectric effect in argon as Geant4's option-0 physics computes it (G4LivermorePhotoElectricModel, Geant4 11.4.0,
# ComputeCrossSectionPerAtom), from G4EMLOW 8.8 livermore/phot_epics2014 (EPICS2014) for Z = 18: the last-shell rows
# of pe-high-18.dat and pe-low-18.dat (sigma = sum_i a_i / E^(i+1), barn and MeV, above 59.5662 keV and above
# 6.30957 keV) and pe-cs-18.dat (sigma E^3 in barn MeV^3 from the K edge to 6.6 keV; Geant4 splines it, linear here
# on its 28 nodes). Geant4 clamps the energy below at the last shell's binding energy (14.43 eV).
PE_AR_HIGH = (0.0595662, (0.00176596, 0.00233442, 0.00115495, 0.00047547, -3.32075e-05, 8.61093e-07))
PE_AR_LOW = (0.00630957, (1.19022, -0.155437, 0.0105813, -0.000100674, 6.7825e-07, -1.75352e-09))
PE_AR_EMIN = 1.443e-05
PE_AR_KEDGE = 0.0031776
PE_AR_SIGMA_E3 = np.array([
    (0.0031776, 0.00280166),
    (0.00331131, 0.00292244),
    (0.00334965, 0.00295718),
    (0.00335, 0.00295749),
    (0.00338844, 0.00297437),
    (0.00358922, 0.00306119),
    (0.0036, 0.00306578),
    (0.0036789, 0.00309253),
    (0.00380189, 0.00313332),
    (0.00384592, 0.00314775),
    (0.00389045, 0.00316221),
    (0.00416869, 0.00325059),
    (0.0042658, 0.00328061),
    (0.00436516, 0.00331073),
    (0.00446684, 0.00334116),
    (0.00467735, 0.00340282),
    (0.00484172, 0.00344782),
    (0.0049545, 0.00347814),
    (0.00506991, 0.00350852),
    (0.00512861, 0.00352103),
    (0.005188, 0.00353358),
    (0.00562341, 0.00362276),
    (0.0057544, 0.00364868),
    (0.00588844, 0.00367457),
    (0.00595662, 0.0036876),
    (0.0060256, 0.00370068),
    (0.00630957, 0.00375344),
    (0.00660693, 0.0037962)])
N_ATOM_MM3 = RHO_LAR * N_A / A_AR * 1e-3  # atoms per mm^3
N_E_MM3 = Z_AR * N_ATOM_MM3

T_SB_MAX = 1000.0  # hepsimkernels.ebrem's upper limit


def lar_material():
    """The G4_lAr material, registered with hepsimkernels (idempotent)."""
    from hepsimkernels import materials as mats

    m = mats.Material("G4_lAr", RHO_LAR, I_LAR_EV, {"Ar": 1.0}, sternheimer=STERNHEIMER_LAR)
    try:
        return mats.get("G4_lAr")
    except KeyError:
        mats.register(m)
        return mats.get("G4_lAr")


# --------------------------------------------------------------------------------------------------------- Coulomb
def coulomb_fc(z=Z_AR):
    """Davies-Bethe-Maximon Coulomb correction f_c(Z)."""
    a2 = (ALPHA * z) ** 2
    return a2 * (1.0 / (1.0 + a2) + 0.20206 - 0.0369 * a2 + 0.0083 * a2 ** 2 - 0.002 * a2 ** 3)


def radiation_length_g_cm2(z=Z_AR, a=A_AR):
    """Tsai's radiation length: 1/X0 = 4 alpha r_e^2 N_A/A [Z^2 (L_rad - f_c) + Z L'_rad]."""
    l_rad, l_rad_p = np.log(184.15 * z ** (-1 / 3)), np.log(1194.0 * z ** (-2 / 3))
    re_cm = RE_MM / 10.0
    inv = 4 * ALPHA * re_cm ** 2 * N_A / a * (z * z * (l_rad - coulomb_fc(z)) + z * l_rad_p)
    return 1.0 / inv


# ------------------------------------------------------------------------------------------- brem above 1 GeV (Tsai)
def tsai_dsigma_dk(k, t, z=Z_AR):
    """Complete-screening bremsstrahlung dsigma/dk per atom [mm^2/MeV] with the dielectric factor; k < T."""
    k = np.asarray(k, float)
    etot = t + ME
    y = k / etot
    l_rad, l_rad_p = np.log(184.15 * z ** (-1 / 3)), np.log(1194.0 * z ** (-2 / 3))
    body = ((4 / 3 - 4 / 3 * y + y * y) * (z * z * (l_rad - coulomb_fc(z)) + z * l_rad_p)
            + (1 - y) * (z * z + z) / 9.0)
    kp2 = MIGDAL * N_E_MM3 * etot ** 2
    return 4 * ALPHA * RE_MM ** 2 * body / k * (k * k / (k * k + kp2))


def _ln_integral(f, a, b, n=400):
    """Integral of f(k) dk from a to b (arrays broadcast) on a log grid (trapezoid in ln k)."""
    u = np.linspace(0.0, 1.0, n)
    k = a[..., None] * (b[..., None] / a[..., None]) ** u
    v = f(k) * k
    return np.trapezoid(v, np.log(k), axis=-1)


def tsai_xsec_per_mm(t, kmin):
    """Photons above kmin per mm (T array)."""
    t = np.atleast_1d(np.asarray(t, float))
    kmin = np.broadcast_to(np.asarray(kmin, float), t.shape)
    out = np.where(t > kmin, N_ATOM_MM3 * _ln_integral(lambda k: tsai_dsigma_dk(k, t[..., None]),
                                                        np.minimum(kmin, t), t), 0.0)
    return out


def tsai_dedx_restricted(t, kcut):
    """Radiative loss to photons below kcut [MeV/mm] (T array)."""
    t = np.atleast_1d(np.asarray(t, float))
    kmax = np.minimum(kcut, t)
    kmin = np.full_like(t, 1e-6)
    return N_ATOM_MM3 * _ln_integral(lambda k: k * tsai_dsigma_dk(k, t[..., None]), kmin, kmax)


# ------------------------------------------------------------------------------------------------------- Compton
def photoelectric_xsec_per_atom_barn(k):
    """Photoelectric cross section of argon [barn per atom] at photon energy k [MeV] (G4LivermorePhotoElectricModel,
    EPICS2014). Above the K edge only: below it (3.18 keV) photons are absorbed where they are anyway."""
    k = np.maximum(np.asarray(k, float), PE_AR_EMIN)
    x1 = 1.0 / k

    def par(c):
        return x1 * (c[0] + x1 * (c[1] + x1 * (c[2] + x1 * (c[3] + x1 * (c[4] + x1 * c[5])))))

    tab = x1 ** 3 * np.interp(k, PE_AR_SIGMA_E3[:, 0], PE_AR_SIGMA_E3[:, 1])
    return np.where(k >= PE_AR_HIGH[0], par(PE_AR_HIGH[1]), np.where(k >= PE_AR_LOW[0], par(PE_AR_LOW[1]), tab))


def klein_nishina_xsec_per_electron(k):
    """Total Klein-Nishina cross section per free electron [mm^2]."""
    x = np.asarray(k, float) / ME
    l12 = np.log1p(2 * x)
    return 2 * np.pi * RE_MM ** 2 * ((1 + x) / x ** 2 * (2 * (1 + x) / (1 + 2 * x) - l12 / x)
                                    + l12 / (2 * x) - (1 + 3 * x) / (1 + 2 * x) ** 2)


def klein_nishina_deps(eps, k):
    """dsigma/d(eps) up to a constant, eps = k'/k in [1/(1+2 kappa), 1]."""
    x = k / ME
    cost = 1.0 - (1.0 - eps) / (x * eps)
    sin2 = np.clip(1.0 - cost * cost, 0.0, 1.0)
    return (1.0 / eps + eps) * (1.0 - eps * sin2 / (1.0 + eps * eps))


# ---------------------------------------------------------------------------------------------- pair (Bethe-Heitler)
def _phi12(delta):
    d = np.asarray(delta, float)
    big = 21.12 - 4.184 * np.log(d + 0.952)
    phi1 = np.where(d <= 1.0, 20.867 - 3.242 * d + 0.625 * d * d, big)
    phi2 = np.where(d <= 1.0, 20.209 - 1.930 * d - 0.086 * d * d, big)
    return phi1, phi2


def pair_deps_per_atom(eps, k, z=Z_AR):
    """dsigma/d(eps) per atom [mm^2] for the electron energy fraction eps (total energy eps k)."""
    eps = np.asarray(eps, float)
    k = np.asarray(k, float)
    z3 = z ** (1 / 3)
    fz = 8.0 / 3.0 * np.log(z) + np.where(k > 50.0, 8.0 * coulomb_fc(z), 0.0)
    xi = np.log(1440.0 / z3 ** 2) / (np.log(183.0 / z3) - coulomb_fc(z))
    e1 = np.clip(eps * (1.0 - eps), 1e-300, None)
    delta = 136.0 * ME / (z3 * k * e1)
    phi1, phi2 = _phi12(delta)
    val = ((eps * eps + (1 - eps) ** 2) * (phi1 - fz / 2) + (2.0 / 3.0) * e1 * (phi2 - fz / 2))
    return ALPHA * RE_MM ** 2 * z * (z + xi) * np.clip(val, 0.0, None)


def pair_xsec_per_mm(k, n=801):
    k = np.atleast_1d(np.asarray(k, float))
    out = np.zeros_like(k)
    for i, kk in enumerate(k):
        if kk <= 2 * ME * 1.0001:
            continue
        e0 = ME / kk
        eps = np.linspace(e0, 1 - e0, n)
        out[i] = N_ATOM_MM3 * np.trapezoid(pair_deps_per_atom(eps, kk), eps)
    return out


# ------------------------------------------------------------------------------------------------------- tables
@dataclass
class HostTables:
    """Everything the cascade needs, float64 NumPy. e+- quantities have a leading axis (0: e-, 1: e+)."""

    lnt0: float
    dlnt: float
    t_nodes: np.ndarray  # (NT,)
    dedx: np.ndarray  # (2, NT) restricted ionisation + radiative loss [MeV/mm]
    rng: np.ndarray  # (2, NT) restricted CSDA range [mm]
    sig_brem: np.ndarray  # (2, NT) photons above kcut [1/mm]
    sig_delta: np.ndarray  # (2, NT) deltas above tcut [1/mm]
    sig_annih: np.ndarray  # (2, NT) two-photon annihilation in flight [1/mm] (row 0, e-: zero)
    brem_q: np.ndarray  # (2, NT, NC) x = ln(k/kcut)/ln(T/kcut) at CDF nodes
    delta_q: np.ndarray  # (2, NT, NC) x = ln(w/tcut)/ln(wmax/tcut) at CDF nodes
    lam1: np.ndarray  # (2, NT) first transport mean free path [mm] (hepsimkernels.emsc, Mott-corrected)
    msc_k: np.ndarray  # (2, NT) screened-Rutherford prefactor n K Z / (beta^2 p^2) [1/mm] (emsc._parts)
    msc_s: np.ndarray  # (2, NT) screening s in u = 1 - cos theta
    msc_ue: np.ndarray  # (2, NT) largest u of scattering on atomic electrons (transfers above tcut are deltas)
    msc_env: np.ndarray  # (2, NT) upper bound of the Mott factor (1 for e+)
    mott_coef: np.ndarray  # (5, 6) G4ScreeningMottCrossSection fit for Z = 18 (e- Mott / Rutherford)
    msc_ffa: float  # nuclear form factor constant f_A = 6.937e-6 A^0.54 [MeV^-2] (hard collisions above 100 MeV)
    wentzel_lnq: np.ndarray  # (NQ,) ln <1 - cos> nodes of the Wentzel screening table
    wentzel_lna: np.ndarray  # (NQ,) ln A with <1 - cos>_Wentzel(A) = q
    lnk0: float
    dlnk: float
    k_nodes: np.ndarray  # (NK,)
    sig_compton: np.ndarray  # (NK,) [1/mm]
    sig_pair: np.ndarray  # (NK,) [1/mm]
    sig_photo: np.ndarray  # (NK,) photoelectric [1/mm]
    compton_q: np.ndarray  # (NK, NC) eps = k'/k
    pair_q: np.ndarray  # (NK, NC) electron total-energy fraction
    tcut: float
    kcut: float
    x0_mm: float

    def as_dict(self):
        return dict(self.__dict__)


def _inverse_cdf(x, pdf, c_nodes):
    """Quantiles of a density sampled at increasing x (rows), at CDF values c_nodes."""
    cdf = np.concatenate([np.zeros(pdf.shape[:-1] + (1,)),
                          np.cumsum(0.5 * (pdf[..., 1:] + pdf[..., :-1]) * np.diff(x, axis=-1), axis=-1)], axis=-1)
    tot = cdf[..., -1:]
    cdf = np.where(tot > 0, cdf / np.where(tot > 0, tot, 1.0), np.linspace(0, 1, x.shape[-1]))
    cdf = np.maximum.accumulate(cdf + 1e-15 * np.arange(x.shape[-1]), axis=-1)
    cdf /= cdf[..., -1:]
    out = np.empty(pdf.shape[:-1] + (len(c_nodes),))
    for idx in np.ndindex(pdf.shape[:-1]):
        out[idx] = np.interp(c_nodes, cdf[idx], x[idx] if x.ndim > 1 else x)
    return out


def wentzel_mean(a):
    """<1 - cos theta> of the Wentzel distribution p(mu) ~ 1 / (1 - mu + 2A)^2 on [-1, 1]."""
    a = np.asarray(a, float)
    return 2.0 * a * ((1.0 + a) * np.log1p(1.0 / a) - 1.0)


def wentzel_table(nq=601, qmin=1e-10, qmax=0.999):
    """ln q -> ln A table: the screening A whose Wentzel distribution has <1 - cos> = q (bisection in ln A). Nodes
    are log-spaced in q up to 0.5 and log-spaced in 1 - q above (A grows fast as q -> 1)."""
    q = np.concatenate([np.geomspace(qmin, 0.5, nq), 1.0 - np.geomspace(0.5, 1.0 - qmax, nq // 2)[1:]])
    nq = len(q)
    lo, hi = np.full(nq, -60.0), np.full(nq, 12.0)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f = wentzel_mean(np.exp(mid)) - q
        lo, hi = np.where(f < 0, mid, lo), np.where(f < 0, hi, mid)
    return np.log(q), 0.5 * (lo + hi)


KCUT_REFERENCE = 0.00441638  # MeV: the reference's gamma production threshold in LAr (500 um range cut, its log)


def build_tables(tcut=0.22, kcut=0.1, t_min=0.05, t_max=1e5, nt=241, k_min=0.002, nk=281, nc=129,
                 nx=129):
    """Tables for liquid argon with delta production cut ``tcut`` and photon cut ``kcut`` (MeV).

    The Geant4 reference (DUNE ND production edep-sim settings, 500 um range cut) prints its LAr thresholds: e- 220.878
    keV, e+ 215.258 keV, gamma 4.41638 keV. tcut 0.22 MeV sets where the fluctuated restricted loss ends, so it
    matters for the dE/dx spread (0.35 MeV gave a 10-20 % wider fixed-length spread). kcut sets which bremsstrahlung
    photons fly instead of depositing along the track. It stays at 0.1 MeV: the reference's 4.42 keV
    (``KCUT_REFERENCE``, with photons tracked down to 4 keV, ``Config.k_track``) left deposited fraction, depth and
    radial containment unchanged against Geant4 (1 GeV electrons, 2026-11-22) and cost 30 % more time. Photon tables
    run from k_min (2 keV) with Klein-Nishina Compton, Bethe-Heitler pair and the Livermore photoelectric effect (no
    Rayleigh scattering), so a lower kcut / k_track is a setting away."""
    from hepsimkernels import ebrem, eioni, emsc

    mat = lar_material()
    t_nodes = np.geomspace(t_min, t_max, nt)
    lnt0, dlnt = np.log(t_min), np.log(t_max / t_min) / (nt - 1)
    sb = t_nodes <= T_SB_MAX
    c_nodes = np.linspace(0.0, 1.0, nc)
    x = np.linspace(0.0, 1.0, nx)

    dedx = np.zeros((2, nt))
    sig_b = np.zeros((2, nt))
    sig_d = np.zeros((2, nt))
    brem_q = np.zeros((2, nt, nc))
    delta_q = np.zeros((2, nt, nc))
    for ip, pos in enumerate((False, True)):
        ion = eioni.dedx_restricted(mat, t_nodes, tcut, positron=pos)
        rad = np.zeros(nt)
        rad[sb] = ebrem.dedx_restricted(mat, t_nodes[sb], kcut, positron=pos)
        rad[~sb] = tsai_dedx_restricted(t_nodes[~sb], kcut)
        dedx[ip] = ion + rad
        sig_d[ip] = eioni.xsec_per_mm(mat, t_nodes, tcut, positron=pos)
        sig_b[ip, sb] = ebrem.xsec_per_mm(mat, t_nodes[sb], kcut, positron=pos)
        sig_b[ip, ~sb] = tsai_xsec_per_mm(t_nodes[~sb], kcut)
        # brem photon energy: CDF from sigma(> k) on k = kcut (T/kcut)^x. Seltzer-Berger: sigma(> k) on a global
        # log k grid (one vectorised call per k), interpolated in ln k per T node.
        tsb = t_nodes[sb & (t_nodes > kcut * 1.001)]
        kg = np.geomspace(kcut, T_SB_MAX, 241)
        sig_kg = np.array([ebrem.xsec_per_mm(mat, tsb, kk, positron=pos) for kk in kg])  # (NKg, NTsb)
        for i, t in enumerate(t_nodes):
            if t <= kcut * 1.001:
                brem_q[ip, i] = c_nodes
                continue
            ks = kcut * (t / kcut) ** x
            if t <= T_SB_MAX:
                j = int(np.searchsorted(tsb, t))
                col = sig_kg[:, j]
                ok = kg < t
                above = np.interp(np.log(ks), np.log(np.append(kg[ok], t)), np.append(col[ok], 0.0))
            else:
                above = np.array([tsai_xsec_per_mm([t], kk)[0] for kk in ks[:-1]] + [0.0])
            cdf = 1.0 - above / max(above[0], 1e-300)
            cdf = np.maximum.accumulate(cdf + 1e-12 * np.arange(nx))
            brem_q[ip, i] = np.interp(c_nodes, cdf / cdf[-1], x)
        # delta energy: CDF from sigma(> w) on w = tcut (wmax/tcut)^x
        wmax = eioni.max_secondary(t_nodes, positron=pos)
        for i, t in enumerate(t_nodes):
            if wmax[i] <= tcut * 1.001:
                delta_q[ip, i] = c_nodes
                continue
            ws = tcut * (wmax[i] / tcut) ** x
            above = np.array([eioni.xsec_per_mm(mat, [t], w, positron=pos)[0] for w in ws[:-1]] + [0.0])
            cdf = 1.0 - above / max(above[0], 1e-300)
            cdf = np.maximum.accumulate(cdf + 1e-12 * np.arange(nx))
            delta_q[ip, i] = np.interp(c_nodes, cdf / cdf[-1], x)

    lam1 = np.stack([1.0 / np.maximum(emsc.transport_xsec(mat, t_nodes, 2.0, tcut, positron=pos), 1e-300)
                     for pos in (False, True)])
    # mixed multiple scattering (v0.0.7): screened-Rutherford pieces of hepsimkernels.emsc per species and energy
    # (one element: argon), so the device can split soft and hard collisions at any boundary u_c
    msc_k, msc_s, msc_ue, msc_env = (np.zeros((2, nt)) for _ in range(4))
    for ip, pos in enumerate((False, True)):
        z_, a_, s_, k_, ue_ = emsc._parts(mat, t_nodes, tcut, pos)
        msc_k[ip], msc_s[ip], msc_ue[ip] = k_[0], s_[0], ue_
        msc_env[ip] = emsc.mott_envelope(z_, t_nodes, pos)[0]
    msc_ffa = float(emsc.FORM_FACTOR * float(emsc.elements(mat)[2][0]) ** 0.54)
    # positron annihilation in flight (v0.0.8): Heitler per atomic electron (G4eeToTwoGammaModel, as option 0)
    from hepsimkernels import eannihil

    sig_a = np.zeros((2, nt))
    sig_a[1] = N_E_MM3 * eannihil.xsec_per_electron(t_nodes)
    mott_coef = np.asarray(emsc.mott_coefficients()[Z_AR], float)
    w_lnq, w_lna = wentzel_table()
    # restricted CSDA range: R(T) = integral of dT / dedx from 0 (a 1/dedx extrapolation below t_min)
    rng = np.zeros((2, nt))
    for ip in range(2):
        inv = 1.0 / np.maximum(dedx[ip], 1e-12)
        rng[ip, 0] = t_nodes[0] * inv[0]
        rng[ip, 1:] = rng[ip, 0] + np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(t_nodes))

    k_nodes = np.geomspace(k_min, t_max, nk)
    lnk0, dlnk = np.log(k_min), np.log(t_max / k_min) / (nk - 1)
    sig_c = N_E_MM3 * klein_nishina_xsec_per_electron(k_nodes)
    sig_p = pair_xsec_per_mm(k_nodes)
    sig_ph = N_ATOM_MM3 * 1e-22 * photoelectric_xsec_per_atom_barn(k_nodes)  # barn = 1e-22 mm^2
    compton_q = np.zeros((nk, nc))
    pair_q = np.zeros((nk, nc))
    for i, k in enumerate(k_nodes):
        emin = 1.0 / (1.0 + 2.0 * k / ME)
        eps = emin + (1.0 - emin) * np.linspace(0.0, 1.0, 1025)
        compton_q[i] = _inverse_cdf(eps, klein_nishina_deps(eps, k), c_nodes)
        if k > 2 * ME * 1.0001:
            e0 = ME / k
            ep = np.linspace(e0, 1 - e0, 1025)
            pair_q[i] = _inverse_cdf(ep, pair_deps_per_atom(ep, k), c_nodes)
        else:
            pair_q[i] = 0.5
    x0_mm = radiation_length_g_cm2() / RHO_LAR * 10.0
    return HostTables(lnt0, dlnt, t_nodes, dedx, rng, sig_b, sig_d, sig_a, brem_q, delta_q, lam1, msc_k, msc_s, msc_ue,
                      msc_env, mott_coef, msc_ffa, w_lnq, w_lna, lnk0, dlnk,
                      k_nodes, sig_c, sig_p, sig_ph, compton_q, pair_q, float(tcut), float(kcut), float(x0_mm))
