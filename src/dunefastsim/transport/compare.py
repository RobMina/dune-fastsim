"""Compare option A (parametric showers) with a Geant4 reference in the step-record format.

Both sides are seen through the same geometry: the reference's steps exist only in the active argon (edep-sim's
segment detector), and option A's spots are kept only inside the drift volumes of the same JAXTPC config. Each
reference event has one primary electron (bdnmc-dune writes one interaction per event); option A showers that
electron (same vertex, direction and kinetic energy).

Per energy bin it reports: the deposited fraction (summed deposits / summed kinetic energy, and per-event quantiles),
the energy-weighted mean depth along the electron direction (radiation lengths), the depth of the profile maximum,
and the fraction of deposits within 1 and 2 Moliere radii (PDG R_M = 9.04 cm) of the axis; and, for the reference,
the dE/dx of its steps (option A uses one MIP value).

Step-level dE/dx depends on the step length (Geant4's edep-sim segments run from microns to centimetres, option B's
steps are at most ``s_max``), so for option B the tool also measures dE/dx at a fixed length on both sides
(``fixed_length_dedx``): each e+- track's deposits are spread uniformly along their step and summed in consecutive
segments of length L along the track; only full segments count. Reported for every e+- track and for the primary
electron's first ``PRIMARY_MM`` of track, for L = ``SEG_MM`` (the ND-LAr pixel pitch, 3.72 mm, and 10 mm).

The reference's "steps" are edep-sim hit segments, not Geant4 steps: edep-sim's segment detector merges consecutive
Geant4 steps of one track while the segment stays within 3 mm end to end and 1 mm sagitta (its defaults; DUNE's ND
production keeps them and sets only hitSeparation -1 mm, so different tracks are never merged). Spreading a merged
segment's energy uniformly smooths the fixed-length dE/dx over lengths comparable to 3 mm. ``edepsim_segments``
applies the same merging to option B's steps, and the report carries option B's fixed-length dE/dx both as stepped
(``fixed_length_dedx``) and after the merging (``fixed_length_dedx_edepsim``), the like-for-like comparison.
"""

from __future__ import annotations

import json
import time

import h5py
import numpy as np

X0_MM = 140.0
RM_PDG_MM = 90.43
E_BINS_MEV = (30.0, 100.0, 300.0, 1000.0, 3000.0, 30000.0)
T_EDGES = np.linspace(0.0, 25.0, 51)  # X0
R_EDGES = np.linspace(0.0, 5.0, 51)  # R_M
LOG_DEDX_EDGES = np.linspace(-1.5, 2.5, 161)  # log10(MeV/cm)
SEG_MM = (3.72, 10.0)  # fixed dE/dx lengths: ND-LAr pixel pitch (JAXTPC config pixel_pitch 0.372 cm) and 1 cm
PRIMARY_MM = 50.0  # primary electron: first 5 cm of track
SEG_EDGES = np.linspace(0.0, 10.0, 201)  # MeV/cm, fixed-length dE/dx histograms (overflow in the last bin)


def fixed_length_dedx(tid, t, de, dx, length_mm, first_mm=None, track=None):
    """dE/dx (MeV/cm) of every full segment of length ``length_mm`` along each track.

    Steps are ordered by (track id, time); each step's energy is spread uniformly over its length; a segment counts
    only if the track (as recorded: steps outside the active volume are absent on both sides) reaches its end. With
    ``first_mm`` only segments ending within the first ``first_mm`` of track count; with ``track`` only that track id.
    """
    tid, t, de, dx = (np.asarray(a) for a in (tid, t, de, dx))
    if track is not None:
        m = tid == track
        tid, t, de, dx = tid[m], t[m], de[m], dx[m]
    if len(tid) == 0:
        return np.zeros(0)
    o = np.lexsort((t, tid))
    tid, de, dx = tid[o], de[o].astype(np.float64), np.maximum(dx[o].astype(np.float64), 0.0)
    new = np.r_[True, tid[1:] != tid[:-1]]
    trk = np.cumsum(new) - 1
    c = np.cumsum(dx)
    b = c - (c - dx)[np.flatnonzero(new)][trk]  # path length at each step's end, from the track's first step
    a = np.maximum(b - dx, 0.0)  # rounding can put a track's start at -1e-13: no segment -1
    b = np.maximum(b, a)
    total = np.zeros(trk[-1] + 1)
    np.maximum.at(total, trk, b)
    L = float(length_mm)
    j0 = np.floor(a / L).astype(np.int64)
    keys, vals = [], []
    for k in range(int(np.ceil(dx.max() / L)) + 2 if len(dx) else 1):
        j = j0 + k
        ov = np.minimum(b, (j + 1) * L) - np.maximum(a, j * L)
        w = np.where(dx > 0, np.clip(ov, 0.0, None) / np.where(dx > 0, dx, 1.0), (k == 0) * 1.0)
        sel = w > 0
        keys.append(np.stack([trk[sel], j[sel]], 1))
        vals.append(de[sel] * w[sel])
    keys, vals = np.concatenate(keys), np.concatenate(vals)
    u, inv = np.unique(keys, axis=0, return_inverse=True)
    e = np.bincount(inv.ravel(), weights=vals, minlength=len(u))
    end = (u[:, 1] + 1) * L
    ok = (end <= total[u[:, 0]] + 1e-9) & (u[:, 1] >= 0)
    if first_mm is not None:
        ok &= end <= first_mm + 1e-9
    return 10.0 * e[ok] / L


EDEPSIM_MAX_LENGTH_MM = 3.0  # EDepSim::SegmentSD fMaximumHitLength default
EDEPSIM_MAX_SAGITTA_MM = 1.0  # EDepSim::SegmentSD fMaximumHitSagitta default


def edepsim_segments(tid, t, pos, de, dx, dirn, max_length=EDEPSIM_MAX_LENGTH_MM, max_sagitta=EDEPSIM_MAX_SAGITTA_MM,
                     kmax=48):
    """Merge consecutive steps of each track as edep-sim's EDepSim::HitSegment::SameHit does (same track, segment
    front to new post-step point <= max_length, every recorded path point within max_sagitta of the line through the
    front along the new step's direction); returns per-segment (tid, t, de, dx) with t the first step's time and dx
    the summed step lengths. Step end points are pos -/+ dirn dx / 2 (pos the step midpoint, dirn the chord
    direction). Vectorised over tracks, one pass per step index; a segment holds at most kmax - 1 steps."""
    tid, t, de, dx = (np.asarray(a) for a in (tid, t, de, dx))
    n = len(tid)
    if n == 0:
        return tid, t, de, dx
    o = np.lexsort((t, tid))
    tid, t, de, dx = tid[o], t[o], de[o].astype(np.float64), dx[o].astype(np.float64)
    pos, u = np.asarray(pos, np.float64)[o], np.asarray(dirn, np.float64)[o]
    u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-30)
    pre, post = pos - 0.5 * dx[:, None] * u, pos + 0.5 * dx[:, None] * u
    starts = np.flatnonzero(np.r_[True, tid[1:] != tid[:-1]])
    nst = np.diff(np.r_[starts, n])
    ntr = len(starts)
    gid = np.empty(n, np.int64)
    front = np.zeros((ntr, 3))
    buf = np.zeros((ntr, kmax, 3))
    cnt = np.zeros(ntr, np.int64)
    nxt = 0
    for k in range(int(nst.max())):
        act = np.flatnonzero(nst > k)
        i = starts[act] + k
        if k == 0:
            same = np.zeros(len(act), bool)
        else:
            e2e = np.linalg.norm(post[i] - front[act], axis=1)
            back = buf[act, cnt[act] - 1]
            nd = post[i] - back
            nd /= np.maximum(np.linalg.norm(nd, axis=1, keepdims=True), 1e-30)
            delta = buf[act] - front[act][:, None, :]
            dist = np.einsum("akc,ac->ak", delta, nd)
            perp = np.linalg.norm(delta - dist[:, :, None] * nd[:, None, :], axis=2)
            perp = np.where(np.arange(kmax)[None, :] < cnt[act][:, None], perp, 0.0)
            same = (e2e <= max_length) & (perp.max(axis=1) <= max_sagitta) & (cnt[act] < kmax)
        a_s, i_s = act[same], i[same]
        gid[i_s] = gid[i_s - 1]
        buf[a_s, cnt[a_s]] = post[i_s]
        cnt[a_s] += 1
        a_n, i_n = act[~same], i[~same]
        gid[i_n] = nxt + np.arange(len(i_n))
        nxt += len(i_n)
        front[a_n] = pre[i_n]
        buf[a_n, 0] = pre[i_n]
        buf[a_n, 1] = post[i_n]
        cnt[a_n] = 2
    first = np.r_[True, gid[1:] != gid[:-1]]  # gid is constant over each run of merged steps
    return (tid[first], t[first], np.bincount(gid, weights=de, minlength=nxt)[gid[first]],
            np.bincount(gid, weights=dx, minlength=nxt)[gid[first]])


def _seg_hists(tid, t, de, dx, pdg, primary_tid):
    """Fixed-length dE/dx histograms (counts) for all e+- tracks and for the primary's first PRIMARY_MM."""
    em = np.abs(pdg) == 11
    out = {}
    for L in SEG_MM:
        allv = fixed_length_dedx(tid[em], t[em], de[em], dx[em], L)
        prim = fixed_length_dedx(tid[em], t[em], de[em], dx[em], L, first_mm=PRIMARY_MM, track=primary_tid)
        out[f"all_{L:g}"] = np.histogram(np.minimum(allv, SEG_EDGES[-1] - 1e-9), SEG_EDGES)[0]
        out[f"primary_{L:g}"] = np.histogram(np.minimum(prim, SEG_EDGES[-1] - 1e-9), SEG_EDGES)[0]
    return out


def primaries(path: str):
    """Vertex (n, 3) mm, direction (n, 3) and kinetic energy (n,) MeV of each event's primary electron."""
    with h5py.File(path, "r") as f:
        n = f["primary/geant4"].shape[0]
        vtx, mom, ke = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n)
        for i in range(n):
            p = f["primary/geant4"][i]
            v = f["vertex/geant4"][i]
            if len(p) != 1 or abs(int(p["pdg"][0])) != 11 or len(v) < 1:
                raise ValueError(f"event {i}: expected one primary electron, got pdg {list(p['pdg'])}")
            vtx[i] = (v["x"][0], v["y"][0], v["z"][0])
            mom[i] = (p["px"][0], p["py"][0], p["pz"][0])
            ke[i] = p["ke"][0]
    return vtx, mom / np.linalg.norm(mom, axis=1, keepdims=True), ke


def inside(pos: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Mask of points (n, 3) inside any box [lo, hi] (m, 3)."""
    out = np.zeros(len(pos), dtype=bool)
    for s in range(0, len(lo), 10):
        out |= np.any(np.all((pos[:, None, :] >= lo[None, s:s + 10]) & (pos[:, None, :] <= hi[None, s:s + 10]), axis=2),
                      axis=1)
    return out


def summarize(pos, de, vertex, direction, dx_mm=None, tracks=None):
    """Per-event shower summary in the electron's frame (and the energy-weighted dE/dx histogram when dx is given;
    fixed-length dE/dx histograms when ``tracks`` = (tid, t, pdg, primary tid) is given)."""
    d = pos - vertex
    t_mm = d @ direction
    r_mm = np.linalg.norm(d - np.outer(t_mm, direction), axis=1)
    e = float(de.sum())
    return {
        "edep": e,
        "t_hist": np.histogram(t_mm / X0_MM, T_EDGES, weights=de)[0],
        "r_hist": np.histogram(r_mm / RM_PDG_MM, R_EDGES, weights=de)[0],
        "t_mean": float(np.sum(de * t_mm) / e / X0_MM) if e > 0 else np.nan,
        "dedx_hist": (np.histogram(np.log10(np.maximum(10.0 * de / np.maximum(dx_mm, 1e-9), 1e-3)), LOG_DEDX_EDGES,
                                   weights=de)[0] if dx_mm is not None else np.zeros(len(LOG_DEDX_EDGES) - 1)),
        "seg": _seg_hists(tracks[0], tracks[1], de, dx_mm, tracks[2], tracks[3]) if tracks is not None else None,
    }


def _hist_quantiles(h, edges, qs):
    c = np.cumsum(h) / max(h.sum(), 1e-30)
    return [float(10 ** np.interp(q, c, edges[1:])) for q in qs]


def _lognormal_fit(h, edges):
    """Energy-weighted median and sigma of ln(dE/dx) from a log10 histogram."""
    x = 0.5 * (edges[1:] + edges[:-1]) * np.log(10.0)
    w = h / max(h.sum(), 1e-30)
    mu = float(np.sum(w * x))
    return {"median_MeV_cm": float(np.exp(mu)), "sigma_ln": float(np.sqrt(np.sum(w * (x - mu) ** 2)))}


def _seg_summary(h):
    c = np.cumsum(h) / max(h.sum(), 1)
    x = 0.5 * (SEG_EDGES[1:] + SEG_EDGES[:-1])
    k = max(int(np.argmax(h)), 1)
    # mode: parabola through the three bins around the histogram maximum
    y0, y1, y2 = h[k - 1], h[k], h[min(k + 1, len(h) - 1)]
    den = y0 - 2 * y1 + y2
    mode = x[k] + (0.5 * (y0 - y2) / den * (x[1] - x[0]) if den != 0 else 0.0)
    return {"segments": int(h.sum()), "mpv": round(float(mode), 3),
            "q10_50_90": [round(float(np.interp(q, c, SEG_EDGES[1:])), 3) for q in (0.1, 0.5, 0.9)],
            "hist": h.astype(int).tolist()}


def _bin_report(rows, ke):
    edep = np.array([r["edep"] for r in rows])
    th = np.sum([r["t_hist"] for r in rows], axis=0)
    rh = np.sum([r["r_hist"] for r in rows], axis=0)
    tm = np.array([r["t_mean"] for r in rows])
    dh = np.sum([r["dedx_hist"] for r in rows], axis=0)
    frac = edep / ke
    tc = 0.5 * (T_EDGES[1:] + T_EDGES[:-1])
    rc = np.cumsum(rh) / max(rh.sum(), 1e-30)
    return {
        "events": int(len(rows)),
        "deposited_fraction": float(edep.sum() / ke.sum()),
        "per_event_fraction_q10_50_90": np.quantile(frac, [0.1, 0.5, 0.9]).round(4).tolist(),
        "mean_depth_x0": float(np.dot(th / max(th.sum(), 1e-30), 0.5 * (T_EDGES[1:] + T_EDGES[:-1]))),
        "per_event_mean_depth_x0_median": float(np.nanmedian(tm)),
        "profile_max_x0": float(tc[np.argmax(th)]),
        "within_1_RM": float(rc[np.searchsorted(R_EDGES, 1.0) - 1]),
        "within_2_RM": float(rc[np.searchsorted(R_EDGES, 2.0) - 1]),
        "dedx_energy_weighted_q10_50_90": [round(x, 3) for x in _hist_quantiles(dh, LOG_DEDX_EDGES, (0.1, 0.5, 0.9))],
        "dedx_lognormal": _lognormal_fit(dh, LOG_DEDX_EDGES),
        "t_profile": (th / max(th.sum(), 1e-30)).round(6).tolist(),
        "r_profile": (rh / max(rh.sum(), 1e-30)).round(6).tolist(),
        "fixed_length_dedx": ({k: _seg_summary(np.sum([r["seg"][k] for r in rows], axis=0)) for k in rows[0]["seg"]}
                              if rows and rows[0].get("seg") is not None else None),
        **({"fixed_length_dedx_edepsim": {k: _seg_summary(np.sum([r["seg_edepsim"][k] for r in rows], axis=0))
                                          for k in rows[0]["seg_edepsim"]}}
           if rows and rows[0].get("seg_edepsim") is not None else {}),
    }


def energy_batches(ke, budget_mev: float, b_max: int):
    """Consecutive event index groups of at most ``b_max`` events and (unless one event exceeds it) ``budget_mev``."""
    out, cur, tot = [], [], 0.0
    for i, e in enumerate(ke):
        if cur and (tot + e > budget_mev or len(cur) >= b_max):
            out.append(np.array(cur))
            cur, tot = [], 0.0
        cur.append(i)
        tot += e
    if cur:
        out.append(np.array(cur))
    return out


def run(ref_path: str, config: dict, n_pairs: int = 2048, batch: int = 100, seed: int = 0, dtype: str = "float64",
        max_events: int | None = None, model: str = "A", b_config=None, b_tables=None, b_batch: int = 1,
        b_budget_mev: float = 20000.0, b_edepsim_merge: bool = True) -> dict:
    """Compare option A (``model="A"``) or option B (``"B"``) with the reference; return a JSON-able report.

    Option B runs ``b_batch`` showers per call (B2; padded with zero-energy primaries so one compile serves all
    calls), at most ``b_budget_mev`` of primary energy per call so the deposit buffer holds them; ``b_batch=1`` is
    one shower per call as in B0 / B1. ``b_edepsim_merge``: also report option B's fixed-length dE/dx after
    edep-sim's step merging (``edepsim_segments``), as the reference's segments are."""
    import jax
    import jax.numpy as jnp

    if dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    from dunefastsim.transport import parametric as pm

    jdt = jnp.float64 if dtype == "float64" else jnp.float32
    r = np.array([v["geometry"]["ranges"] for v in config["volumes"]], dtype=np.float64) * 10.0
    lo, hi = r[:, :, 0] - 1e-3, r[:, :, 1] + 1e-3
    vtx, direc, ke = primaries(ref_path)
    n = len(ke) if max_events is None else min(max_events, len(ke))
    vtx, direc, ke = vtx[:n], direc[:n], ke[:n]

    ref_rows, dedx = [], []
    with h5py.File(ref_path, "r") as f:
        for i in range(n):
            s = f["pstep/lar_vol"][i]
            pos = np.column_stack([s["x"], s["y"], s["z"]]).astype(np.float64)
            de = s["de"].astype(np.float64)
            tracks = None
            if model == "B":
                tracks = (s["track_id"], s["t"].astype(np.float64), s["pdg"], int(f["primary/geant4"][i]["track_id"][0]))
            ref_rows.append(summarize(pos, de, vtx[i], direc[i], s["dx"].astype(np.float64), tracks))
            ok = s["dx"] > 0.05  # mm; drop zero-length steps
            if len(dedx) < 2_000_000:
                dedx.append(10.0 * s["de"][ok] / s["dx"][ok])  # MeV/cm
    dedx = np.concatenate(dedx) if dedx else np.zeros(0)

    key = jax.random.PRNGKey(seed)
    a_rows, t_compile, t_run = [], None, 0.0
    b_stats = {"lost_mev": 0.0, "peak_max": 0, "iters_max": 0, "deposits_max": 0}
    if model == "B":
        from dunefastsim.transport import cascade as cs
        from dunefastsim.transport import emphysics as em

        host = b_tables if b_tables is not None else em.build_tables()
        cfg = b_config if b_config is not None else cs.Config()
        fb = cs.make_showers_fn(cs.device_tables(host, jdt), cfg)
        groups = energy_batches(ke, b_budget_mev, b_batch)
        b_stats.update(calls=len(groups), batch=b_batch, budget_mev=b_budget_mev, n_cap=cfg.n_cap, m_cap=cfg.m_cap,
                       fill=cfg.fill, deposits_per_mev_max=0.0)
        a_rows = [None] * n
        for g, idx in enumerate(groups):
            pad = b_batch - len(idx)
            e = np.pad(ke[idx], (0, pad))
            v = np.pad(vtx[idx], ((0, pad), (0, 0)))
            d = np.pad(direc[idx], ((0, pad), (0, 0)), constant_values=1.0)
            t0 = time.perf_counter()
            out = fb(jax.random.fold_in(key, g), e, v, d)
            evs, info = cs.split_events(out, b_batch)
            dt_g = time.perf_counter() - t0
            if t_compile is None:
                t_compile, n_first = dt_g, len(idx)
            else:
                t_run += dt_g
            b_stats["lost_mev"] += float(sum(ev["lost_mev"] for ev in evs))
            b_stats["peak_max"] = max(b_stats["peak_max"], info["peak"])
            b_stats["iters_max"] = max(b_stats["iters_max"], info["iters"])
            b_stats["deposits_max"] = max(b_stats["deposits_max"], info["deposits"])
            b_stats["deposits_per_mev_max"] = max(b_stats["deposits_per_mev_max"], info["deposits"] / ke[idx].sum())
            for j, i_ev in enumerate(idx):
                ev = evs[j]
                pos = ev["positions_mm"].astype(np.float64)
                m = inside(pos, lo, hi)
                a_rows[i_ev] = summarize(pos[m], ev["de"][m].astype(np.float64), vtx[i_ev], direc[i_ev],
                                         ev["dx"][m].astype(np.float64),
                                         (ev["tid"][m], ev["t"][m].astype(np.float64), ev["pdg"][m], 0))
                if b_edepsim_merge:
                    em_ = m & (np.abs(ev["pdg"]) == 11)
                    mt, mtt, mde, mdx = edepsim_segments(ev["tid"][em_], ev["t"][em_].astype(np.float64), pos[em_],
                                                         ev["de"][em_], ev["dx"][em_], ev["dir"][em_])
                    a_rows[i_ev]["seg_edepsim"] = _seg_hists(mt, mtt, mde, mdx, np.full(len(mt), 11), 0)
        batch = n_first
    f = jax.jit(lambda k, e, v, d: pm.showers(k, e, v, d, n_pairs=n_pairs, dtype=jdt))
    for b, s in enumerate(range(0, n if model == "A" else 0, batch)):
        idx = np.arange(s, min(s + batch, n))
        pad = batch - len(idx)
        e = np.pad(ke[idx], (0, pad), constant_values=100.0)
        v = np.pad(vtx[idx], ((0, pad), (0, 0)))
        d = np.pad(direc[idx], ((0, pad), (0, 0)), constant_values=1.0)
        t0 = time.perf_counter()
        out = f(jax.random.fold_in(key, b), e, v, d)
        pos = np.asarray(out["positions_mm"], np.float64)
        de = np.asarray(out["de"], np.float64)
        dxa = np.asarray(out["dx"], np.float64)
        dt = time.perf_counter() - t0
        if t_compile is None:
            t_compile = dt
        else:
            t_run += dt
        for j, i in enumerate(idx):
            m = inside(pos[j], lo, hi)
            a_rows.append(summarize(pos[j][m], de[j][m], vtx[i], direc[i], dxa[j][m]))

    report = {"events": int(n), "model": model, "n_pairs": n_pairs, "dtype": dtype, "option_B": b_stats,
              "timing_s": {"first_batch_incl_compile": t_compile,
                           "per_shower_after_compile": t_run / max(n - batch, 1),
                           "device": str(jax.devices()[0])},
              "reference_step_dedx_MeV_cm_q10_50_90": np.quantile(dedx, [0.1, 0.5, 0.9]).round(3).tolist()
              if len(dedx) else None,
              "option_A_dedx_MeV_cm": pm.DEDX_MIP_MEV_CM,
              "t_edges_x0": T_EDGES.tolist(), "r_edges_rm": R_EDGES.tolist(), "bins": {}}
    edges = list(E_BINS_MEV)
    for lo_e, hi_e in [(edges[0], edges[-1])] + list(zip(edges[:-1], edges[1:])):
        sel = np.where((ke >= lo_e) & (ke < hi_e))[0]
        if not len(sel):
            continue
        name = f"{lo_e:g}-{hi_e:g} MeV"
        report["bins"][name] = {"geant4": _bin_report([ref_rows[i] for i in sel], ke[sel]),
                                f"option_{model}": _bin_report([a_rows[i] for i in sel], ke[sel])}
    return report


def plot(report: dict, out_png: str) -> None:
    """Longitudinal and radial profiles per energy bin, Geant4 against option A."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = [k for k in report["bins"] if not k.startswith(f"{E_BINS_MEV[0]:g}-{E_BINS_MEV[-1]:g}")]
    tc = 0.5 * (np.array(report["t_edges_x0"][1:]) + np.array(report["t_edges_x0"][:-1]))
    rc = 0.5 * (np.array(report["r_edges_rm"][1:]) + np.array(report["r_edges_rm"][:-1]))
    fig, ax = plt.subplots(2, len(bins), figsize=(3.6 * len(bins), 6.2), squeeze=False)
    for j, name in enumerate(bins):
        mk = f"option_{report.get('model', 'A')}"
        g, a = report["bins"][name]["geant4"], report["bins"][name][mk]
        ax[0, j].step(tc, g["t_profile"], where="mid", color="#1f5fa8", label="Geant4 (edep-sim)")
        ax[0, j].step(tc, a["t_profile"], where="mid", color="#c4561d", label=mk.replace("_", " "))
        ax[0, j].set_title(f"{name}  (n={g['events']})", fontsize=9)
        ax[0, j].set_xlabel("depth along electron [X0]", fontsize=8)
        ax[1, j].step(rc, g["r_profile"], where="mid", color="#1f5fa8")
        ax[1, j].step(rc, a["r_profile"], where="mid", color="#c4561d")
        ax[1, j].set_yscale("log")
        ax[1, j].set_xlabel("distance from axis [R_M = 9.04 cm]", fontsize=8)
    ax[0, 0].set_ylabel("energy fraction per bin", fontsize=8)
    ax[1, 0].set_ylabel("energy fraction per bin", fontsize=8)
    ax[0, 0].legend(fontsize=7)
    fig.suptitle(f"ND-LAr active argon: Geant4 reference vs {mk.replace('_', ' ')} (same electrons, same volumes)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)


def plot_dedx(report: dict, out_png: str) -> None:
    """Fixed-length dE/dx, all energies: every e+- track and the primary's first PRIMARY_MM, per segment length."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mk = f"option_{report.get('model', 'B')}"
    b = next(iter(report["bins"].values()))
    keys = list(b["geant4"]["fixed_length_dedx"])
    x = 0.5 * (SEG_EDGES[1:] + SEG_EDGES[:-1])
    fig, ax = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.4), squeeze=False)
    for j, k in enumerate(keys):
        for side, col, lab in (("geant4", "#1f5fa8", "Geant4 (edep-sim)"), (mk, "#c4561d", mk.replace("_", " "))):
            s = b[side]["fixed_length_dedx"][k]
            h = np.array(s["hist"], float)
            ax[0, j].step(x, h / max(h.sum(), 1), where="mid", color=col,
                          label=f"{lab}: MPV {s['mpv']:.2f}, median {s['q10_50_90'][1]:.2f}")
        if "fixed_length_dedx_edepsim" in b[mk]:
            s = b[mk]["fixed_length_dedx_edepsim"][k]
            h = np.array(s["hist"], float)
            ax[0, j].step(x, h / max(h.sum(), 1), where="mid", color="#c4561d", ls="--",
                          label=f"{mk.replace('_', ' ')}, edep-sim 3 mm merging: MPV {s['mpv']:.2f}, "
                                f"median {s['q10_50_90'][1]:.2f}")
        what, L = k.split("_")
        ax[0, j].set_title(f"{'all e+- tracks' if what == 'all' else f'primary, first {PRIMARY_MM / 10:g} cm'}, "
                           f"L = {L} mm", fontsize=9)
        ax[0, j].set_xlabel("dE/dx over L [MeV/cm]", fontsize=8)
        ax[0, j].set_xlim(1.0, 4.0)
        ax[0, j].legend(fontsize=6)
    ax[0, 0].set_ylabel("segments (normalised)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)


def main(argv=None) -> int:
    import argparse

    import yaml

    p = argparse.ArgumentParser(prog="python -m dunefastsim.transport.compare")
    p.add_argument("ref")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True, help="JSON report")
    p.add_argument("--png", default=None)
    p.add_argument("--n-pairs", type=int, default=2048)
    p.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--model", default="A", choices=["A", "B"])
    p.add_argument("--n-cap", type=int, default=2048)
    p.add_argument("--m-cap", type=int, default=1 << 18)
    p.add_argument("--b-batch", type=int, default=1, help="option B showers per call (B2)")
    p.add_argument("--b-budget-gev", type=float, default=20.0, help="option B primary energy per call")
    p.add_argument("--fill", type=float, default=0.25, help="option B: inject primaries below fill * n_cap alive")
    p.add_argument("--s-max", type=float, default=10.0, help="option B: longest e+- step (mm)")
    p.add_argument("--no-fluct", action="store_true", help="option B: mean loss, no fluctuations")
    p.add_argument("--fluct", default="universal", choices=["universal", "urban"],
                   help="option B: G4UniversalFluctuation (EM option 0, the reference's) or G4UrbanFluctuation")
    p.add_argument("--no-merge", action="store_true", help="option B: skip the edep-sim merged dE/dx")
    p.add_argument("--tcut", type=float, default=0.22, help="option B: delta-ray production cut (MeV)")
    p.add_argument("--kcut", type=float, default=0.1,
                   help="option B: bremsstrahlung photon cut (MeV; the reference's is 0.00441638)")
    p.add_argument("--k-track", type=float, default=0.1, help="option B: photons below are absorbed in place (MeV)")
    a = p.parse_args(argv)
    with open(a.config) as fh:
        cfg = yaml.safe_load(fh)
    b_cfg = None
    if a.model == "B":
        from dunefastsim.transport import cascade as cs
        b_cfg = cs.Config(n_cap=a.n_cap, m_cap=a.m_cap, fill=a.fill, s_max=a.s_max, fluctuate=not a.no_fluct,
                          fluct_model=a.fluct, k_track=a.k_track)
    b_tab = None
    if a.model == "B" and (a.tcut, a.kcut) != (0.22, 0.1):
        from dunefastsim.transport import emphysics as em
        b_tab = em.build_tables(tcut=a.tcut, kcut=a.kcut)
    rep = run(a.ref, cfg, n_pairs=a.n_pairs, dtype=a.dtype, max_events=a.max_events, model=a.model, b_config=b_cfg,
              b_tables=b_tab,
              b_batch=a.b_batch, b_budget_mev=1000.0 * a.b_budget_gev, b_edepsim_merge=not a.no_merge)
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1)
    if a.png:
        plot(rep, a.png)
        if a.model == "B":
            plot_dedx(rep, a.png.replace(".png", "_dedx.png"))
    short = {k: {s: {m: v[s][m] for m in ("events", "deposited_fraction", "per_event_fraction_q10_50_90",
                                          "mean_depth_x0", "profile_max_x0", "within_1_RM", "within_2_RM",
                                          "dedx_energy_weighted_q10_50_90", "dedx_lognormal")}
                     for s in ("geant4", f"option_{a.model}")} for k, v in rep["bins"].items()}
    first = next(iter(rep["bins"].values()))
    seg = {s: {k: {m: v[m] for m in ("segments", "mpv", "q10_50_90")} for k, v in (first[s]["fixed_length_dedx"] or {}).items()}
           for s in ("geant4", f"option_{a.model}")}
    if "fixed_length_dedx_edepsim" in first[f"option_{a.model}"]:
        seg[f"option_{a.model}_edepsim_merged"] = {k: {m: v[m] for m in ("segments", "mpv", "q10_50_90")}
                                                   for k, v in first[f"option_{a.model}"]["fixed_length_dedx_edepsim"].items()}
    print(json.dumps({"timing_s": rep["timing_s"], "option_B": rep["option_B"], "fixed_length_dedx_all_energies": seg,
                      "ref_dedx": rep["reference_step_dedx_MeV_cm_q10_50_90"],
                      "bins": short}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
