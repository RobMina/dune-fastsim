"""Compare option A (parametric showers) with a Geant4 reference in the step-record format.

Both sides are seen through the same geometry: the reference's steps exist only in the active argon (edep-sim's
segment detector), and option A's spots are kept only inside the drift volumes of the same JAXTPC config. Each
reference event has one primary electron (bdnmc-dune writes one interaction per event); option A showers that
electron (same vertex, direction and kinetic energy).

Per energy bin it reports: the deposited fraction (summed deposits / summed kinetic energy, and per-event quantiles),
the energy-weighted mean depth along the electron direction (radiation lengths), the depth of the profile maximum,
and the fraction of deposits within 1 and 2 Moliere radii (PDG R_M = 9.04 cm) of the axis; and, for the reference,
the dE/dx of its steps (option A uses one MIP value).
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


def summarize(pos, de, vertex, direction, dx_mm=None):
    """Per-event shower summary in the electron's frame (and the energy-weighted dE/dx histogram when dx is given)."""
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
    }


def run(ref_path: str, config: dict, n_pairs: int = 2048, batch: int = 100, seed: int = 0, dtype: str = "float64",
        max_events: int | None = None, model: str = "A", b_config=None, b_tables=None) -> dict:
    """Compare option A (``model="A"``) or option B (``"B"``) with the reference; return a JSON-able report."""
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
            ref_rows.append(summarize(pos, de, vtx[i], direc[i], s["dx"].astype(np.float64)))
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
        fb = cs.make_shower_fn(cs.device_tables(host, jdt), cfg)
        for i in range(n):
            t0 = time.perf_counter()
            dep = cs.to_numpy(fb(jax.random.fold_in(key, i), ke[i], vtx[i], direc[i]))
            dt_i = time.perf_counter() - t0
            if t_compile is None:
                t_compile = dt_i
            else:
                t_run += dt_i
            b_stats["lost_mev"] += dep["lost_mev"]
            b_stats["peak_max"] = max(b_stats["peak_max"], dep["peak"])
            b_stats["iters_max"] = max(b_stats["iters_max"], dep["iters"])
            b_stats["deposits_max"] = max(b_stats["deposits_max"], len(dep["de"]))
            pos = dep["positions_mm"].astype(np.float64)
            m = inside(pos, lo, hi)
            a_rows.append(summarize(pos[m], dep["de"][m].astype(np.float64), vtx[i], direc[i],
                                    dep["dx"][m].astype(np.float64)))
        batch = 1
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
    a = p.parse_args(argv)
    with open(a.config) as fh:
        cfg = yaml.safe_load(fh)
    b_cfg = None
    if a.model == "B":
        from dunefastsim.transport import cascade as cs
        b_cfg = cs.Config(n_cap=a.n_cap, m_cap=a.m_cap)
    rep = run(a.ref, cfg, n_pairs=a.n_pairs, dtype=a.dtype, max_events=a.max_events, model=a.model, b_config=b_cfg)
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1)
    if a.png:
        plot(rep, a.png)
    short = {k: {s: {m: v[s][m] for m in ("events", "deposited_fraction", "per_event_fraction_q10_50_90",
                                          "mean_depth_x0", "profile_max_x0", "within_1_RM", "within_2_RM",
                                          "dedx_energy_weighted_q10_50_90", "dedx_lognormal")}
                     for s in ("geant4", f"option_{a.model}")} for k, v in rep["bins"].items()}
    print(json.dumps({"timing_s": rep["timing_s"], "option_B": rep["option_B"],
                      "ref_dedx": rep["reference_step_dedx_MeV_cm_q10_50_90"],
                      "bins": short}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
