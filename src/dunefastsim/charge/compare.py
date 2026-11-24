"""Option B through JAXTPC against Geant4 through JAXTPC (B3, forward only): charge and light per event.

For each event of a Geant4 step-record file, the Geant4 steps (loaded as JAXTPC's production/run_batch.py loads them)
and option B's shower of the same primary electron (same vertex, direction and energy) are given to the same JAXTPC
DetectorSimulator. ``process_event_light`` turns each deposit into ionisation charge (recombination: the config's
model, EMB for ND-LAr) and scintillation photons; the report compares, per event and in total, the energy inside
JAXTPC's volumes, the charge and the light, and the charge per MeV (the recombination survival, which depends on each
deposit's dE/dx and angle to the drift field). ``--pixels K`` also runs the full pixel response on the first K events
(sparse pixel hits; about 12 min and 44 GB per ND-LAr event on CPU) and compares hit counts and summed signal.

``--device-split`` sends option B's buffer to JAXTPC through ``charge.device`` (volume split on the device, JAXTPC's
jitted light calculator on the stacked volumes) instead of ``split_events`` + ``build_deposit_data`` on the host.

CLI: python -m dunefastsim.charge.compare steps.h5 --config ndlar.yaml --out report.json [--max-events N] [--pixels K]
     [--device-split]
"""

from __future__ import annotations

import json
import time

import numpy as np


def _sums(dep):
    """(energy in volumes [MeV], charge [e-], photons) of a light-filled DepositData."""
    e = q = L = 0.0
    for v in dep.volumes:
        n = int(v.n_actual)
        e += float(np.asarray(v.de)[:n].sum())
        q += float(np.asarray(v.charge)[:n].sum())
        L += float(np.asarray(v.photons)[:n].sum())
    return e, q, L


def _pixel_summary(resp):
    """Number of pixel hits and summed signal over all volumes of a process_event response (sparse pixel dicts)."""
    n, s = 0, 0.0
    for v in resp.values():
        arrs = [np.asarray(a) for a in (v.values() if isinstance(v, dict) else [v])]
        val = arrs[-1]
        n += int(val.size)
        s += float(val.sum())
    return n, s


def run(ref_path, config_yaml, max_events=None, total_pad=20000, pixels=0, seed=0, b_dtype="float64",
        device_split=False):
    import jax

    if b_dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    from dunefastsim import charge as ch
    from dunefastsim.transport import cascade as cs
    from dunefastsim.transport import compare as C
    from dunefastsim.transport import emphysics as em

    sim = ch.make_simulator(config_yaml, total_pad=total_pad)
    cfg = sim.config
    vtx, direc, ke = C.primaries(ref_path)
    n = len(ke) if max_events is None else min(max_events, len(ke))
    jdt = jax.numpy.float64 if b_dtype == "float64" else jax.numpy.float32
    fb = cs.make_showers_fn(cs.device_tables(em.build_tables(), jdt), cs.Config(n_cap=16384, m_cap=1 << 20, fill=0.15))
    key = jax.random.PRNGKey(seed)
    rows = {"geant4": [], "option_B": []}
    pix = {"geant4": [], "option_B": []}
    t_light = {"geant4": 0.0, "option_B": 0.0}
    if device_split:
        from dunefastsim.charge import device as dv

        split = dv.make_event_split(dv.volume_table(cfg), total_pad)
        overflow = 0.0
    for g, idx in enumerate(C.energy_batches(ke[:n], 120000.0, 64)):
        pad = 64 - len(idx)
        out = fb(jax.random.fold_in(key, g), np.pad(ke[idx], (0, pad)), np.pad(vtx[idx], ((0, pad), (0, 0))),
                 np.pad(direc[idx], ((0, pad), (0, 0)), constant_values=1.0))
        evs = None if device_split else cs.split_events(out, 64)[0]
        for j, i in enumerate(idx):
            deps = {"geant4": ch.step_file_deposits(ref_path, int(i), cfg)}
            if device_split:  # option B's buffer -> JAXTPC volumes on the device, no host step
                t0 = time.perf_counter()
                stacked, info = split(out, j)
                with ch.jaxtpc_precision():
                    q, L = dv.light_stacked(sim, stacked)
                e, qs, Ls = float(stacked.de.sum()), float(q.sum()), float(L.sum())
                t_light["option_B"] += time.perf_counter() - t0
                overflow += float(info.overflow_mev)
                rows["option_B"].append((float(ke[i]), e, qs, Ls))
            else:
                deps["option_B"] = ch.option_b_deposits(evs[j], cfg)
            for side, dep in deps.items():
                t0 = time.perf_counter()
                lit = ch.light(sim, dep)
                e, q, L = _sums(lit)
                t_light[side] += time.perf_counter() - t0
                rows[side].append((float(ke[i]), e, q, L))
                if int(i) < pixels:
                    resp, _, _ = ch.pixels(sim, dep)
                    pix[side].append(_pixel_summary(resp))
    report = {"events": int(n), "config": str(config_yaml), "total_pad": total_pad, "option_B_dtype": b_dtype,
              "option_B_split": "device" if device_split else "host",
              "option_B_overflow_mev": overflow if device_split else None,
              "light_seconds_per_event": {k: v / max(n, 1) for k, v in t_light.items()}, "sides": {}}
    for side, r in rows.items():
        a = np.array(r)
        ke_, e, q, L = a.T
        ok = e > 1.0
        report["sides"][side] = {
            "energy_in_volumes_over_ke": float(e.sum() / ke_.sum()),
            "charge_e_per_mev": float(q.sum() / e.sum()),
            "photons_per_mev": float(L.sum() / e.sum()),
            "charge_e_per_mev_per_event_q10_50_90": np.quantile(q[ok] / e[ok], [0.1, 0.5, 0.9]).round(1).tolist(),
            "charge_over_ke_q10_50_90": np.quantile(q / ke_, [0.1, 0.5, 0.9]).round(1).tolist(),
            "per_event": a.round(4).tolist(),
        }
        if pix[side]:
            p = np.array(pix[side])
            report["sides"][side]["pixels"] = {"events": len(p), "hits_mean": float(p[:, 0].mean()),
                                               "signal_mean": float(p[:, 1].mean()), "per_event": p.tolist()}
    g, b = report["sides"]["geant4"], report["sides"]["option_B"]
    report["ratio_B_over_G4"] = {k: b[k] / g[k] for k in ("energy_in_volumes_over_ke", "charge_e_per_mev",
                                                           "photons_per_mev")}
    return report


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m dunefastsim.charge.compare")
    p.add_argument("ref")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--total-pad", type=int, default=20000)
    p.add_argument("--pixels", type=int, default=0, help="full pixel response on the first K events")
    p.add_argument("--b-dtype", default="float64", choices=["float32", "float64"])
    p.add_argument("--device-split", action="store_true",
                   help="option B to JAXTPC through charge.device (no host step); --pixels then runs only Geant4")
    a = p.parse_args(argv)
    rep = run(a.ref, a.config, a.max_events, a.total_pad, a.pixels, b_dtype=a.b_dtype, device_split=a.device_split)
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1)
    short = {s: {k: v for k, v in d.items() if k not in ("per_event",) and not isinstance(v, dict)}
             for s, d in rep["sides"].items()}
    print(json.dumps({"events": rep["events"], "light_seconds_per_event": rep["light_seconds_per_event"],
                      "ratio_B_over_G4": rep["ratio_B_over_G4"], "sides": short}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
