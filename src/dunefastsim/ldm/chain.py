"""BdNMC (JAX port) -> option B -> JAXTPC on the device (B3's chain).

BdNMC's port generates dark-matter scatters with ``pipeline.events(..., stream=True)``, which leaves each batch's
accepted rows on the device: the recoil electron's four-momentum (GeV) and its vertex and time (m and s, BdNMC's beam
frame). Each electron becomes an option B primary in ND-LAr's frame (bdnmc-dune's ``geometry.frame_from_card``: the
detector-centred frame by default, ``gdml_offset`` for a GDML's world frame), option B showers it, ``charge.device``
splits each shower into JAXTPC's drift volumes and JAXTPC computes charge and light, all on the device. Only per-event
totals come back to the host, and the electrons' energies, which the host uses to pack showers into option B's
buffer (at most ``budget_mev`` and ``b_batch`` showers per call).

With ``ref``, a step file simulated from a run of the same card and seed, Geant4's steps go through the same JAXTPC
simulator, and chain events are paired with reference events by content (kinetic energy, vertex, direction): the
paired events are the same scatters, compared event by event; the rest only as a sample. The port reproduces a run
event by event on GPUs (batch-invariant); on CPU its last bits differ between machines and a rejection step then
draws differently, so on another CPU node only part of the events pair (1457 of the first 2000 on the eta reference).

The port runs in float32 by default and turns JAX's 64-bit mode off on import. It runs here inside
:func:`generator_precision` (64-bit mode set globally: the port dispatches from a worker thread, which thread-local
contexts do not reach), JAXTPC inside ``charge.jaxtpc_precision`` and option B in float64 (or float32, ``b_dtype``).
The port needs JAX >= 0.11 (it locks 0.12.0.1); JAXTPC (declares jax<0.7) and option B pass their tests on 0.12.0.1.

CLI: python -m dunefastsim.ldm.chain card.dat --trials N --config ndlar.yaml --out report.json [--seed S]
     [--ref steps.h5] [--max-events N] [--total-pad N]
"""

from __future__ import annotations

import contextlib
import json
import time

import numpy as np

M_E_MEV = 0.51099895


@contextlib.contextmanager
def generator_precision():
    """64-bit mode as BdNMC's port runs (its PRECISION), set globally for the block and restored after. Not the
    thread-local ``jax.enable_x64`` / ``disable_x64`` contexts: the port dispatches batches from a worker thread, which
    does not see them (it then traces with 64-bit mode on and its int32 loop counters fail)."""
    import bdnmc_jax
    import jax

    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", bdnmc_jax.PRECISION == "float64")
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old)


def frame(card, gdml_offset=(0.0, 0.0, 0.0)):
    """(origin m, rotation 3x3) of ND-LAr's frame for a BdNMC card: r_det = rotation (r - origin)."""
    from bdnmc_dune import geometry

    origin, rot = geometry.frame_from_card(str(card), gdml_offset)
    return np.asarray(origin, float), np.asarray(rot, float)


def primaries(rows, n, origin, rot, dtype=None):
    """Option B primaries from accepted BdNMC rows (device arrays): kinetic energy (MeV), vertex (mm, ND-LAr frame),
    unit direction and vertex time (ns), each (n,), on the device."""
    import jax.numpy as jnp

    dt = jnp.float64 if dtype is None else dtype
    e4 = jnp.asarray(rows["electron"][:n]).astype(dt)
    v4 = jnp.asarray(rows["vertex"][:n]).astype(dt)
    o, r = jnp.asarray(origin, dt), jnp.asarray(rot, dt)
    pos = (v4[:, :3] - o) @ r.T * 1000.0
    p = e4[:, 1:] @ r.T
    d = p / jnp.linalg.norm(p, axis=1, keepdims=True)
    ke = e4[:, 0] * 1000.0 - M_E_MEV
    return {"ke": ke, "vtx": pos, "dir": d, "t_ns": v4[:, 3] * 1e9}


def run(card, n_trials, config_yaml, seed=None, ref=None, max_events=None, total_pad=100_000, b_batch=64,
        budget_mev=120_000.0, b_dtype="float64", key_seed=0, gdml_offset=(0.0, 0.0, 0.0), n_ref=None):
    import bdnmc_jax  # sets 64-bit mode to its precision; option B is set up after it
    from bdnmc_jax import pipeline

    import jax
    import jax.numpy as jnp

    if b_dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    from dunefastsim import charge as ch
    from dunefastsim.charge import device as dv
    from dunefastsim.charge.compare import _sums
    from dunefastsim.transport import cascade as cs
    from dunefastsim.transport import emphysics as em

    sync = jax.block_until_ready
    t = {"generator": 0.0, "primaries": 0.0, "option_B": 0.0, "split_and_charge": 0.0, "geant4_light": 0.0}

    t0 = time.perf_counter()
    with generator_precision():
        res = pipeline.events(str(card), int(n_trials), seed=seed, stream=True)
        sync([d for d, _ in res["device_rows"]])
    t["generator"] = time.perf_counter() - t0

    sim = ch.make_simulator(config_yaml, total_pad=total_pad)
    cfg = sim.config
    origin, rot = frame(card, gdml_offset)
    jdt = jnp.float64 if b_dtype == "float64" else jnp.float32
    fb = cs.make_showers_fn(cs.device_tables(em.build_tables(), jdt),
                            cs.Config(n_cap=16384, m_cap=1 << 20, fill=0.15))
    split = dv.make_event_split(dv.volume_table(cfg), total_pad)
    key = jax.random.PRNGKey(key_seed)

    t0 = time.perf_counter()
    prims = [primaries(d, n, origin, rot, jdt) for d, n in res["device_rows"] if n > 0]
    prim = {k: jnp.concatenate([p[k] for p in prims]) for k in prims[0]} if prims else None
    n_acc = 0 if prim is None else int(prim["ke"].shape[0])
    n = n_acc if max_events is None else min(int(max_events), n_acc)
    prim = {k: v[:n] for k, v in prim.items()}
    sync(prim)
    t["primaries"] = time.perf_counter() - t0
    ke_host = np.asarray(prim["ke"])  # the host packs showers into buffers by energy

    from dunefastsim.transport.compare import energy_batches

    per = []  # per event: device scalars (ke, e in volumes, charge, photons, overflow, lost, nuclear)
    for g, idx in enumerate(energy_batches(ke_host, budget_mev, b_batch)):
        pad = b_batch - len(idx)
        ii = jnp.asarray(np.pad(idx, (0, pad)))
        live = jnp.arange(b_batch) < len(idx)
        e = jnp.where(live, prim["ke"][ii], 0.0)
        v = jnp.where(live[:, None], prim["vtx"][ii], 0.0)
        d = jnp.where(live[:, None], prim["dir"][ii], jnp.array([0.0, 0.0, 1.0], jdt))
        t0 = time.perf_counter()
        out = sync(fb(jax.random.fold_in(key, g), e, v, d))
        t["option_B"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        for j, i in enumerate(idx):
            stacked, info = split(out, j, prim["t_ns"][i])
            with ch.jaxtpc_precision():
                q, L = dv.light_stacked(sim, stacked)
            per.append(sync((prim["ke"][i], stacked.de.sum(), q.sum(), L.sum(), info.overflow_mev, out.lost_mev[j],
                             out.nuclear_mev[j])))
        t["split_and_charge"] += time.perf_counter() - t0
    rows_b = np.array(jax.device_get(per), np.float64) if per else np.zeros((0, 7))

    report = {"card": str(card), "trials": int(res["trials"]), "seed": res["seed"], "accepted": int(res["accepted"]),
              "se": float(res["se"]), "generator_precision": bdnmc_jax.PRECISION, "backend": jax.default_backend(),
              "events": int(n), "option_B_dtype": b_dtype, "total_pad": total_pad, "config": str(config_yaml),
              "seconds": t, "seconds_per_event": {k: v / max(n, 1) for k, v in t.items() if k != "generator"},
              "option_B": _side(rows_b[:, :4]) | {
                  "overflow_mev": float(rows_b[:, 4].sum()), "lost_mev": float(rows_b[:, 5].sum()),
                  "nuclear_over_ke": float(rows_b[:, 6].sum() / max(rows_b[:, 0].sum(), 1e-30)),
                  "per_event": rows_b.round(4).tolist()}}

    if ref is not None:
        from dunefastsim.transport.compare import primaries as ref_primaries

        rv, rd, rk = ref_primaries(str(ref))
        m = len(rk) if n_ref is None else min(len(rk), int(n_ref))
        rv, rd, rk = rv[:m], rd[:m] / np.linalg.norm(rd[:m], axis=1, keepdims=True), rk[:m]
        pv, pdir, pk = (np.asarray(prim[k]) for k in ("vtx", "dir", "ke"))
        # Pair chain events with reference events by content (the port reproduces a run event by event only on the
        # same kind of device: on GPU it is batch-invariant, on CPU last bits differ between machines and a
        # rejection step then draws differently).
        order = np.argsort(rk)
        pairs = []
        for i in range(n):
            j0 = np.searchsorted(rk[order], pk[i] * (1 - 1e-5))
            for j in order[j0:j0 + 8]:
                if (abs(rk[j] / pk[i] - 1) < 1e-5 and np.abs(rv[j] - pv[i]).max() < 1.0
                        and float(pdir[i] @ rd[j]) > 1 - 1e-8):
                    pairs.append((i, int(j)))
                    break
        pi = np.array([p[0] for p in pairs], int)
        pj = np.array([p[1] for p in pairs], int)
        report["pairing"] = {
            "chain_events": int(n), "reference_events": int(m), "paired": len(pairs),
            "criteria": "KE within 1e-5, vertex within 1 mm, direction within 1.4e-4 rad",
            "max_vertex_mm": float(np.abs(pv[pi] - rv[pj]).max()) if len(pairs) else None,
            "max_ke_rel": float(np.abs(pk[pi] / rk[pj] - 1).max()) if len(pairs) else None}
        g4 = []
        for i in range(m):
            t0 = time.perf_counter()
            lit = ch.light(sim, ch.step_file_deposits(str(ref), i, cfg))
            g4.append((float(rk[i]), *_sums(lit)))
            t["geant4_light"] += time.perf_counter() - t0
        g4 = np.array(g4)
        keys = ("energy_in_volumes_over_ke", "charge_e_per_mev", "photons_per_mev")
        report["geant4"] = _side(g4) | {"per_event": g4.round(4).tolist()}
        report["ratio_B_over_G4_same_sample_different_events"] = {
            k: report["option_B"][k] / report["geant4"][k] for k in keys}
        if len(pairs) > 1:
            b, g = _side(rows_b[pi, :4]), _side(g4[pj])
            report["paired"] = {"option_B": b, "geant4": g, "ratio_B_over_G4": {k: b[k] / g[k] for k in keys},
                                "pairs": [[int(a), int(c)] for a, c in pairs]}
    return report


def _side(a):
    ke, e, q, L = np.asarray(a, np.float64).T
    ok = e > 1.0
    return {"events": int(len(ke)), "energy_in_volumes_over_ke": float(e.sum() / ke.sum()),
            "charge_e_per_mev": float(q.sum() / e.sum()), "photons_per_mev": float(L.sum() / e.sum()),
            "charge_e_per_mev_per_event_q10_50_90": np.quantile(q[ok] / e[ok], [0.1, 0.5, 0.9]).round(1).tolist()}


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m dunefastsim.ldm.chain")
    p.add_argument("card")
    p.add_argument("--trials", type=int, required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ref", default=None, help="Geant4 step file simulated from the same generator run")
    p.add_argument("--n-ref", type=int, default=None, help="reference events to compare (default: all in --ref)")
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--total-pad", type=int, default=100_000)
    p.add_argument("--b-dtype", default="float64", choices=["float32", "float64"])
    a = p.parse_args(argv)
    rep = run(a.card, a.trials, a.config, seed=a.seed, ref=a.ref, max_events=a.max_events, total_pad=a.total_pad,
              b_dtype=a.b_dtype, n_ref=a.n_ref)
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1)
    short = {k: ({kk: vv for kk, vv in v.items() if kk != "per_event"} if isinstance(v, dict) else v)
             for k, v in rep.items()}
    print(json.dumps(short, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
