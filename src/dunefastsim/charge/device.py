"""On-device volume split: option B deposits -> JAXTPC's per-volume padded deposits, with no host step.

JAXTPC's ``loader.build_deposit_data`` does this split in NumPy (assign each deposit to the drift volume that holds
it, move it to that volume's local frame, group consecutive deposits of a track, pad every volume to ``total_pad``).
:func:`split_volumes` is the same computation in JAX: jit-compiled, on the device, and differentiable in the deposit
values (energies, lengths, positions within a volume; which volume a deposit lands in is piecewise constant). Its
output is the stacked ``VolumeDeposits`` pytree JAXTPC's jitted calculators take (one leading axis over volumes,
``n_actual`` an array), so option B's buffer goes to JAXTPC's charge, light and pixel response without leaving the
device. Differences from ``build_deposit_data``: a volume with more than ``total_pad`` deposits is truncated and the
dropped energy reported (``overflow_mev``) instead of raising, and nothing is converted to NumPy.

The test ``tests/test_charge_device.py`` checks it field by field against ``build_deposit_data``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class VolumeTable(NamedTuple):
    """Per-volume constants of a JAXTPC SimConfig, as float32 device arrays."""

    lo_cm: object  # (V, 3)
    hi_cm: object  # (V, 3)
    x_anode_mm: object  # (V,)
    drift: object  # (V,) +1 / -1
    yz_center_mm: object  # (V, 2)


class SplitInfo(NamedTuple):
    n_in_volumes: object  # deposits inside some volume (before truncation)
    outside_mev: object  # energy of valid deposits outside every volume (JAXTPC drops them too)
    overflow_mev: object  # energy dropped because a volume had more than total_pad deposits
    multi: object  # deposits inside more than one volume (0 for a valid geometry; build_deposit_data duplicates them)
    group_to_track: object  # (V, total_pad + 1) track id of each group, per volume (JAXTPC's group_to_track)
    original_index: object  # (V, total_pad) index of each slot's deposit in the input (-1 for padding)


def volume_table(sim_config) -> VolumeTable:
    """The volume boxes and frames of a JAXTPC SimConfig. Constants are rounded to float32 the way
    ``build_deposit_data`` rounds them (float64 products, then float32 arithmetic with float32 positions)."""
    import jax.numpy as jnp

    vols = sim_config.volumes
    r = np.array([v.ranges_cm for v in vols], np.float64)  # (V, 3, 2)
    f32 = lambda a: jnp.asarray(np.asarray(a, np.float64).astype(np.float32))  # noqa: E731
    return VolumeTable(
        lo_cm=f32(r[:, :, 0]), hi_cm=f32(r[:, :, 1]),
        x_anode_mm=f32([v.x_anode_cm * 10.0 for v in vols]),
        drift=f32([v.drift_direction for v in vols]),
        yz_center_mm=f32([(v.yz_center_cm[0] * 10.0, v.yz_center_cm[1] * 10.0) for v in vols]),
    )


def split_volumes(vt: VolumeTable, total_pad: int, positions_mm, de, dx, theta, phi, track_ids, t0_us,
                  interaction_ids, root_track_ids, pdg, valid, group_size: int = 5, gap_threshold_mm: float = 5.0):
    """Flat deposits (N,) -> (stacked VolumeDeposits with leading axis V, SplitInfo). Traceable (jit it with
    ``total_pad``, ``group_size`` and ``gap_threshold_mm`` static). Deposits with ``valid`` False are ignored."""
    import jax.numpy as jnp
    from jax import lax

    from . import _tools

    _tools()
    from tools.config import VolumeDeposits

    f32, i32 = jnp.float32, jnp.int32
    n = positions_mm.shape[0]
    nv = vt.lo_cm.shape[0]
    idx = jnp.arange(n, dtype=i32)

    # Volume of each deposit, as build_deposit_data: float32 mm -> float32 cm, half-open boxes.
    p = positions_mm.astype(f32)
    cm = p / f32(10.0)
    inside = jnp.all((cm[:, None, :] >= vt.lo_cm[None]) & (cm[:, None, :] < vt.hi_cm[None]), axis=-1) & valid[:, None]
    n_hit = inside.sum(axis=1, dtype=i32)
    vid = jnp.where(n_hit > 0, jnp.argmax(inside, axis=1).astype(i32), nv)

    # Local frame of the deposit's volume: x = drift * (x_anode - x), y and z from the volume centre.
    vc = jnp.minimum(vid, nv - 1)
    local = jnp.stack([vt.drift[vc] * (vt.x_anode_mm[vc] - p[:, 0]), p[:, 1] - vt.yz_center_mm[vc, 0],
                       p[:, 2] - vt.yz_center_mm[vc, 1]], axis=1)

    # Slot of each deposit: input order within its volume (stable sort by volume).
    order = jnp.argsort(vid, stable=True)
    sv = vid[order]
    counts = jnp.bincount(vid, length=nv + 1).astype(i32)
    starts = (jnp.cumsum(counts) - counts).astype(i32)
    slot = (idx - starts[sv]).astype(i32)
    keep = (sv < nv) & (slot < total_pad)
    flat = jnp.where(keep, sv * total_pad + slot, nv * total_pad)  # out of range -> dropped by the scatter
    de32 = de.astype(f32)
    overflow = jnp.sum(jnp.where((sv < nv) & (slot >= total_pad), de32[order], 0.0))
    outside = jnp.sum(jnp.where(valid & (vid == nv), de32, 0.0))

    # Groups (JAXTPC's compute_group_ids, per volume): kept deposits sorted by (volume, track, input order); a new
    # group at a volume or track change, at a gap above gap_threshold_mm, and every group_size deposits; 1-based.
    kept_dep = jnp.zeros(n, bool).at[order].set(keep)
    gv = jnp.where(kept_dep, vid, nv)
    tid = track_ids.astype(i32)
    g = jnp.lexsort((idx, tid, gv))
    gv_s, tid_s, pos_s = gv[g], tid[g], lax.stop_gradient(local)[g]  # grouping only: no gradient (norm(0) NaN)
    step = jnp.linalg.norm(pos_s[1:] - pos_s[:-1], axis=1)
    seg = jnp.concatenate([jnp.ones(1, bool), (gv_s[1:] != gv_s[:-1]) | (tid_s[1:] != tid_s[:-1])
                           | (step > gap_threshold_mm)])
    seg_pos = lax.cummax(jnp.where(seg, idx, 0), axis=0)
    within = idx - seg_pos
    gstart = seg | ((within % group_size == 0) & (within > 0))
    cs = jnp.cumsum(gstart.astype(i32)).astype(i32)
    kept_counts = jnp.bincount(gv, length=nv + 1).astype(i32)
    vstart = (jnp.cumsum(kept_counts) - kept_counts).astype(i32)
    label = cs - cs[vstart[jnp.minimum(gv_s, nv)]] + 1
    label = jnp.where(gv_s < nv, label, 0).astype(i32)
    gid_dep = jnp.zeros(n, i32).at[g].set(label)
    g2t = jnp.zeros((nv + 1, total_pad + 1), i32).at[gv_s, jnp.minimum(label, total_pad)].set(
        jnp.where(gv_s < nv, tid_s, 0))[:nv]

    def scatter(a, fill, dtype):
        a = a.astype(dtype)
        shape = (nv * total_pad,) + a.shape[1:]
        out = jnp.full(shape, fill, dtype).at[flat].set(a[order], mode="drop")
        return out.reshape((nv, total_pad) + a.shape[1:])

    stacked = VolumeDeposits(
        positions_mm=scatter(local, 0.0, f32), de=scatter(de, 0.0, f32), dx=scatter(dx, 1.0, f32),
        theta=scatter(theta, 0.0, f32), phi=scatter(phi, 0.0, f32), track_ids=scatter(tid, -1, i32),
        group_ids=scatter(gid_dep, 0, i32), t0_us=scatter(t0_us, 0.0, f32),
        interaction_ids=scatter(interaction_ids, -1, jnp.int16), root_track_ids=scatter(root_track_ids, -1, i32),
        pdg=scatter(pdg, 0, i32), charge=jnp.zeros((nv, total_pad), f32), photons=jnp.zeros((nv, total_pad), f32),
        qs_fractions=jnp.zeros((nv, total_pad), f32), n_actual=jnp.minimum(counts[:nv], total_pad).astype(i32))
    info = SplitInfo(n_in_volumes=jnp.sum(counts[:nv]), outside_mev=outside, overflow_mev=overflow,
                     multi=jnp.sum(n_hit > 1), group_to_track=g2t, original_index=scatter(idx, -1, i32))
    return stacked, info


def option_b_event(vt: VolumeTable, total_pad: int, dep, b, interaction_id: int = 0, group_size: int = 5,
                   gap_threshold_mm: float = 5.0):
    """Primary ``b``'s deposits of an option B ``showers`` buffer (``transport.cascade.Deposits``, on the device)
    -> (stacked VolumeDeposits, SplitInfo). Same mapping as :func:`dunefastsim.charge.option_b_deposits` (theta, phi
    from the direction; ns -> us; track ids from 0 per primary; root track 0) without leaving the device."""
    import jax.numpy as jnp

    m = dep.de.shape[0]
    valid = (jnp.arange(m) < dep.n) & (dep.anc == b)
    tid = jnp.where(valid, dep.tid, jnp.iinfo(jnp.int32).max)
    tid = jnp.where(valid, dep.tid - jnp.min(tid), -1)
    d = dep.dirn
    theta = jnp.arccos(jnp.clip(d[:, 2], -1.0, 1.0))
    phi = jnp.arctan2(d[:, 1], d[:, 0])
    z = jnp.zeros(m, jnp.int32)
    return split_volumes(vt, total_pad, dep.pos, dep.de, dep.dx, theta, phi, tid, dep.t / 1000.0,
                         z + interaction_id, z, dep.pdg, valid, group_size, gap_threshold_mm)


def make_event_split(vt: VolumeTable, total_pad: int, group_size: int = 5, gap_threshold_mm: float = 5.0):
    """jit-compiled ``(Deposits, b) -> (stacked VolumeDeposits, SplitInfo)`` (one compile per buffer size)."""
    import jax

    return jax.jit(lambda dep, b: option_b_event(vt, total_pad, dep, b, 0, group_size, gap_threshold_mm))


def unstack(stacked, info: SplitInfo | None = None):
    """A JAXTPC DepositData (tuple of per-volume VolumeDeposits, device slices) from the stacked pytree, for
    JAXTPC's public ``process_event`` / ``process_event_light`` (which stack them again on the device)."""
    import jax

    from . import _tools

    _tools()
    from tools.config import DepositData

    nv = stacked.de.shape[0]
    vols = tuple(jax.tree.map(lambda a, v=v: a[v], stacked) for v in range(nv))
    g2t = tuple(info.group_to_track[v] for v in range(nv)) if info is not None else None
    oi = tuple(info.original_index[v] for v in range(nv)) if info is not None else None
    return DepositData(volumes=vols, group_to_track=g2t, original_indices=oi)


def light_stacked(sim, stacked, sim_params=None):
    """JAXTPC charge and photons per slot, (V, total_pad) each, straight from the stacked deposits (JAXTPC's jitted
    light calculator; what ``process_event_light`` runs after stacking). Call in :func:`jaxtpc_precision`."""
    params = sim.default_sim_params if sim_params is None else sim_params
    charges, photons, *_ = sim._light_calculator_jit(params, stacked)
    return charges, photons
