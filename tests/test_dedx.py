"""Fixed-length dE/dx (compare.fixed_length_dedx) on synthetic tracks: pure NumPy."""

import numpy as np

from dunefastsim.transport import compare as C
from dunefastsim.transport.compare import energy_batches, fixed_length_dedx


def _track(rng, length, dedx_mev_cm, tid):
    dx = rng.uniform(0.01, 6.0, 400)
    dx = dx[np.cumsum(dx) <= length]
    dx = np.r_[dx, length - dx.sum()]
    return np.full(len(dx), tid), np.cumsum(dx), dedx_mev_cm * dx / 10.0, dx


def test_uniform_track_any_steps():
    rng = np.random.default_rng(0)
    a = _track(rng, 100.0, 2.1, 7)
    b = _track(rng, 40.0, 5.0, 3)
    # interleave the two tracks and add a zero-length step: order comes from (track, time)
    tid, t, de, dx = (np.r_[x, y] for x, y in zip(a, b))
    p = rng.permutation(len(tid))
    tid, t, de, dx = tid[p], t[p], de[p], dx[p]
    v = fixed_length_dedx(tid, t, de, dx, 3.72)
    assert len(v) == int(100 // 3.72) + int(40 // 3.72)
    assert np.allclose(np.sort(v), np.r_[[2.1] * 26, [5.0] * 10])
    v7 = fixed_length_dedx(tid, t, de, dx, 10.0, first_mm=35.0, track=7)
    assert np.allclose(v7, 2.1) and len(v7) == 3


def test_zero_length_step_and_partial_segment():
    tid = np.zeros(4, int)
    t = np.arange(4.0)
    dx = np.array([5.0, 0.0, 5.0, 3.0])
    de = np.array([1.0, 0.5, 1.0, 9.0])
    v = fixed_length_dedx(tid, t, de, dx, 10.0)  # one full segment [0, 10): 1 + 0.5 + 1; [10, 13) is partial
    assert np.allclose(v, [2.5])


def test_energy_batches():
    ke = np.array([5.0, 5.0, 30.0, 1.0, 1.0, 1.0])
    g = energy_batches(ke, 12.0, 2)
    assert [list(x) for x in g] == [[0, 1], [2], [3, 4], [5]]


def _brute(tid, t, de, dx, L):
    out = []
    for k in np.unique(tid):
        s = tid == k
        o = np.argsort(t[s], kind="stable")
        d, e = dx[s][o], de[s][o]
        b = np.cumsum(d)
        a = b - d
        for j in range(int(b[-1] // L)):
            ov = np.clip(np.minimum(b, (j + 1) * L) - np.maximum(a, j * L), 0, None)
            inbin = (a >= j * L) & (a < (j + 1) * L)
            out.append(10 * np.sum(e * np.where(d > 0, ov / np.where(d > 0, d, 1), inbin)) / L)
    return np.sort(out)


def test_matches_brute_force_many_tracks():
    """Many tracks of many small steps (cumulative-sum rounding once opened a spurious segment -1 per track)."""
    rng = np.random.default_rng(3)
    n = 6000
    tid = rng.integers(0, 1200, n).astype(np.int32)
    t = rng.uniform(0, 1, n)
    dx = rng.uniform(1e-6, 2.0, n) * (rng.uniform(size=n) > 0.02)
    de = 0.19 * dx * rng.lognormal(0, 0.3, n)
    for L in (3.72, 10.0):
        assert np.allclose(np.sort(fixed_length_dedx(tid, t, de, dx, L)), _brute(tid, t, de, dx, L))


# ---------------------------------------------------------------------------------------- edep-sim step merging
def _line(tid, n, step, d, start=(0.0, 0.0, 0.0), t0=0.0):
    d = np.asarray(d, float) / np.linalg.norm(d)
    ends = np.asarray(start) + np.outer(np.arange(1, n + 1) * step, d)
    mids = ends - 0.5 * step * d
    return (np.full(n, tid), t0 + np.arange(n, dtype=float), mids, np.full(n, 0.1), np.full(n, step), np.tile(d, (n, 1)))


def _cat(*parts):
    return tuple(np.concatenate([p[k] for p in parts]) for k in range(6))


def test_merge_straight_track_into_3mm_segments():
    tid, t, pos, de, dx, d = _line(4, 20, 0.5, [1, 0, 0])
    mt, mtt, mde, mdx = C.edepsim_segments(tid, t, pos, de, dx, d)
    assert np.allclose(mdx, [3.0, 3.0, 3.0, 1.0]) and np.allclose(mde, [0.6, 0.6, 0.6, 0.2])
    assert np.all(mt == 4) and np.allclose(mtt, [0, 6, 12, 18])


def test_merge_splits_at_a_kink_beyond_the_sagitta():
    a = _line(1, 3, 0.5, [1, 0, 0])  # 1.5 mm along x, then along y from (1.5, 0, 0)
    b = _line(1, 8, 0.5, [0, 1, 0], start=(1.5, 0.0, 0.0), t0=3.0)
    mt, mtt, mde, mdx = C.edepsim_segments(*_cat(a, b))
    assert np.allclose(mdx, [1.5, 3.0, 1.0])  # the x leg, then the y leg in 3 mm pieces


def test_merge_never_joins_tracks_and_keeps_energy():
    rng = np.random.default_rng(5)
    parts = [_line(k, int(rng.integers(1, 30)), float(rng.uniform(0.1, 2.0)), rng.normal(size=3),
                   start=rng.normal(size=3), t0=float(k)) for k in range(6)]
    tid, t, pos, de, dx, d = _cat(*parts)
    de = rng.uniform(0.01, 0.5, len(tid))
    o = rng.permutation(len(tid))  # input order does not matter
    mt, mtt, mde, mdx = C.edepsim_segments(tid[o], t[o], pos[o], de[o], dx[o], d[o])
    for k in range(6):
        assert np.isclose(mde[mt == k].sum(), de[tid == k].sum()) and np.isclose(mdx[mt == k].sum(), dx[tid == k].sum())
        assert np.all(mdx[mt == k] <= 3.0 + 1e-9)
