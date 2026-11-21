"""The step record: JAXTPC's input HDF5 layout (the H5DLP writer of the drinkingkazu/edep-sim fork).

Read from JAXTPC's ``tools/loader.py`` (commit 50ed555) and its sample ``muon.h5`` on 2026-11-21; not copied from
either. One row per event in every dataset; each row of the ``object`` datasets is a variable-length array of the
compound types below.

Units (as the fork writes them and JAXTPC reads them): positions in mm, times in ns, energies and momenta in MeV,
step length ``dx`` in mm, ``theta`` / ``phi`` the step direction's polar and azimuthal angles in rad.

Datasets:

- ``pstep/lar_vol``               : PSTEP rows, sorted by ``track_id`` (contiguous per track)
- ``particle/geant4``             : PARTICLE rows, row k is the particle with ``track_id`` k
- ``primary/geant4``              : PRIMARY rows
- ``vertex/geant4``               : VERTEX rows
- ``ass/particle_pstep_lar_vol``  : ASS rows, row k = [start, end) of particle k's steps in ``pstep/lar_vol``
- ``event/geant4``                : EVENT, one fixed-size row per event
"""

from __future__ import annotations

import numpy as np

# Names: the fork writes variable-length strings; h5py cannot write those nested in variable-length compound rows
# ("invalid background buffer pointer"), so they are fixed 32-byte strings here. JAXTPC skips string fields.
_STR = "S32"

PSTEP = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("t", "<f4"),
    ("theta", "<f4"), ("phi", "<f4"), ("p", "<f4"),
    ("pdg", "<i4"), ("track_id", "<i4"), ("ancestor_track_id", "<i4"),
    ("de", "<f4"), ("dx", "<f4"),
    ("proc_start", "<i4"), ("subproc_start", "<i2"), ("proc_stop", "<i4"), ("subproc_stop", "<i2"),
])

PARTICLE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("t", "<f4"),
    ("px", "<f4"), ("py", "<f4"), ("pz", "<f4"), ("ke", "<f4"),
    ("end_x", "<f4"), ("end_y", "<f4"), ("end_z", "<f4"), ("end_t", "<f4"),
    ("end_px", "<f4"), ("end_py", "<f4"), ("end_pz", "<f4"), ("end_ke", "<f4"),
    ("track_id", "<i4"), ("parent_track_id", "<i4"), ("ancestor_track_id", "<i4"),
    ("pdg", "<i4"), ("mass", "<f4"),
    ("proc_start", "<i4"), ("subproc_start", "<i4"), ("proc_name_start", _STR),
    ("proc_end", "<i4"), ("subproc_end", "<i4"), ("proc_name_end", _STR),
])

PRIMARY = np.dtype([
    ("px", "<f4"), ("py", "<f4"), ("pz", "<f4"), ("ke", "<f4"),
    ("interaction_id", "<i4"), ("track_id", "<i4"), ("pdg", "<i4"), ("mass", "<f4"), ("name", _STR),
])

VERTEX = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("t", "<f4"),
    ("energy_sum", "<f4"), ("ke_sum", "<f4"), ("num_particles", "<i4"),
    ("generator", _STR), ("reaction", _STR), ("filename", _STR),
    ("generator_vertex_id", "<i4"), ("xs", "<f4"), ("diff_xs", "<f4"), ("weight", "<f4"), ("probability", "<f4"),
    ("interaction_id", "<i2"),
])

ASS = np.dtype([("start", "<u8"), ("end", "<u8")])

EVENT = np.dtype([
    ("run_id", "<i4"), ("event_id", "<i4"), ("num_vertices", "<i4"),
    ("num_primaries", "<i4"), ("num_particles", "<i4"), ("num_steps", "<i4"),
])

DATASETS = {
    "pstep/lar_vol": PSTEP,
    "particle/geant4": PARTICLE,
    "primary/geant4": PRIMARY,
    "vertex/geant4": VERTEX,
    "ass/particle_pstep_lar_vol": ASS,
}

# What JAXTPC's loader reads (tools/loader.py, 50ed555): the rest is truth for analysis.
REQUIRED = {
    "pstep/lar_vol": ("x", "y", "z", "t", "theta", "phi", "de", "dx", "track_id", "ancestor_track_id", "pdg"),
    "primary/geant4": ("track_id", "interaction_id"),
    "particle/geant4": ("track_id", "parent_track_id"),
}


def empty(dtype: np.dtype, n: int = 0) -> np.ndarray:
    """A zero-filled record array of ``n`` rows (string fields empty)."""
    return np.zeros(n, dtype=dtype)
