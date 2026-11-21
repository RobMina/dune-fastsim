"""The ``dune`` experiment profile (hep-simlib when installed, else a local stand-in)."""

from __future__ import annotations

from dataclasses import dataclass

SUBSYSTEMS = ("ndlar", "pixel", "light", "beamline", "component")

DUNE_PROFILE_KW = dict(
    name="dune",
    subsystems=SUBSYSTEMS,
    dtype_env="DUNE_DEVICE_FLOAT",
    default_dtype="float64",
    cache_env="DUNE_LIBRARY_CACHE",
    cache_dir="~/.cache/dunefastsim/libraries",
    bank_subsystem="component",
    bank_prefix="component.",
)


@dataclass(frozen=True)
class LocalProfile:
    """Minimal stand-in for hepsimlib.profile.Profile."""

    name: str = "dune"
    subsystems: tuple = SUBSYSTEMS
    dtype_env: str = "DUNE_DEVICE_FLOAT"
    default_dtype: str = "float64"


def install():
    """Make the dune profile current; prefer hep-simlib when it is installed."""
    try:
        from hepsimlib import profile as hp
    except ImportError:
        return LocalProfile()
    p = hp.Profile(**DUNE_PROFILE_KW)
    hp.configure(p)
    return p
