import os

import numpy as np
import pytest

from dunefastsim import dtype, profile, site


def test_dtype_default_and_fail_closed(monkeypatch):
    monkeypatch.delenv("DUNE_DEVICE_FLOAT", raising=False)
    assert dtype.resolve() == "float64"
    monkeypatch.setenv("DUNE_DEVICE_FLOAT", "float32")
    assert dtype.compute_dtype() == np.float32
    monkeypatch.setenv("DUNE_DEVICE_FLOAT", "bfloat16")
    with pytest.raises(ValueError):
        dtype.resolve()


def test_profile_stand_in():
    p = profile.install()
    assert p.name == "dune" and p.dtype_env == "DUNE_DEVICE_FLOAT"


def test_site_env_rules(tmp_path, monkeypatch):
    env = tmp_path / "site.env"
    env.write_text("export A_DFS=1 # c\nA_DFS=2\nB_DFS=\nC_DFS='x y'\n")
    monkeypatch.delenv("A_DFS", raising=False)
    monkeypatch.setenv("C_DFS", "keep")
    site.load_dotenv(env)
    assert os.environ["A_DFS"] == "2" and "B_DFS" not in os.environ and os.environ["C_DFS"] == "keep"
