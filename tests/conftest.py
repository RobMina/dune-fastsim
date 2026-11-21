import os
from pathlib import Path

import pytest


@pytest.fixture
def jaxtpc_root():
    """A JAXTPC checkout from JAXTPC_ROOT, or skip (JAXTPC is not vendored)."""
    r = os.environ.get("JAXTPC_ROOT")
    if not r or not (Path(r) / "tools" / "loader.py").is_file():
        pytest.skip("JAXTPC_ROOT not set")
    return Path(r)
