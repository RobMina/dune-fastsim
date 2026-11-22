"""On-device particle transport in liquid argon.

- ``parametric``: option A, GFLASH-style parametric showers (differentiable baseline).
- ``emphysics`` + ``cascade``: option B, a condensed-history electron/photon Monte Carlo in JAX (B0 + B1).
"""

import os as _os

# hep-simkernels reads its device float at import; keep it equal to dune-fastsim's (explicit dtypes are passed anyway).
_os.environ.setdefault("HEPSIMK_DEVICE_FLOAT", _os.environ.get("DUNE_DEVICE_FLOAT") or "float64")
