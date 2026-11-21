"""dune-fastsim: JAX fast simulation of the DUNE Near Detector (ND-LAr charge and light).

Subpackages:

- ``steps``    : the step record (JAXTPC's HDF5 layout) and the edep-sim ROOT converter
- ``geometry`` : one ND-LAr frame; JAXTPC volume configuration generated from the GDML
- ``charge``   : adapter over JAXTPC (not vendored)
- ``light``    : adapter over lumen + lumen-tpc (not vendored)
- ``ldm``      : thin orchestration over bdnmc-dune (not vendored)
"""

__version__ = "0.0.1"
