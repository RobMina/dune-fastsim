# dune-fastsim

JAX fast simulation of the DUNE Near Detector's liquid-argon TPC (ND-LAr) for light-dark-matter studies: charge
readout through [JAXTPC](https://github.com/DeepLearnPhysics/JAXTPC), light through
[lumen](https://github.com/OmarAlterkait/lumen) and [lumen-tpc](https://github.com/OmarAlterkait/lumen-tpc), fed by
edep-sim energy deposits from [bdnmc-dune](https://github.com/RobMina/bdnmc-dune) or any other edep-sim job.

This package **depends on** those projects and vendors none of them. What it adds is the glue between them:

- a **step record** (`dunefastsim.steps`): JAXTPC's input HDF5 layout, a writer, a checker, and a converter from
  stock edep-sim ROOT output;
- **one ND-LAr frame** (`dunefastsim.geometry`): JAXTPC's drift-volume configuration generated from the same GDML the
  events were simulated in, so charge and light agree on where the detector is;
- adapters for charge (`dunefastsim.charge`), light (`dunefastsim.light`) and LDM event generation
  (`dunefastsim.ldm`).

Public material only: no collaboration-internal geometry, configuration or results. Site paths come from
`site.env` (see `site.env.example`), never from the code.

## Status (v0.0.1, scaffold)

| Piece | State |
|---|---|
| `dtype` / `profile` / `site` | done; `DUNE_DEVICE_FLOAT` = `float32` or `float64` (default), other values fail |
| `steps` schema, writer, checker | done; JAXTPC's own loader reads the files (test, with `JAXTPC_ROOT`) |
| `steps.edepsim.convert` (edep-sim ROOT, PyROOT) | written, **not run yet**: needs ROOT + edep-sim I/O (the bdnmc-dune image) |
| `geometry` JAXTPC volumes from the GDML | done on synthetic boxes; reproduces JAXTPC's drift directions; GDML path needs `[ldm]` |
| `charge` (JAXTPC) | status check only; no simulation run yet |
| `light` (lumen 0.2 + lumen-tpc `dune_ndlar`) | status check only |
| `ldm` (bdnmc-dune) | status check only |
| `container/dune-fastsim-jax.def` | stub, not built |

### Known issue: two ND-LAr layouts

JAXTPC's `config/dune_ndlar_config.yaml` puts 5 modules across x and 7 along the beam (z). DUNE's ND hall geometry
(`nd_hall_with_lar_tms_sand_stt1_v2026.03.06.gdml`, ND_Production) has 7 across and 5 along, as the ND CDR. Use
`dunefastsim ndlar-config` to generate the volumes from the GDML the events came from instead of the shipped config.

## Install

Python 3.11 to 3.13 (JAX 0.6.2 and lumen 0.2 have no 3.14 wheels).

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e '.[test,jax]'
.venv/bin/python -m pytest -q
.venv/bin/dunefastsim describe
```

Extras: `[ldm]` (bdnmc-dune, also the GDML reader), `[light]` (lumen-tpc and lumen 0.2; Linux, CUDA 12 for full
detectors).

**JAXTPC** has no licence file yet, so it is not an extra and nothing of it is redistributed here. It supports only
an editable install from a checkout:

```bash
git clone https://github.com/DeepLearnPhysics/JAXTPC.git "$JAXTPC_ROOT"
git -C "$JAXTPC_ROOT" checkout 50ed555ec69adf1c5a7fdf67af81c544a751dd2e
.venv/bin/pip install -e "$JAXTPC_ROOT"
```

With `JAXTPC_ROOT` set, the tests also run JAXTPC's loader on a written step file and compare the generated volumes
with JAXTPC's own ND-LAr config.

## Use

```bash
# edep-sim ROOT -> step record, in the detector frame (where PyROOT and edep-sim's I/O library are)
dunefastsim steps-convert events.edepsim.root events.steps.h5 --offset-m 0,-0.668713,6.657559
dunefastsim steps-check events.steps.h5

# JAXTPC ND-LAr config whose volumes come from the GDML the events were simulated in
dunefastsim ndlar-config nd_hall_with_lar_tms_sand_stt1_v2026.03.06.gdml \
    --template "$JAXTPC_ROOT/config/dune_ndlar_config.yaml" --out ndlar_v2026.03.06.yaml
```

The offset is the centre of ND-LAr's active argon in that GDML's world frame (bdnmc-dune's
`python -m bdnmc_dune gdml-frame <gdml>` reads it).

## Layout

```text
src/dunefastsim/
  dtype.py profile.py site.py   device float mode, experiment profile, site.env loader
  steps/      schema.py write.py edepsim.py
  geometry/   ndlar.py
  charge/     JAXTPC adapter      light/  lumen adapter      ldm/  bdnmc-dune adapter
  cli.py
tests/        synthetic pytest (JAXTPC tests skip without JAXTPC_ROOT)
container/    Apptainer def for the JAX host (stub)
```

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
