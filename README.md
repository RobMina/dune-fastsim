# dune-fastsim

A fully differentiable, fully on-device JAX simulation of the DUNE Near Detector's liquid-argon TPC (ND-LAr) for
light-dark-matter studies: dark-matter scatters from BdNMC's JAX port (via
[bdnmc-dune](https://github.com/RobMina/bdnmc-dune)), electron and photon transport in liquid argon on the device
(to be written), charge readout through [JAXTPC](https://github.com/DeepLearnPhysics/JAXTPC) and light through
[lumen](https://github.com/OmarAlterkait/lumen) and [lumen-tpc](https://github.com/OmarAlterkait/lumen-tpc). Geant4 is
not part of the chain: edep-sim output is used only as an offline validation reference.

This package **depends on** those projects and vendors none of them. What it adds is the glue between them:

- a **step record** (`dunefastsim.steps`): JAXTPC's input HDF5 layout, a writer, a checker, and a converter from
  stock edep-sim ROOT output;
- **one ND-LAr frame** (`dunefastsim.geometry`): JAXTPC's drift-volume configuration generated from the same GDML the
  events were simulated in, so charge and light agree on where the detector is;
- adapters for charge (`dunefastsim.charge`), light (`dunefastsim.light`) and LDM event generation
  (`dunefastsim.ldm`).

Public material only: no collaboration-internal geometry, configuration or results. Site paths come from
`site.env` (see `site.env.example`), never from the code.

## Status (v0.0.4)

| Piece | State |
|---|---|
| `dtype` / `profile` / `site` | done; `DUNE_DEVICE_FLOAT` = `float32` or `float64` (default), other values fail |
| `steps` schema, writer, checker | done; JAXTPC's own loader reads the files (test, with `JAXTPC_ROOT`) |
| `steps.edepsim.convert` (edep-sim ROOT, PyROOT) | validation only (Geant4 reference); run on a 2000-event edep-sim sample: summed deposits match edep-sim's own check exactly |
| `steps.validate`, `dunefastsim steps-validate` | energy sums and containment of steps in a JAXTPC config's drift volumes |
| `transport.parametric` (option A) | GFLASH parametric showers in liquid argon plus a leading track, a fluctuation ramp, a dE/dx spread and a photon conversion delay; differentiable, f32/f64; compared with Geant4 (below) |
| `transport.compare` | option A or B against a Geant4 reference in the step format: deposited fraction, longitudinal and radial profiles, dE/dx, dE/dx over fixed lengths (3.72 mm pixel pitch, 10 mm), per energy bin |
| `transport.emphysics` + `transport.cascade` (option B) | condensed-history electron/photon cascade in JAX: e± physics from [hep-simkernels](https://github.com/RobMina/hep-simkernels) up to 1 GeV (Moller/Bhabha, Seltzer-Berger, Mott-corrected transport cross section, Urban fluctuations), Tsai bremsstrahlung above, Klein-Nishina Compton, Bethe-Heitler pair; Wentzel multiple scattering with lateral displacement, modified-Tsai emission angles; fixed-capacity particle and deposit buffers with overflow accounting; energy conserved per shower; many showers share one buffer (new primaries enter as lanes free up); extra `[transport]` |
| `geometry` JAXTPC volumes from the GDML | done; on DUNE's hall GDML v2026.03.06 every Geant4 step in the active argon lands in exactly one generated drift volume |
| `charge` (JAXTPC) | status check only; no simulation run yet |
| `light` (lumen 0.2 + lumen-tpc `dune_ndlar`) | status check only |
| `ldm` (bdnmc-dune) | status check only |
| `container/dune-fastsim-jax.def` | stub, not built |

### Option A against Geant4

Same 2000 dark-matter-scattered electrons (34 MeV to 22 GeV) in ND-LAr, same drift volumes, Geant4 (edep-sim) vs
option A: deposited fraction 0.740 vs 0.745 overall and within 0.02 in every energy bin; energy-weighted mean depth
within 0.26 X0 in every bin. Remaining: below 300 MeV the radial profile is too peaked on the axis (no multiple
scattering on the straight leading track), and radial containment is 1-4 % low at all energies. Two constants (the
dE/dx spread and the photon-delay energy scale) were fitted to the same sample: an independent check is pending.
About 4 ms per shower in float32 on 4 CPU cores.

### Option B against Geant4

Full samples (2000 electrons each, 34 MeV to 22 GeV; one from pi0 decays, one from eta decays), same drift volumes,
float64. Delta-ray production threshold 0.22 MeV, the same as the Geant4 reference.

| | pi0 sample, Geant4 | pi0 sample, option B | eta sample, Geant4 | eta sample, option B |
|---|---|---|---|---|
| deposited fraction | 0.740 | 0.752 | 0.714 | 0.720 |
| mean depth (X0) | 4.79 | 4.74 | 5.97 | 5.96 |
| within 1 Moliere radius | 0.865 | 0.873 | 0.867 | 0.874 |

dE/dx summed over fixed lengths of e± track (pi0 sample; eta sample the same within 0.02 MeV/cm), in MeV/cm:

| segments | Geant4 MPV | Geant4 q10 / q50 / q90 | option B MPV | option B q10 / q50 / q90 |
|---|---|---|---|---|
| all tracks, 3.72 mm | 1.84 | 1.64 / 1.85 / 2.10 | 1.74 | 1.59 / 1.81 / 2.18 |
| all tracks, 10 mm | 1.84 | 1.68 / 1.85 / 2.04 | 1.81 | 1.66 / 1.84 / 2.07 |
| primary, first 5 cm, 10 mm | 1.89 | 1.77 / 1.91 / 2.09 | 1.85 | 1.74 / 1.91 / 2.13 |

Segment counts agree within 2.5 %. Over 10 mm the two agree; over one pixel pitch option B's most probable value is
5 % low and its high tail longer. Changing the maximum step from 0.5 to 5 mm moves this little; the energy-loss fluctuation
model is the next thing to check.

Speed, 64 showers per call in one 8192-lane buffer, no energy lost to buffer overflow:

| | pi0 sample | eta sample (higher energies) |
|---|---|---|
| 4 CPU cores, float64 | 0.37 s per shower | 1.27 s |
| one RTX A6000, float32 | 17 ms per shower | 61 ms |

One shower at a time on CPU took 3-5 s. Float32 on the GPU gives the same physics as float64 on CPU (deposited
fraction 0.751 vs 0.752, within 1 Moliere radius 0.872 vs 0.873, dE/dx quantiles within 0.01 MeV/cm). The buffer
was sized for CPU and peaked at half full; larger buffers on the GPU are untried.

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
  steps/      schema.py write.py edepsim.py validate.py
  transport/  parametric.py (option A) emphysics.py cascade.py (option B) compare.py
  geometry/   ndlar.py
  charge/     JAXTPC adapter      light/  lumen adapter      ldm/  bdnmc-dune adapter
  cli.py
tests/        synthetic pytest (JAXTPC tests skip without JAXTPC_ROOT)
container/    Apptainer def for the JAX host (stub)
```

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
