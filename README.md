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

## Status (v0.0.11)

| Piece | State |
|---|---|
| `dtype` / `profile` / `site` | done; `DUNE_DEVICE_FLOAT` = `float32` or `float64` (default), other values fail |
| `steps` schema, writer, checker | done; JAXTPC's own loader reads the files (test, with `JAXTPC_ROOT`) |
| `steps.edepsim.convert` (edep-sim ROOT, PyROOT) | validation only (Geant4 reference); run on a 2000-event edep-sim sample: summed deposits match edep-sim's own check exactly |
| `steps.validate`, `dunefastsim steps-validate` | energy sums and containment of steps in a JAXTPC config's drift volumes |
| `transport.parametric` (option A) | GFLASH parametric showers in liquid argon plus a leading track, a fluctuation ramp, a dE/dx spread and a photon conversion delay; differentiable, f32/f64; compared with Geant4 (below) |
| `transport.compare` | option A or B against a Geant4 reference in the step format: deposited fraction, longitudinal and radial profiles, dE/dx, dE/dx over fixed lengths (3.72 mm pixel pitch, 10 mm), per energy bin |
| `transport.emphysics` + `transport.cascade` (option B) | condensed-history electron/photon cascade in JAX: e± physics from [hep-simkernels](https://github.com/RobMina/hep-simkernels) up to 1 GeV (Moller/Bhabha, Seltzer-Berger, Mott-corrected transport cross section, Urban fluctuations; Geant4's option-0 G4UniversalFluctuation from v0.0.5), Tsai bremsstrahlung above, positron annihilation in flight (v0.0.8), Klein-Nishina Compton, pair production (Geant4's total cross section below 30 GeV, v0.0.9), photoelectric effect, photonuclear absorption (v0.0.10) (Livermore EPICS2014, v0.0.6); mixed multiple scattering (soft Gaussian-like plus hard screened-Rutherford collisions with the Mott factor, v0.0.7) with lateral displacement, modified-Tsai emission angles; fixed-capacity particle and deposit buffers with overflow accounting; energy conserved per shower; many showers share one buffer (new primaries enter as lanes free up); extra `[transport]` |
| `geometry` JAXTPC volumes from the GDML | done; on DUNE's hall GDML v2026.03.06 every Geant4 step in the active argon lands in exactly one generated drift volume |
| `charge` (JAXTPC) | option B showers and Geant4 step files into JAXTPC's `DetectorSimulator` (host-side volume split, JAXTPC in float32); `charge.compare` runs both through charge and light, and optionally the pixel response (v0.0.11, forward only) |
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
float64, v0.0.8 defaults: longest e± step 10 mm, Geant4's option-0 fluctuation model, delta rays from 0.22 MeV (all
three as in the Geant4 reference), mixed multiple scattering, positron annihilation in flight, Geant4's pair cross
section, photonuclear sink (v0.0.10).

| | pi0 sample, Geant4 | pi0 sample, option B | eta sample, Geant4 | eta sample, option B |
|---|---|---|---|---|
| deposited fraction | 0.740 | 0.744 | 0.714 | 0.715 |
| mean depth (X0) | 4.79 | 4.78 | 5.97 | 5.94 |
| within 1 Moliere radius | 0.865 | 0.862 | 0.867 | 0.863 |
| within 2 Moliere radii | 0.948 | 0.947 | 0.949 | 0.948 |

dE/dx summed over fixed lengths of e± track (pi0 sample; eta sample the same within 0.01 MeV/cm), in MeV/cm:

| segments | Geant4 MPV | Geant4 q10 / q50 / q90 | option B MPV | option B q10 / q50 / q90 |
|---|---|---|---|---|
| all tracks, 3.72 mm (pixel pitch) | 1.838 | 1.640 / 1.848 / 2.099 | 1.809 | 1.624 / 1.835 / 2.102 |
| all tracks, 10 mm | 1.840 | 1.677 / 1.847 / 2.035 | 1.822 | 1.669 / 1.842 / 2.048 |
| primary, first 5 cm, 3.72 mm | 1.884 | 1.743 / 1.904 / 2.112 | 1.871 | 1.725 / 1.900 / 2.145 |

**Step length.** Geant4 takes long steps for fast electrons in argon (the primary's: median 4.8 mm, energy-weighted
mean 13 mm) and a step's energy is spread evenly along it, so the reference has no energy-loss fluctuations below
about a centimetre. With 2 mm steps (v0.0.4) option B resolved them and its pixel-pitch MPV came out 5.6 % below
Geant4's; with 20 mm it reproduces Geant4 at every length, but steps that long cross active-volume edges whole
(deposited fraction 0.898 vs 0.937 for 1 GeV electrons); 10 mm (the default) leaves 1.6 %. Checked with Geant4
directly and ruled out on the way: the fluctuation model (the two-level G4UrbanFluctuation of hepsimkernels gives the
same dE/dx; `--fluct urban`), the Geant4 version (the model is identical in 11.3.2 and 11.4.0) and edep-sim's merging
of steps into 3 mm segments (turning it off changes nothing).

**Passive material.** Option B has no geometry: everything is liquid argon, and deposits outside the drift volumes
are dropped. Between modules along the beam the reference has 2 x 1.03 cm of ArCLight (plastic), 2 x 0.635 cm of G10
and 1 cm of inactive argon (0.19 X0, against 0.31 X0 of argon in option B). Measured with 500 electrons of 1 GeV whose
showers peak at a module wall or mid-module: the wall costs Geant4 2.70 points of deposited energy and option B 2.98,
and shortens option B's showers by a further 0.08 X0 (mean depth 4.26 vs 4.45 X0 at the wall, 4.10 vs 4.21
mid-module). Smaller than option B's other residuals (deposited fraction 0.6-1 point high and 1 Moliere radius
containment 1 point high at every energy); 2 mm steps do no better at the wall.

**Multiple scattering (v0.0.7).** Up to v0.0.6 each step drew one screened-Rutherford (Wentzel) deflection with its
mean matched to the transport mean free path: right tail, but in the many-collision regime a typical angle several
times below the Gaussian core, so showers came out 1-1.5 points too contained within 1 Moliere radius (outside the
0.3-point spread between Geant4's EM options 0, 3 and 4). Option B now uses the mixed scheme of hepsimkernels.emsc:
collisions below a boundary angle summed into one Gaussian-like deflection, plus a Poisson number of hard
screened-Rutherford collisions above it (Mott factor, nuclear form factor above 100 MeV), at most 2 expected and 6
sampled per step. Per-step angles agree with hepsimkernels' own sampler within 6 % (median) and 10 % (90 %) from 1 MeV
to 1 GeV; containment within 1 Moliere radius went from 1-1.2 points high to within 0.5 points.

**Positron annihilation in flight (v0.0.8).** Positrons now also annihilate on atomic electrons before stopping
(Heitler cross section, two photons isotropic in the centre-of-mass frame: Geant4's G4eeToTwoGammaModel, from
hepsimkernels.eannihil). Their share of the deposited energy went from 26.8 % to 25.7 % (Geant4 25.8 %) on 1 GeV
electrons; deposited fraction from 1.4 / 0.7 points high to 1.0 / 0.55 (pi0 / eta samples), mean depth from 0.06 /
0.09 X0 short to 0.05 / 0.02.

**Pair production (v0.0.9).** Option B's screened Bethe-Heitler integral ran 2-3 % above Geant4's pair cross section
from 10 to 100 MeV. Below 30 GeV option B now uses Geant4's own total cross section (the parametrisation of
G4PairProductionRelModel, Geant4's option 0), keeping its own energy sharing. Mean depth now agrees within the samples'
statistics (0.02 X0 on 2000 showers; a 500-shower mean carries about 0.07 X0 of noise).

**Nuclear reactions (v0.0.10).** The Geant4 reference (QGSP_BERT) has photonuclear and electronuclear reactions: 16 %
of its showers make neutrons and other nuclear products. Rerun with them switched off (`/physics_lists/em/GammaNuclear
false`, `ElectroNuclear false`), Geant4 deposits 0.48 points more of 1 GeV showers' energy, two thirds of option B's
excess then. Option B now absorbs photons on argon nuclei with Geant4's evaluated cross section (G4PARTICLEXS 4.2
`gamma/inel18` below 130 MeV, giant dipole resonance 37 mb at 17 MeV; a rough curve above 150 MeV, where 5 % of the
reference's reactions are) and deposits 20 % of the absorbed photon's energy where it is absorbed; the other 80 % goes
to a per-shower `nuclear_mev` total, so deposits + `nuclear_mev` + `lost_mev` still equal the primary energy. The 80 %
is measured: Geant4's photons absorbed by nuclei carry 0.62 % of the energy, and switching the reactions off returns
0.48 points of visible energy. Electronuclear reactions (5 % of the reference's) are not modelled. Deposited fraction
now: mid-module 0.960 vs 0.958, at a wall 0.931 vs 0.931, full samples +0.5 / +0.07 points (pi0 / eta). Option B has the photoelectric effect (Geant4's option-0 Livermore
model, EPICS2014 data for argon) from v0.0.6; lowering its photon cut from 0.1 MeV to the reference's 4.4 keV
(`--kcut 0.00441638 --k-track 0.004`) changes none of these, so the default stays 0.1 MeV (30 % faster).

Speed, 64 showers per call in one buffer, no energy lost to buffer overflow:

| | pi0 sample | eta sample (higher energies) |
|---|---|---|
| 4 CPU cores, float64, 10 mm steps, v0.0.10 physics | 0.10 s per shower | 0.34 s |
| 4 CPU cores, float64, 10 mm steps, mixed scattering, annihilation in flight (v0.0.8) | 0.13 s | 0.40 s |
| 4 CPU cores, float64, 10 mm steps, mixed scattering (v0.0.7) | 0.18 s | 0.56 s |
| 4 CPU cores, float64, 10 mm steps, Wentzel scattering (v0.0.5) | 0.11 s | 0.34 s |
| 4 CPU cores, float64, 2 mm steps (8192 lanes, v0.0.4) | 0.37 s | 1.27 s |
| one RTX A6000, float32, 2 mm steps (v0.0.4) | 17 ms | 61 ms |

One shower at a time on CPU took 3-5 s. Float32 on the GPU gave the same physics as float64 on CPU (deposited
fraction 0.751 vs 0.752, within 1 Moliere radius 0.872 vs 0.873, dE/dx quantiles within 0.01 MeV/cm). With 10 mm
steps more of a shower's particles are alive at once: the buffer needs about 1.4 x as many lanes (peak 11245).

### Option B through JAXTPC (v0.0.11)

Same electrons, same JAXTPC ND-LAr configuration (generated from the hall GDML, EMB recombination), option B's
deposits and Geant4's steps each through JAXTPC's charge and light (`python -m dunefastsim.charge.compare`):

| | 1 GeV, mid-module (500) | pi0 sample (2000) |
|---|---|---|
| energy in drift volumes, option B / Geant4 | 1.002 | 1.007 |
| charge per MeV, Geant4 (e-) | 28,061 | 28,069 |
| charge per MeV, option B / Geant4 | 1.003 | 1.003 |
| scintillation photons per MeV, option B / Geant4 | 0.997 | 0.997 |
| per-event charge-per-MeV ratio, 10 / 50 / 90 % | 1.000 / 1.002 / 1.005 | 0.998 / 1.002 / 1.007 |

The charge per MeV is the recombination survival, which depends on each deposit's dE/dx, so it tests the deposit
lengths as well as the energies. Before v0.0.11 option B came out 1.4 % low: an electron that fell below the 0.1 MeV
tracking cut booked its remaining energy over its last step alone (often 0.1 mm), which recombination saw as very
high dE/dx (5.9 % of option B's energy above 10 MeV/cm). Each deposit's length now covers the energy it holds: a
stopping electron's step plus its residual range, an absorbed photon's photoelectron range, and the visible part of a
photonuclear absorption the range of an electron of that energy. Option B now has 0.08 % of its energy above
10 MeV/cm against Geant4's 1.6 % (Geant4 follows electrons to 1 keV and resolves their dense ends), with the same
charge per MeV overall. The fixed-length dE/dx and shower numbers above are unchanged by it.

Cost on 4 CPU cores: charge and light 0.6 s per event. The full pixel response of one ND-LAr event takes about 12 min
and 44 GB on CPU (sparse pixel hits), so a pixel-level comparison over many events is for a GPU. This chain is not
yet fully on the device: JAXTPC's volume split and padding (`build_deposit_data`) run on the host between option B and
JAXTPC, and JAXTPC's differentiable path (`forward_segments`) is wire-only at the pinned commit.

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
  charge/     JAXTPC adapter, compare.py (B vs Geant4 through JAXTPC)      light/  lumen adapter      ldm/  bdnmc-dune adapter
  cli.py
tests/        synthetic pytest (JAXTPC tests skip without JAXTPC_ROOT)
container/    Apptainer def for the JAX host (stub)
```

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
