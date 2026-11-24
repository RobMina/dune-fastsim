"""BdNMC rows -> option B primaries on the device (ldm.chain) against bdnmc-dune's HEPEVT path, the one the Geant4
references were simulated from. Needs bdnmc-jax and bdnmc-dune (the [ldm] extra); run on UVA."""

import importlib.util

import numpy as np
import pytest

if not (importlib.util.find_spec("bdnmc_dune") and importlib.util.find_spec("bdnmc_jax")):
    pytest.skip("bdnmc-dune / bdnmc-jax not installed (the [ldm] extra)", allow_module_level=True)

import jax  # noqa: E402

if tuple(int(x) for x in jax.__version__.split(".")[:2]) < (0, 11):
    # The port locks JAX 0.12.0.1; on 0.6.2 one 65536-trial block did not finish in 13 min (UVA job 23406583).
    pytest.skip(f"BdNMC's port needs JAX >= 0.11 (locks 0.12.0.1), not {jax.__version__}", allow_module_level=True)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    import jax

    x64 = jax.config.jax_enable_x64
    from bdnmc_dune import cards, config, hepevt  # bdnmc_jax's import sets 64-bit mode to its precision
    from bdnmc_jax import output, pipeline

    jax.config.update("jax_enable_x64", x64)

    d = tmp_path_factory.mktemp("bdnmc")
    text, run, _ = cards.card_text(config.load(), "pi0", 0.01, seed=11)
    card = d / f"{run}.dat"
    card.write_text(text)
    from dunefastsim.ldm import chain

    with chain.generator_precision():  # the port's precision, whatever 64-bit mode other tests left on
        res = pipeline.events(str(card), 2 * (1 << 16), batch=2, stream=True)
        jax.block_until_ready([r for r, _ in res["device_rows"]])
    info = hepevt.write(output.collect(dict(res)), d / "events.hepevt")
    return card, res, hepevt.read(info["path"])


@pytest.mark.parametrize("gdml_offset", [(0.0, 0.0, 0.0), (0.0, -0.668713, 6.657559)])
def test_primaries_match_the_hepevt_input(generated, gdml_offset):
    import jax.numpy as jnp

    from dunefastsim.ldm import chain

    card, res, events = generated
    origin, rot = chain.frame(card, gdml_offset)
    got = [chain.primaries(r, n, origin, rot, jnp.float64) for r, n in res["device_rows"] if n > 0]
    p = {k: np.concatenate([np.asarray(g[k]) for g in got]) for k in got[0]}
    assert len(p["ke"]) == len(events) > 100
    el = np.array([[float(x) for x in parts[0]] for _, parts in events])  # status pdg links px py pz E m x y z t
    # the HEPEVT file is in the detector-centred frame; a GDML offset moves the vertices by it
    expect = el[:, 11:14] * 10.0 + np.asarray(gdml_offset) * 1000.0
    np.testing.assert_allclose(p["vtx"], expect, rtol=0, atol=1e-4)  # HEPEVT cm, 10 digits
    np.testing.assert_allclose(p["t_ns"], el[:, 14], rtol=1e-9)
    np.testing.assert_allclose(p["ke"], el[:, 9] * 1000.0 - chain.M_E_MEV, rtol=1e-6, atol=1e-6)
    pd = el[:, 6:9] / np.linalg.norm(el[:, 6:9], axis=1, keepdims=True)
    np.testing.assert_allclose(p["dir"], pd, atol=1e-8)
    if gdml_offset == (0.0, 0.0, 0.0):  # detector-centred: inside ND-LAr's 7 x 3 x 5 m active box
        assert np.all(np.abs(p["vtx"]) <= np.array([3500.0, 1500.0, 2500.0]) + 1e-3)
