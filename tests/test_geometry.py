import numpy as np
import pytest

from dunefastsim.geometry import Boxes, boxes_from_jaxtpc, jaxtpc_volumes, layout
from dunefastsim.geometry.ndlar import drift_directions

# DUNE v2026.03.06-like ND-LAr: 7 modules across x (14 drift columns of 468 mm), 5 along z (957 mm), y 2996 mm
HALF = np.array([234.0, 1498.0, 478.5])


def _dune_like(offset=(0.0, -668.713, 6657.559)):
    xs = []
    for m in range(7):
        cx = (m - 3) * 1000.0
        xs += [cx - 3.175 - HALF[0], cx + 3.175 + HALF[0]]
    zs = [(k - 2) * 1000.0 for k in range(5)]
    c = np.array([[x, 0.0, z] for z in zs for x in xs]) + offset
    return Boxes(c, np.tile(HALF, (len(c), 1)))


def test_layout_seven_across_five_along():
    lay = layout(_dune_like())
    assert lay == {**lay, "drift_volumes": 70, "modules_x": 7, "modules_z": 5, "y_rows": 1}


def test_volumes_detector_frame_and_drift():
    vols = jaxtpc_volumes(_dune_like())
    r = np.array([v["geometry"]["ranges"] for v in vols])
    assert len(vols) == 70 and np.allclose(r[:, :, 0].min(0), -r[:, :, 1].max(0), atol=1e-3)  # centred
    d = [v["geometry"]["drift_direction"] for v in vols[:14]]
    assert d == [-1, 1] * 7  # first z row, increasing x: lower volume of each module drifts to -x
    assert vols[0]["readout"]["pixel_shape"] == [805, 257]


def test_drift_rejects_odd_columns():
    with pytest.raises(ValueError):
        drift_directions([0.0, 500.0, 1000.0])


def test_reproduces_jaxtpc_config(jaxtpc_root):
    """Our generator, fed JAXTPC's own ND-LAr boxes, gives back its drift directions; and its layout is 5 x 7."""
    import yaml

    cfg = yaml.safe_load(open(jaxtpc_root / "config" / "dune_ndlar_config.yaml"))
    boxes = boxes_from_jaxtpc(cfg)
    lay = layout(boxes)
    assert (lay["modules_x"], lay["modules_z"]) == (5, 7)  # transposed against DUNE's geometry (plan page)
    ours = {tuple(np.round(np.array(v["geometry"]["ranges"])[:, 0], 2)): v["geometry"]["drift_direction"]
            for v in jaxtpc_volumes(boxes, offset_mm=(0.0, 0.0, 0.0))}
    theirs = {tuple(np.round(np.array(v["geometry"]["ranges"])[:, 0], 2)): v["geometry"]["drift_direction"]
              for v in cfg["volumes"]}
    assert ours == theirs
