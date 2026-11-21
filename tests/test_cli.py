import json

from dunefastsim.cli import main


def test_describe(capsys, monkeypatch):
    monkeypatch.delenv("JAXTPC_ROOT", raising=False)
    assert main(["describe"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["charge (JAXTPC)"]["importable"] is False and "device_float" in out
