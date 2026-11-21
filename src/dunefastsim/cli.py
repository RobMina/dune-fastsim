"""``dunefastsim`` command line: describe, steps-check, steps-convert, ndlar-config."""

from __future__ import annotations

import argparse
import json
import sys


def _describe(_args) -> int:
    from dunefastsim import __version__, charge, dtype, ldm, light

    out = {"dunefastsim": __version__, "device_float": dtype.resolve(),
           "charge (JAXTPC)": charge.status(), "light (lumen)": light.status(), "ldm (bdnmc-dune)": ldm.status()}
    print(json.dumps(out, indent=2))
    return 0


def _steps_check(args) -> int:
    from dunefastsim.steps import check

    print(json.dumps(check(args.path), indent=2))
    return 0


def _steps_convert(args) -> int:
    from dunefastsim.steps.edepsim import convert

    off = [float(x) * 1e3 for x in args.offset_m.split(",")] if args.offset_m else (0.0, 0.0, 0.0)
    convert(args.root, args.out, detector=args.detector, offset_mm=off, max_events=args.max_events)
    return _steps_check(argparse.Namespace(path=args.out))


def _steps_validate(args) -> int:
    import yaml

    from dunefastsim.steps import validate

    out = {"energy": validate.energy(args.path, args.max_events)}
    for cfg in args.config:
        with open(cfg) as f:
            out[f"containment {cfg}"] = validate.containment(args.path, yaml.safe_load(f), args.max_events)
    print(json.dumps(out, indent=2))
    return 0


def _ndlar_config(args) -> int:
    import yaml

    from dunefastsim.geometry import boxes_from_gdml, jaxtpc_config, layout

    boxes = boxes_from_gdml(args.gdml)
    print(json.dumps(layout(boxes)), file=sys.stderr)
    with open(args.template) as f:
        template = yaml.safe_load(f)
    with open(args.out, "w") as f:
        yaml.safe_dump(jaxtpc_config(template, boxes), f, sort_keys=False)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="dunefastsim")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("describe", help="versions and which upstream pieces are installed").set_defaults(fn=_describe)
    c = sub.add_parser("steps-check", help="check a step-record HDF5 against the schema")
    c.add_argument("path")
    c.set_defaults(fn=_steps_check)
    v = sub.add_parser("steps-convert", help="edep-sim ROOT -> step record (needs PyROOT + edep-sim I/O)")
    v.add_argument("root")
    v.add_argument("out")
    v.add_argument("--detector", default="volTPCActive")
    v.add_argument("--offset-m", default="", help="x,y,z of the detector-frame origin in the world frame (m)")
    v.add_argument("--max-events", type=int, default=None)
    v.set_defaults(fn=_steps_convert)
    w = sub.add_parser("steps-validate", help="energy sums and containment in JAXTPC configs' drift volumes")
    w.add_argument("path")
    w.add_argument("--config", action="append", default=[], help="JAXTPC detector config YAML (repeatable)")
    w.add_argument("--max-events", type=int, default=None)
    w.set_defaults(fn=_steps_validate)
    g = sub.add_parser("ndlar-config", help="JAXTPC ND-LAr config with volumes from a GDML (needs [ldm])")
    g.add_argument("gdml")
    g.add_argument("--template", required=True, help="JAXTPC's config/dune_ndlar_config.yaml")
    g.add_argument("--out", required=True)
    g.set_defaults(fn=_ndlar_config)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
