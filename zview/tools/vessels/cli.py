"""`zview-vessels`: blood-vessel segmentation and rendering.

    zview-vessels segment INPUT --out vessels.zarr [...]      pre-masked volume -> mask Zarr
    zview-vessels fullres render.json --zarr rec.zarr --out vessels.zarr [...]
    zview-vessels render vessels.zarr --out DIR [--image INPUT | --config render.json] [...]

Each subcommand has --help.
"""

import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "segment":
        from zview.tools.vessels import segment as m
    elif cmd == "fullres":
        from zview.tools.vessels import fullres as m
    elif cmd == "render":
        from zview.tools.vessels import render as m
    else:
        raise SystemExit(f"unknown command {cmd!r}\n\n{__doc__}")
    return m.main(rest)


if __name__ == "__main__":
    main()
