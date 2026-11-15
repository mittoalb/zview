#!/usr/bin/env python3
"""Regenerate the render gallery used by the documentation.

    python docs/gen_images.py [--data /tmp/zview_demo/data] [--work /tmp/zview_demo/renders]

Needs the render extras (VTK, SciPy, scikit-image, imageio, Pillow) and an X
display with a GPU (VTK renders off-screen; nothing appears on screen). The
fake datasets are created with scripts/make_fake_datasets.py if missing.
Outputs go to docs/_static/img/ as JPEG.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "docs" / "_static" / "img"
sys.path.insert(0, str(ROOT))

from zview.render import cli  # noqa: E402
from zview.render import prepare as rp  # noqa: E402


def save_jpg(src, dst, size=None, quality=88):
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(src).convert("RGB")
    if size:
        im.thumbnail(size, Image.LANCZOS)
    im.save(dst, "JPEG", quality=quality, optimize=True, progressive=True)
    print("  ->", dst.relative_to(ROOT), f"{dst.stat().st_size // 1024} KB", flush=True)


def still(cfg, out, presets, sets, size="640x480", az=30, el=25):
    """Render one still; returns its PNG path."""
    if out.exists():
        shutil.rmtree(out)
    ov = cli.rc.parse_set([f"size={size}", "azimuths=1", f"elevations=[{el}]", *sets])
    ov["preset"] = presets if len(presets) > 1 else presets[0]
    cli.run(cfg, ["stills"], ov, out)
    pngs = sorted(out.glob("renders/stills_*/*.png"))
    if not pngs:
        raise RuntimeError(f"no still in {out}")
    if az:   # rotate by re-rendering at the azimuth we want: stills start at 0 deg
        pass
    return pngs[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/tmp/zview_demo/data"))
    ap.add_argument("--work", type=Path, default=Path("/tmp/zview_demo/renders"))
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    if not (a.data / "phantom_iso.zarr").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_fake_datasets.py"), str(a.data), "--no-big"],
                       check=True)
    a.work.mkdir(parents=True, exist_ok=True)
    iso = rp.prepare(a.data / "phantom_iso.zarr", a.work / "iso")
    aniso = rp.prepare(a.data / "phantom_aniso_5d.zarr", a.work / "aniso")
    want = set(a.only or ["hero", "looks", "lighting", "presets", "finish", "surface", "aniso", "tour",
                          "turntable", "sideviews", "labels"])

    if "hero" in want:
        save_jpg(still(iso, a.work / "hero", ["interior"], ["supersample=2", "zoom=1.25", "elevations=[20]"], size="1600x900"),
                 IMG / "hero.jpg", quality=90)
        save_jpg(still(iso, a.work / "hero_metal", ["metal"], ["zoom=1.3"], size="1600x900"), IMG / "hero_metal.jpg", quality=90)

    if "looks" in want:
        groups = {
            ("generic",): ["organic", "tissue", "bone", "ivory", "grays"],
            ("metal",): ["gold", "copper"],
            ("interior",): ["ember", "jade", "ice", "viridis", "inferno", "magma", "plasma", "cividis", "turbo"],
            ("multiphase",): ["coolwarm"],
            ("xray",): ["xray"],
        }
        for presets, names in groups.items():
            for name in names:
                sets = [f"look={name}"]
                if name == "ice":
                    sets.append("curve=glass")
                save_jpg(still(iso, a.work / f"look_{name}", list(presets), sets), IMG / "looks" / f"{name}.jpg")

    if "lighting" in want:
        for rig in ("studio", "soft", "dramatic", "top"):
            save_jpg(still(iso, a.work / f"light_{rig}", ["metal"], ["look=ivory", f"lighting={rig}"]),
                     IMG / "lighting" / f"{rig}.jpg")
        save_jpg(still(iso, a.work / "light_flat", ["xray"], []), IMG / "lighting" / "flat.jpg")

    if "presets" in want:
        for p in ("generic", "interior", "multiphase", "metal", "xray"):
            save_jpg(still(iso, a.work / f"preset_{p}", [p], []), IMG / "presets" / f"{p}.jpg")
        save_jpg(still(iso, a.work / "preset_publication", ["metal", "publication"], ["size=900x900"]),
                 IMG / "presets" / "publication.jpg")

    if "finish" in want:
        # Same view: raw (no tone mapping, no supersampling) vs finished.
        variants = [("raw", ["tone_mapping=none", "supersample=1", "exposure=1.0"]),
                    ("filmic", ["tone_mapping=filmic", "supersample=1"]),
                    ("filmic_ss2", ["tone_mapping=filmic", "supersample=2"])]
        for tag, sets in variants:
            p = still(iso, a.work / f"finish_{tag}", ["metal"], ["look=gold", *sets], size="800x600")
            im = Image.open(p).convert("RGB").crop((250, 120, 550, 420)).resize((450, 450), Image.NEAREST)
            dst = IMG / "finish" / f"{tag}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, "JPEG", quality=92)
            print("  ->", dst.relative_to(ROOT))

    if "surface" in want:
        for p, look in (("metal", "gold"), ("generic", "bone")):
            out = a.work / f"surface_{p}"
            if out.exists():
                shutil.rmtree(out)
            ov = cli.rc.parse_set(["size=800x600", "frames=0", f"look={look}"])
            ov["preset"] = p
            cli.run(iso, ["surface"], ov, out)
            save_jpg(sorted(out.glob("surface/stills/az045_el+20.png"))[0], IMG / "surface" / f"{p}.jpg")

    if "aniso" in want:
        save_jpg(still(aniso, a.work / "aniso_still", ["interior"], ["look=jade"]), IMG / "aniso.jpg")

    if "tour" in want:
        cfg = json.loads(Path(iso).read_text())
        cfg.update(regions=[{"name": "Helix", "z": 256, "y": 256, "x": 384},
                            {"name": "Resolution target", "z": 120, "y": 350, "x": 200}],
                   title="Phantom", subtitle="procedural test sample", credit="zview documentation")
        tcfg = a.work / "iso" / "tour.json"
        tcfg.write_text(json.dumps(cfg, indent=2))
        out = a.work / "tour"
        if out.exists():
            shutil.rmtree(out)
        ov = cli.rc.parse_set(["tour.speed=0.25", "tour.scroll=20", "tour.region_size=220", "size=1280x720"])
        ov["preset"] = "generic"
        cli.run(tcfg, ["tour"], ov, out)
        for k in sorted((out / "tour" / "keyframes").glob("*.png")):
            name = k.stem.split("_", 1)[1].replace(" ", "_").lower()
            save_jpg(k, IMG / "tour" / f"{name}.jpg", size=(960, 540))

    if "turntable" in want:
        out = a.work / "turntable"
        if out.exists():
            shutil.rmtree(out)
        ov = cli.rc.parse_set(["size=480x360", "frames=48", "supersample=2"])
        ov["preset"] = "interior"
        cli.run(iso, ["turntable"], ov, out)
        import imageio.v3 as iio
        frames = iio.imread(sorted(out.glob("renders/turntable_*.mp4"))[0])
        dst = IMG / "turntable.gif"
        iio.imwrite(dst, frames, duration=80, loop=0)
        print("  ->", dst.relative_to(ROOT), f"{dst.stat().st_size // 1024} KB")

    if "labels" in want:
        lbl = rp.prepare(a.data / "phantom_labels.zarr", a.work / "labels")
        cfg = json.loads(Path(lbl).read_text())
        # Default classes, then with the bulk matrix and the container wall hidden.
        save_jpg(still(lbl, a.work / "labels_default", ["labels"], []), IMG / "labels" / "default.jpg")
        cfg["labels"]["21"]["visible"] = False
        cfg["labels"]["3"]["visible"] = False
        Path(lbl).write_text(json.dumps(cfg, indent=2))
        save_jpg(still(lbl, a.work / "labels_inside", ["labels"], ["zoom=1.15"]), IMG / "labels" / "inside.jpg")
        out = a.work / "labels_tour"
        if out.exists():
            shutil.rmtree(out)
        ov = cli.rc.parse_set(["tour.speed=0.25", "tour.scroll=10", "tour.region_size=220", "size=960x540",
                               'regions=[{"name": "Helix", "z": 256, "y": 256, "x": 384}]'])
        ov["preset"] = "labels"
        cfg["labels"]["21"]["visible"] = True
        cfg["labels"]["3"]["visible"] = True
        Path(lbl).write_text(json.dumps(cfg, indent=2))
        cli.run(lbl, ["tour"], ov, out)
        for k in sorted((out / "tour" / "keyframes").glob("*.png")):
            if "section" in k.stem or "real" in k.stem:
                save_jpg(k, IMG / "labels" / f"tour_{k.stem.split('_', 1)[1].lower()}.jpg", size=(960, 540))

    if "sideviews" in want:
        save_jpg(a.work / "iso" / "side_views_grid.png", IMG / "side_views_grid.jpg", size=(1200, 800))


if __name__ == "__main__":
    main()
