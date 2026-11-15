# Batch jobs: writing the JSON

A batch render is **one JSON file plus one command**. The JSON says what data
to render, how it should look and what to make; the command runs it, in the
foreground or in the background on a machine with a GPU. This page goes
through writing that file by hand, step by step, with complete examples you
can copy.

```console
$ zview-render run job.json --dry-run      # check the file, render nothing
$ zview-render run job.json                # render
```

## 1. Get a starting file

Any of these gives you a valid file to edit:

- **From the viewer:** open the dataset, **File › Render 3D…**, set things up,
  then **Save config**. The file is `render.json` in the output folder. You can
  also send the 3D view's look and camera there first with **▶ job**.
- **From the command line:** `zview-render prepare DATA --out FOLDER`. This
  writes `FOLDER/render.json` with the common keys (`null` means "use the
  preset's value") and the cached volume.
- **By hand:** create a new file. The smallest valid job is one line, below.

## 2. The smallest job

```json
{ "source": "/data/sample/rec.zarr" }
```

Save it as `/work/sample/job.json` and run:

```console
$ zview-render run /work/sample/job.json
```

With nothing else specified you get the defaults: the `generic` preset
(any data; bone colours, studio light), 32 stills (8 directions × 4
elevations) and a 12 s turntable at 1920×1080.

On the first run, zview reads a render-sized copy of the data
(`volume_<name>_max768.npy`) **into the folder of the JSON** and writes the
keys `volume`, `factor`, `volume_source` and `volume_max_axis` back into the
file. Later runs reuse that copy. It's read again automatically if you
change `source` or `max_axis`. The results go into the same folder:

| Output | Where |
|---|---|
| stills | `renders/stills_<look>/az<deg>_el<deg>.png` |
| turntable | `renders/turntable_<look>.mp4` |
| tour | `tour/tour.mp4` (+ `tour/keyframes/`) |
| surface | `surface/stills/`, `surface/turntable_surface.mp4` |
| exact settings used | `render.effective.json` |

`--out OTHER_FOLDER` puts the results somewhere else; the cached volume stays
next to the JSON.

## 3. Check it before you run it

```console
$ zview-render run job.json --dry-run
```

This prints the complete merged settings and lists problems: misspelled or
unknown keys (with a suggestion), unknown look, lighting or preset names,
malformed values, missing `source` for a tour. It renders nothing, and exits
with code 1 if there is a problem, 0 if the file is OK:

```text
CONFIG WARNING: unknown key 'lok' (did you mean 'look'?) — it is ignored
1 problem(s) found
```

Every real run prints the same warnings at the start, so they also appear in
background logs.

## 4. JSON rules that trip people up

- **No comments.** Put notes in keys starting with `_`, such as `"_comment"`;
  they're ignored.
- **Double quotes** around every key and string. `true`, `false` and `null`
  are lowercase. **No comma after the last item.**
- `null` (or leaving a key out) means "use the preset's value".
- **Lists** are written `[a, b]`. Colours are `[r, g, b]` with values from 0 to
  1.
- **Coordinates are full-resolution voxel indices**, in the order
  **z, y, x**: the numbers shown in the viewer's *Position* panel and status
  bar. That applies to `z_range`, `cylinder` and tour `regions`, whatever
  resolution is rendered.
- **Label classes are keyed by strings**: `"70": {...}`, not `70: {...}`.
- **Relative paths** (`source`, `volume`) are relative to the JSON file's
  folder.

## 5. What to put in

Settings are layered. Each layer overrides the one before:

```text
built-in defaults  <  preset(s)  <  your JSON  <  --set key=value on the command line
```

So start from a preset and set only what you want different. The keys you'll
use most:

| Purpose | Keys | Example |
|---|---|---|
| The data | `source`, `hdf5_dataset` (HDF5 only) | `"source": "/data/rec.zarr"` |
| Detail | `max_axis`: longest side of the volume read for rendering | `"max_axis": 1536` |
| Starting point | `preset`: one name or a list | `"preset": ["interior", "presentation-4k"]` |
| Look | `look`, `lighting`, `curve`, `opacity`, `background` | `"look": "jade"`, `"lighting": "soft"` |
| What to make | `outputs`, `size`, `frames`, `fps` | `"outputs": ["stills", "turntable"]` |
| Camera | `elev`, `elevations`, `azimuths`, `azimuth`, `zoom` | `"elevations": [25]`, `"zoom": 1.3` |
| Intensity window | `lo_pct`/`hi_pct` (percentiles) or `window_values` | `"lo_pct": 70` |
| Crop | `z_range`, `cylinder` | `"z_range": [1700, 13800]` |
| Text | `title`, `subtitle`, `credit` | `"credit": "Imaging group"` |
| Tour | `regions`, `tour` (`speed`, `scroll`, `zoom`, `region_size`) | see example 4 |
| Label maps | `labels` | see example 5 |

The names to choose from:

- **Presets:** `generic`, `interior`, `multiphase`, `metal`, `xray`, `labels`,
  `tomo-brain`, `tomo-surface`; quality modifiers `draft`,
  `presentation-4k`, `publication`. See [](presets.md), or run
  `zview-render presets`.
- **Looks:** `organic`, `tissue`, `bone`, `ivory`, `grays`, `gold`, `copper`,
  `jade`, `ice`, `ember`, `coolwarm`, `xray`, `viridis`, `inferno`, `magma`,
  `plasma`, `cividis`, `turbo`, or any matplotlib colormap name. See the
  [gallery](looks.md).
- **Lighting:** `studio`, `soft`, `dramatic`, `top`, `flat`.
- **Curves:** `solid`, `soft`, `layered`, `glass`, `mip`.
- **Outputs:** `stills`, `turntable`, `tour`, `surface`.

Every key is listed in the [](config.md).

## 6. Examples

Each example is a complete file. Change the paths and run it with
`zview-render run FILE`.

### 1 — A quick look at a new dataset

Small, fast, four views. Use it to check the window and crop before a long job.

```json
{
  "source": "/data/sample/rec.zarr",
  "preset": ["generic", "draft"],
  "outputs": ["stills"],
  "elevations": [25]
}
```

### 2 — A 4K presentation turntable

```json
{
  "source": "/data/sample/rec.zarr",
  "preset": ["interior", "presentation-4k"],
  "outputs": ["turntable"],
  "look": "ember",
  "lighting": "soft",
  "elev": 20,
  "frames": 720
}
```

720 frames at 30 fps is a 24 s video.

### 3 — Publication stills from a chosen viewpoint

Square, on white, from one direction. To find the viewpoint, rotate the 3D
pane in the viewer, press **▶ job**, and copy `azimuth`, `elevations` and
`zoom` from *Advanced overrides*.

```json
{
  "source": "/data/sample/rec.zarr",
  "preset": ["metal", "publication"],
  "look": "gold",
  "azimuth": 135,
  "azimuths": 1,
  "elevations": [30],
  "zoom": 1.25,
  "size": "3000x3000"
}
```

### 4 — A guided tour of a tomography scan

The tour flies into each region and shows the real full-resolution slices.
Read the region centres off the viewer's status bar (z, y, x) or off
`side_views_grid.png`.

```json
{
  "source": "/data/sample/rec.zarr",
  "preset": "tomo-brain",
  "outputs": ["stills", "turntable", "tour"],
  "pixel_size_um": 1.88,
  "z_range": [1700, 13800],
  "cylinder": [6186, 5931, 6270, 6267, 5270],
  "title": "Mouse brain",
  "subtitle": "synchrotron micro-CT, 1.88 µm voxels",
  "credit": "Imaging group, Example Institute",
  "regions": [
    {"name": "Cerebellum",  "z": 3800, "y": 6144, "x": 4200},
    {"name": "Hippocampus", "z": 7200, "y": 5100, "x": 6900}
  ],
  "tour": {"speed": 0.7, "scroll": 150}
}
```

`cylinder` is `[CY, CX, R]`, or for a tilted tube
`[CY_TOP, CX_TOP, CY_BOTTOM, CX_BOTTOM, R]`, in full-resolution pixels.

### 5 — A label map, showing only some classes

For segmentations ([](../labels.md)), list the classes you want to change.
The keys are the class values as strings. Everything you don't list keeps the
default from the dataset's metadata. Here the bulk tissue becomes faint glass,
and vessels and nuclei are opaque:

```json
{
  "source": "/data/sample/segmentation.zarr",
  "preset": "labels",
  "outputs": ["stills", "turntable"],
  "labels": {
    "3":   {"visible": true,  "opacity": 0.04},
    "4":   {"visible": true,  "opacity": 0.04},
    "5":   {"visible": false},
    "7":   {"visible": true,  "color": [0.35, 0.40, 0.95]},
    "50":  {"visible": true,  "color": [0.65, 0.12, 0.12]},
    "70":  {"visible": true,  "color": [0.90, 0.12, 0.10]},
    "255": {"visible": false}
  }
}
```

Each class can set `name`, `color`, `opacity` (0–1; below 0.35 it's drawn as
glass) and `visible`. If the data is a label map and the job doesn't say
otherwise, zview switches to class rendering by itself.

### 6 — A massive dataset on a GPU machine

Raise `max_axis` for more detail. It costs about `max_axis³` bytes of GPU
memory and roughly six times that in RAM (2048 needs about 8 GB of GPU memory
and 50 GB of RAM).

```json
{
  "source": "/data/big/rec.zarr",
  "preset": ["interior", "presentation-4k"],
  "max_axis": 2048,
  "outputs": ["stills", "turntable"]
}
```

Run it in the background and keep a log:

```console
$ nohup zview-render run /work/big/job.json > /work/big/render.log 2>&1 &
$ tail -f /work/big/render.log
```

Reading the volume is the slow part when the dataset has no deep pyramid; the
log shows how much is read. On a GPU node without a desktop, run under a
virtual display: `xvfb-run -a zview-render run …`. With VTK ≥ 9.7 the
rendering still uses the GPU (EGL); with older VTK it falls back to slow
software rendering.

### 7 — The same settings for many datasets

Write one template **without** `source`, then give each dataset its own
folder (so each gets its own cached volume and results):

```json
{
  "_comment": "template.json — shared settings for the whole series",
  "preset": ["interior", "draft"],
  "outputs": ["stills", "turntable"],
  "credit": "Imaging group"
}
```

```bash
#!/bin/bash
# render_all.sh — one folder per dataset under /work/series/
for data in /data/series/*.zarr; do
    name=$(basename "$data" .zarr)
    mkdir -p "/work/series/$name"
    cp template.json "/work/series/$name/render.json"
    zview-render run "/work/series/$name/render.json" --set source="$data" \
        > "/work/series/$name/render.log" 2>&1 || echo "FAILED: $name"
done
```

The same in Python:

```python
from pathlib import Path
import json, shutil
from zview.render.cli import run

template = json.loads(Path("template.json").read_text())
for data in sorted(Path("/data/series").glob("*.zarr")):
    folder = Path("/work/series") / data.stem
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "render.json").write_text(json.dumps({**template, "source": str(data)}, indent=2))
    run(folder / "render.json")
```

## 7. Changing settings without editing the file

`--set key=value` overrides any key for one run; nested keys use a dot, and
values are read as JSON:

```console
$ zview-render run job.json stills --set look=jade --set size=2560x1440
$ zview-render run job.json tour --set tour.speed=0.5 --set 'regions=[{"name":"A","z":100,"y":200,"x":300}]'
$ zview-render run job.json --preset metal --preset draft      # replace the file's presets
```

Listing outputs after the file name (`stills`, `turntable`, `tour`,
`surface`, `all`) overrides `outputs` for that run.

## 8. Job scripts from the viewer

**Save script…** in the Render 3D dialog writes `render_job.py`: the same
JSON, embedded in a small Python script that finds zview by itself. Use it
when the render runs somewhere else:

```console
$ python render_job.py --dry-run                         # check
$ nohup python render_job.py --max-axis 2048 > render_job.log 2>&1 &
```

Its options are `--max-axis`, `--what` (outputs), `--out`, `--set key=value`
and `--dry-run`. To change the settings permanently, edit the JSON block
between `CONFIG = json.loads(r"""` and `""")` in the script, using the same
rules as on this page.

## 9. A template with the common keys

Copy this, delete what you don't need, and fill in the rest. Keys starting
with `_` are notes and are ignored.

```json
{
  "_comment": "zview-render job — see the Batch jobs page in the docs",

  "source": "/data/sample/rec.zarr",
  "max_axis": 768,

  "preset": ["generic"],
  "outputs": ["stills", "turntable"],

  "look": "bone",
  "lighting": "studio",
  "background": "look",

  "lo_pct": 1.0,
  "hi_pct": 99.8,
  "z_range": null,
  "cylinder": null,

  "size": "1920x1080",
  "frames": 360,
  "elev": 20,
  "elevations": [20, 45, -20, 0],
  "azimuths": 8,
  "azimuth": 0,
  "zoom": 1.0,

  "title": "",
  "subtitle": "",
  "credit": "",
  "regions": [],
  "tour": {"speed": 1.0, "scroll": 150, "zoom": 1.8, "region_size": 800}
}
```
