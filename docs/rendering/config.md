# Config reference

A render config is a JSON file, normally `render.json` written by
`zview-render prepare`. Configs from zarr-auto-render (`dataset.json`) still
work: their keys are a subset of these. Any key may also be set with
`--set key=value` (values are parsed as JSON, so `--set z_range=[100,900]`
works; nested keys use dots, e.g. `--set tour.speed=0.5`) or in a preset.

`null` means "not set": the preset or default applies. Relative `volume` and
`source` paths are resolved against the config's folder.

## Dataset

Filled in by `prepare`.

| Key | Default | Meaning |
|---|---|---|
| `source` | — | The original data: a Zarr/N5 store, HDF5 file or TIFF folder. The tour reads full resolution from it |
| `hdf5_dataset` | auto | The dataset path inside an HDF5 file |
| `volume` | — | The cached downsampled volume (`.npy`). If missing, `run` prepares it from `source` |
| `factor` | 8 | Cached-volume step vs full resolution: a number, or `[z, y, x]` |
| `pixel_size_um` | 1.0 | x voxel size, for scale bars |
| `voxel_size_um` | null | `[z, y, x]` full-res voxel size, for anisotropic data |
| `max_axis` | 768 | `prepare`: longest side of the cached volume (pyramid level plus stride) |

## What to make

| Key | Default | Meaning |
|---|---|---|
| `preset` | `"generic"` | A name, a list of names (applied left to right), or a JSON file path |
| `outputs` | `["stills", "turntable"]` | Any of `stills`, `turntable`, `tour`, `surface` |

## Preprocessing

| Key | Default | Meaning |
|---|---|---|
| `preprocess` | `none` | `none` (window the intensities), `tomo` (sample in a container: medium removal, wall fit, Z equalisation) or `labels` (class IDs) |
| `window` | auto | `percentile` (default for `none`) or `otsu` (lower edge Otsu/2; default for `tomo`) |
| `lo_pct`, `hi_pct` | 1, 99.8 | Window percentiles |
| `window_values` | null | `[lo, hi]` absolute window (overrides the percentiles; `none` preprocessing) |
| `labels` | from metadata | Label maps: `{value: {name, color, opacity, visible, role}}` (see [](../labels.md)) |
| `label_smooth` | 1.0 | Label maps: smoothing of the shading field (voxels) |
| `smooth` | 0 | Gaussian pre-smoothing (voxels) before windowing; gives noisy data clean normals |
| `z_range` | null | `[Z0, Z1]` full-res slices to keep |
| `cylinder` | null | Container cut, in full-res px: `[CY, CX, R]` or, for a tilted tube, `[CY_TOP, CX_TOP, CY_BOTTOM, CX_BOTTOM, R]` |
| `keep_largest` | false | Keep only the largest connected object (drops debris) |
| `open_iter` | 6 | Opening iterations that cut thin bridges before `keep_largest` |
| `mask_level` | 20 | Material threshold on the windowed 0–255 volume; anchors the opacity curves and the surface |
| `down` | 1 | Extra downsampling of the cached volume when rendering |

## Look

See [](looks.md).

| Key | Default | Meaning |
|---|---|---|
| `look` | `bone` | A look name, a matplotlib colormap, or a JSON object |
| `curve` | from the look | `solid`, `soft`, `layered`, `glass`, `mip` |
| `opacity` | 0.6 | Maximum per-voxel opacity |
| `interior` | 1.0 | Opacity factor for uniform interiors vs surfaces (gradient opacity) |
| `lighting` | `studio` | A rig name or a JSON object |
| `scatter` | from the rig | Volumetric shadows, 0–2 |
| `gi_reach` | 0.4 | How far shadow rays travel, 0–1 |
| `anisotropy` | 0 | Scattering anisotropy, −1 to 1 |
| `specular` | from the look | Specular strength |
| `exposure` | auto | Brightness; 1.0 with filmic tone mapping, 1.35 without |
| `tone_mapping` | `filmic` | `filmic` (ACES) or `none` |
| `supersample` | 2 | Render at N× and average down |
| `sample_distance` | 0.5 | Ray step in voxels |
| `background` | `look` | `look`, `black`, `white`, `grey` |

## Camera and output

| Key | Default | Meaning |
|---|---|---|
| `size` | `1920x1080` | Image size `WxH` |
| `fps` | 30 | Video frame rate |
| `frames` | 360 | Turntable frames |
| `elev` | 20 | Turntable elevation (degrees) |
| `elevations` | null | Still elevations (default `elev`, 45, −20, 0) |
| `azimuths` | 8 | Stills per elevation |
| `azimuth` | 0 | Start azimuth of stills and turntable (degrees) |
| `zoom` | 1.0 | Camera zoom for stills and turntables (>1 is closer) |

## Text, tour, surface

| Key | Default | Meaning |
|---|---|---|
| `title`, `subtitle` | `""` | Overlays on the tour overview and the surface turntable |
| `credit` | `""` | Bottom-right line on every tour and surface frame |
| `regions` | `[]` | Tour close-ups: `[{"name": …, "z": …, "y": …, "x": …}]`, full-res |
| `style` | `volume` | Tour style: `volume` or `surface` |
| `tour.speed` | 1.0 | Below 1 gives a shorter video |
| `tour.scroll` | 150 | Native slices scrolled per close-up |
| `tour.zoom` | 1.8 | Zoom reached at the end of a close-up |
| `tour.region_size` | 800 | Full-res pixels shown in a close-up |
| `surface.ao` | 0.03 | Ambient-occlusion radius (fraction of the size; 0 turns it off) |
| `surface.shadows` | false | Hard shadow map |
| `surface.pbr` | true | Physically based material (false = Phong) |
| `surface.roughness` | 0.65 | PBR roughness |

## Example

```json
{
  "preset": ["interior", "presentation-4k"],
  "source": "/data/sample/rec.zarr",
  "volume": "volume_max768.npy",
  "factor": 16,
  "pixel_size_um": 1.88,
  "z_range": [1700, 13800],
  "cylinder": [6186, 5931, 6270, 6267, 5270],
  "look": "tissue",
  "lighting": "soft",
  "title": "Mouse brain",
  "subtitle": "synchrotron micro-CT, 1.88 µm voxels",
  "credit": "Imaging group, Example Institute",
  "regions": [{"name": "Cerebellum", "z": 3800, "y": 6144, "x": 4200}],
  "outputs": ["stills", "turntable", "tour"]
}
```
