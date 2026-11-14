# Rendering from the viewer

**File › Render 3D…** (Ctrl+Shift+R, or **Render 3D…** in the toolbar)
configures and launches an offline render of the open dataset.

```{image} ../_static/img/ui/render_dialog.jpg
:class: shot
:alt: The Render 3D dialog
```

Preset
: A base preset and a quality modifier. Choosing them fills the controls
  below with the preset's values. Outputs: stills, turntable, tour, surface.

Look & light
: Look (editable: any matplotlib colormap name works too), lighting rig,
  opacity curve, background, filmic tone mapping, exposure (*auto* = 1.0
  with filmic) and supersampling.

Data
: Preprocessing (`none` or `tomo`), window percentiles, smoothing, keep the
  largest object, Z range (full-res slices), and the size of the volume read
  for rendering.

Output
: Image size, turntable frames, elevation, title, subtitle and credit (the
  credit is remembered).

Tour regions
: **Add crosshair position** adds the viewer's current `(z, y, x)` as a
  close-up. Rename regions in the table.

Advanced overrides
: A JSON object with any other [config key](config.md), e.g.
  `{"cylinder": [256, 256, 220], "tour": {"speed": 0.5}}`.

Renderer Python
: The interpreter that runs the render. The dialog looks for one with VTK,
  SciPy, scikit-image and imageio-ffmpeg: first the viewer's own Python, then
  conda environments (names containing *render*, *vtk* or *zview* first). You
  can also choose one by hand. It's remembered.

## Buttons

Render
: Writes `render.json` into the output folder and runs
  `python -m zview.render run render.json …` in the background. The log
  streams into the dialog. **Stop** kills it. When it finishes, the output
  folder opens.

Save config / Copy command
: Writes the config, or copies a ready-to-run command line to the clipboard,
  for running on another machine (for example a GPU node):

  ```console
  $ PYTHONPATH=/path/to/zview /path/to/env/bin/python -u -m zview.render run OUT/render.json stills turntable
  ```

Save as preset…
: Stores the current settings as a [user preset](presets.md#your-own-presets).

The written config holds only the presets plus the values you changed, so it
stays readable. The cached volume is reused between runs as long as the source
and the data settings stay the same.
