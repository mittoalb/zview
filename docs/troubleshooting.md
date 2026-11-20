# Troubleshooting

| Symptom | Fix |
|---|---|
| 3D pane or renders are **black** in a conda environment | PyOpenGL loaded a different `libOpenGL` from Qt's. zview binds PyOpenGL to the versioned libraries on startup; if you import zview modules from your own code, import `zview.viewer.app` (or call `zview.core.sysinfo.apply_gl_compat()`) before pyqtgraph or VisPy. `ZVIEW_NO_GL_COMPAT=1` disables the fix-up, which helps tell whether it's involved |
| VTK: `bad X server connection`, segfault | GPU renders need an X display: run in a desktop or VNC session (`echo $DISPLAY` must print something). On headless servers use `zview-render cpu` |
| A label map shows as grey intensities (or an image as classes) | Toggle **View › Treat as label map** |
| Label maps render flat (no shading) | VTK older than 9.4: `pip install -U vtk` |
| Segfault in VTK 9.4 when rendering off-screen | Its EGL window crashes; zview only uses EGL on VTK ≥ 9.7. Force GLX with `ZVIEW_VTK_OFFSCREEN=default` |
| Render dialog: *No Python with VTK found* | `pip install 'zview[render]'` into some environment, or point **Renderer Python** at one |
| A slow first 3D load or prepare | The dataset has no deep pyramid, so every chunk is decoded (the status line shows how much). Add pyramid levels (`zview-convert`) or lower the quality / `max_axis` |
| HDF5: slow first XZ/YZ view, or a few seconds on open | The file stores each slice as one (compressed) chunk, so every slice in view is decoded once. Later views are cached; for repeated work convert the file to OME-Zarr (`zview-convert`) |
| Rotating the 3D view is slow or jerky; it freezes after each pause | Turn **Shadows** off in the 3D pane's toolbar (volumetric shadows cost a second or more per frame on dense data) |
| HDF5: 3D pane shows a coarser level than expected | The automatic level reads at most 2 GB from an HDF5 file. Click a finer level in the *Pyramid* panel, or raise `ZVIEW_3D_MAX_READ_GB` (e.g. 10 on fast local disks) |
| HDF5: the wrong dataset opens | Name it: `zview scan.h5:/path/to/dataset` |
| Pyramid levels visibly jump when zooming | The per-level `translation` metadata uses a non-standard convention: set `ZVIEW_NO_TRANSLATION=1` |
| XZ/YZ drift between levels | Non-integer downsample ratios between levels in the data; re-author with exact 2× steps |
| Container or air still visible in renders | Set `cylinder` / `z_range` (read them off `side_views_grid.png`), or raise `lo_pct` |
| Surfaces look noisy or torn | Raise `smooth` (e.g. 1.0–1.5), or move the window so `mask_level` doesn't sit on a material's own intensity |
| A translucent phase renders opaque | Use the `layered` or `glass` curve, or lower `opacity`; opacity accumulates over the ray's length |
| Conversion `Killed` | Lower `--slab` |
| Log empty for a long time | Run Python with `-u` / `PYTHONUNBUFFERED=1` |
| Vessels: false hits in gaps or debris | Mask debris in the input, raise `--high`, increase `--erode` |
| Vessels too fragmented | Lower `--low` (e.g. 0.30) |
