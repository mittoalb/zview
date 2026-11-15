"""zview — multi-resolution Zarr/N5 viewer and offline 3D renderer.

Kept import-free on purpose: `zview.render` / `zview.tools.vessels` run headless in
environments without Qt, and the PyOpenGL fix-up (`sysinfo.apply_gl_compat`)
is applied by the GUI modules before they import pyqtgraph/VisPy.
"""
