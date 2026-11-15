"""Sphinx configuration for the zview documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "zview"
author = "zview contributors"
copyright = "2026, zview contributors"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
myst_enable_extensions = ["colon_fence", "deflist", "attrs_inline", "attrs_block"]
myst_heading_anchors = 3
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build", "gen_images.py", "gen_screenshots.py"]

# Heavy / GUI dependencies are mocked so the API pages build without them.
autodoc_mock_imports = [
    "PyQt5", "pyqtgraph", "vispy", "OpenGL", "vtk", "scipy", "skimage", "imageio", "imageio_ffmpeg",
    "h5py", "tifffile", "zarr", "z5py", "napari", "dask", "matplotlib", "PIL", "numcodecs",
]
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": False}
autodoc_member_order = "bysource"

html_theme = "furo"
html_title = "zview"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "sidebar_hide_name": False,
    "light_css_variables": {"color-brand-primary": "#1f6fbf", "color-brand-content": "#1f6fbf"},
    "dark_css_variables": {"color-brand-primary": "#5aa7f0", "color-brand-content": "#5aa7f0"},
}
copybutton_prompt_text = r"\$ "
copybutton_prompt_is_regexp = True
