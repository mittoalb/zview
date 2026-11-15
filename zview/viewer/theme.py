"""Shared colours.

Each plane gets one colour, used for its pane header accent, for the
crosshair line that marks it in the other panes, and for its outline in 3D,
so a plane is recognisable everywhere it appears.
"""

PLANE_COLORS = {
    'XY': (1.00, 0.36, 0.36),
    'XZ': (0.36, 0.90, 0.40),
    'YZ': (0.40, 0.60, 1.00),
}

# Plane whose slice is fixed along each (z, y, x) axis.
AXIS_PLANE = ('XY', 'XZ', 'YZ')
