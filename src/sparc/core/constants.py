"""Global constants for SPARC package."""

import numpy as np
from marslab.compat.mertools import MERSPECT_M20_COLOR_MAPPINGS

# Wavelengths per band (nm)
WAVELENGTHS = [480, 544, 630, 800, 754, 677, 605, 528, 442, 866, 910, 939, 978, 1022]

# Cutoff indices for camera separation
BAYER_CUTOFF_INDEX = WAVELENGTHS.index(800)
LEFT_CUTOFF_INDEX = WAVELENGTHS.index(866)

# RGB band mapping
RGB_BANDS = ['B', 'G', 'R']

# Visualization colors
COLOR_MAPPINGS = MERSPECT_M20_COLOR_MAPPINGS
COLOR_NAMES = list(COLOR_MAPPINGS.keys())
COLORS = list(COLOR_MAPPINGS.values())

# Plot markers
PLOT_MARKERS = ["o", "s", "^", "v", "*", "D", "H"]

# ZCAM camera settings
SHARED_BANDS = {"L": "L1", "R": "R1"}
BAD_PIXEL_FLAGS = ('bad', 'no_signal', 'hot')

# Default processing parameters
DEFAULT_ROI_AREA_THRESHOLD = 50
DEFAULT_EDGE_OFFSET = 10
DEFAULT_ALLOWED_VARIANCE = 1.0
DEFAULT_ALBEDO_RATIO_THRESHOLD = 0.80

# Edge mask cache
_EDGE_MASK_CACHE = {}