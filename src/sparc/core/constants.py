"""Global constants for SPARC package."""

import numpy as np

# Configure threading before any sklearn imports to avoid KMeans warnings
import os
import platform

if platform.system() == "Windows":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

from marslab.compat.mertools import MERSPECT_M20_COLOR_MAPPINGS

# Wavelength per band
WLS = [480, 544, 630, 800, 754, 677, 605, 528, 442, 866, 910, 939, 978, 1022]

# Cutoff indices
BAYER_CUTOFF = WLS.index(800)
L_CUTOFF = WLS.index(866)

# RGB mapping
RGB_MAPPING = ["B", "G", "R"]

# Color mappings for visualization
COLOR_MAPPINGS = MERSPECT_M20_COLOR_MAPPINGS
COLOR_NAMES = list(COLOR_MAPPINGS.keys())
COLORS = list(COLOR_MAPPINGS.values())

# Plot markers (currently randomly assigned to rois)
# TODO: auto-identify roi target type and associate marker
MARKERS = ["o", "s", "^", "v", "*", "D", "H"]

# ZCAM settings
SHARED_BANDS = {"L": "L1", "R": "R1"}
BAD_FLAGS = ("bad", "no_signal", "hot")

# Default thresholds and parameters
DEFAULT_ROI_AREA_THRESHOLD = 50
DEFAULT_EDGE_OFFSET = 10
DEFAULT_ALLOWED_VARIANCE = 1
DEFAULT_ALBEDO_RATIO_THRESHOLD = 0.80

# Cache for edge masks
_EDGE_MASK_CACHE = {}
