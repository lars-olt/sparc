"""Global constants for SPARC."""

from typing import Any, Dict

import numpy as np
from marslab.compat.mertools import MERSPECT_M20_COLOR_MAPPINGS

# Mastcam-Z band wavelengths (nm), in cube band order
WAVELENGTHS = [480, 544, 630, 800, 754, 677, 605, 528, 442, 866, 910, 939, 978, 1022]

BAYER_CUTOFF_INDEX = WAVELENGTHS.index(800)
LEFT_CUTOFF_INDEX  = WAVELENGTHS.index(866)

RGB_BANDS    = ['B', 'G', 'R']
PLOT_MARKERS = ["o", "s", "^", "v", "*", "D", "H"]

COLOR_MAPPINGS = MERSPECT_M20_COLOR_MAPPINGS
COLOR_NAMES    = list(COLOR_MAPPINGS.keys())
COLORS         = list(COLOR_MAPPINGS.values())

SHARED_BANDS = {"L": "L1", "R": "R1"}
BAD_PIXEL_FLAGS = ('bad', 'no_signal', 'hot')

DEFAULT_ROI_AREA_THRESHOLD    = 50
DEFAULT_EDGE_OFFSET           = 10
DEFAULT_ALLOWED_VARIANCE      = 1.0
DEFAULT_ALBEDO_RATIO_THRESHOLD = 0.80

_EDGE_MASK_CACHE: dict = {}

# Shared display-stretch parameters for enhance_color.
RGB_ENHANCE_KWARGS = {'bounds': (0, 1), 'stretch': 0.1}


def get_instrument_config(instrument: str) -> Dict[str, Any]:
    """Return instrument-specific configuration derived from marslab's DERIVED_CAM_DICT."""
    from marslab.compat.xcam import DERIVED_CAM_DICT

    if instrument not in DERIVED_CAM_DICT:
        raise ValueError(
            f"Unsupported instrument: {instrument}. "
            f"Available: {list(DERIVED_CAM_DICT.keys())}"
        )

    filters      = DERIVED_CAM_DICT[instrument]['filters']
    sorted_bands = sorted(filters, key=filters.__getitem__)
    wavelengths  = [filters[b] for b in sorted_bands]

    if instrument == 'ZCAM':
        rgb_bands    = ('L0B', 'L0G', 'L0R')
        bayer_cutoff = next(i for i, w in enumerate(wavelengths) if w >= 800)
        left_cutoff  = next(i for i, w in enumerate(wavelengths) if w >= 866)
        has_stereo   = True
    elif instrument == 'PCAM':
        rgb_bands    = ('L3', 'L5', 'L7')
        bayer_cutoff = 0
        left_cutoff  = len(wavelengths) // 2
        has_stereo   = True
    else:
        rgb_bands    = ('R', 'G', 'B')
        bayer_cutoff = 0
        left_cutoff  = len(wavelengths)
        has_stereo   = False

    return {
        'instrument':  instrument,
        'wavelengths': wavelengths,
        'filters':     filters,
        'rgb_bands':   rgb_bands,
        'bayer_cutoff': bayer_cutoff,
        'left_cutoff': left_cutoff,
        'has_stereo':  has_stereo,
    }