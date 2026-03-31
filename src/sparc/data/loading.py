"""Data loading for SPARC - ZCAM and Pancam hyperspectral cubes."""

import numpy as np
import cv2
from pathlib import Path
from typing import TypedDict, Optional, Dict
from rapid.helpers import get_zcam_bandset
from marslab.imgops.imgutils import crop, eightbit
import asdf_settings.metadata
from asdf_settings import rapidlooks

from ..core.constants import SHARED_BANDS, BAD_PIXEL_FLAGS


class LoadResult(TypedDict):
    cube: np.ndarray
    left_cube: np.ndarray
    right_cube: np.ndarray
    left_cube_aligned: np.ndarray
    base_bands: Dict[str, np.ndarray]
    bandset: object
    homography_mask: np.ndarray
    homography_matrix: np.ndarray
    rgb_img: np.ndarray
    left_rgb_img: np.ndarray
    right_rgb_img: np.ndarray
    id: str


BAD_PIXEL_VALUES = tuple(
    i + 1 for i, flag in enumerate(asdf_settings.metadata.PIXEL_FLAG_NAMES)
    if flag in BAD_PIXEL_FLAGS
)

ZCAM_CROP = rapidlooks.CROP_SETTINGS["crop"]


def load_cube(iof_path: str,
              instrument: str,
              seq_id: Optional[str],
              obs_ix: int,
              do_apply_pixmaps: bool,
              ignore_bayers: bool,
              rgb_bands: Optional[tuple] = None) -> LoadResult:
    """Load and align a hyperspectral cube from ZCAM or Pancam."""
    if instrument == 'PCAM':
        return _load_pcam_cube(iof_path, seq_id, obs_ix, rgb_bands)
    return _load_zcam_cube(iof_path, seq_id, obs_ix, do_apply_pixmaps, ignore_bayers, rgb_bands)


# ---------------------------------------------------------------------------
# ZCAM
# ---------------------------------------------------------------------------

def _load_zcam_cube(iof_path, seq_id, obs_ix, do_apply_pixmaps, ignore_bayers, rgb_bands):
    search_path = Path(iof_path)
    bandset = get_zcam_bandset(search_path, seq_id=seq_id, observation_ix=obs_ix, load=False)
    scene_id = bandset.name

    filters = bandset.metadata["BAND"].sort_values()
    if ignore_bayers:
        filters = filters.loc[~filters.str.contains("0")].reset_index()

    bandset.load("all")
    bandset.bulk_debayer("all")
    base_bands = {b: crop(bandset.get_band(b), ZCAM_CROP).copy() for b in filters}

    if do_apply_pixmaps:
        pixmaps = {
            b: crop(bandset.pixmaps[b], ZCAM_CROP).copy()
            for b in sorted(bandset.metadata["FILTER"].unique())
        }
        bands = apply_pixel_masks(base_bands, pixmaps)
    else:
        bands = base_bands

    left_cube  = np.array([a for b, a in bands.items() if b.startswith("L")])
    right_cube = np.array([a for b, a in bands.items() if b.startswith("R")])

    left_raw  = np.array([a for b, a in base_bands.items() if b.startswith("L")])
    right_raw = np.array([a for b, a in base_bands.items() if b.startswith("R")])
    left_rgb_img  = create_rgb_stretch(left_raw)
    right_rgb_img = create_rgb_stretch(right_raw)

    homography_matrix = compute_homography(
        base_bands[SHARED_BANDS["L"]],
        base_bands[SHARED_BANDS["R"]]
    )
    left_cube_aligned = apply_homography(left_cube, homography_matrix, right_cube[0].shape)

    last_shared_index = sorted(bandset.raw).index(SHARED_BANDS['L'])
    homography_mask   = np.array(left_cube_aligned[last_shared_index] == 0)
    aligned_cube      = merge_left_right_cubes(left_cube_aligned, right_cube, last_shared_index)

    wl_lookup        = bandset.metadata.set_index("BAND")["WAVELENGTH"].to_dict()
    left_bands       = [b for b in bands if b.startswith("L")]
    right_bands      = [b for b in bands if b.startswith("R")]
    shared_wls       = [wl_lookup[b] for b in left_bands[:last_shared_index + 1]]
    unique_left_wls  = [wl_lookup[b] for b in left_bands[last_shared_index + 1:]]
    unique_right_wls = [wl_lookup[b] for b in right_bands[last_shared_index + 1:]]
    bandset._sparc_wavelengths = shared_wls + unique_left_wls + unique_right_wls

    return {
        'cube':             aligned_cube,
        'left_cube':        left_cube,
        'right_cube':       right_cube,
        'left_cube_aligned': left_cube_aligned,
        'base_bands':       base_bands,
        'bandset':          bandset,
        'homography_mask':  homography_mask,
        'homography_matrix': homography_matrix,
        'rgb_img':          right_rgb_img,
        'left_rgb_img':     left_rgb_img,
        'right_rgb_img':    right_rgb_img,
        'id':               scene_id,
        'instrument':       'ZCAM',
    }


# ---------------------------------------------------------------------------
# Pancam
# ---------------------------------------------------------------------------

def _pcam_rgb(r_b: np.ndarray, g_b: np.ndarray, b_b: np.ndarray) -> np.ndarray:
    """Per-channel percentile-stretched RGB from IOF bands."""
    channels = []
    for ch in (r_b, g_b, b_b):
        ch = np.nan_to_num(ch, nan=0.0)
        valid = ch[ch > 0]
        if valid.size > 0:
            lo, hi = np.percentile(valid, [1, 98])
            ch = np.clip((ch - lo) / (hi - lo) if hi > lo else ch, 0, 1)
        else:
            ch = np.zeros_like(ch)
        channels.append(ch)
    return (np.stack(channels, axis=-1) * 255).astype(np.uint8)


def _load_pcam_cube(iof_path, seq_id, obs_ix, rgb_bands):
    import pdr
    from ..utils.pancam_helpers import get_pcam_bandset

    bandset  = get_pcam_bandset(Path(iof_path), seq_id=seq_id, observation_ix=obs_ix, load=True)
    scene_id = bandset.name

    STEREO_PAIRS = [('L2', 'R2'), ('L7', 'R1')]
    stereo_left  = {l for l, r in STEREO_PAIRS}
    stereo_right = {r for l, r in STEREO_PAIRS}
    wl_lookup    = bandset.metadata.set_index("BAND")["WAVELENGTH"].to_dict()

    # Convert raw DNs → IOF using per-file PDS label scale and offset.
    # Values 0 (MISSING) and 4095 (INVALID) are masked per FLAG_VALUES_VAL.
    bands       = {}
    first_label = None
    for _, row in bandset.metadata.iterrows():
        band  = row['BAND']
        label = pdr.Data(row['PATH']).metadata
        if first_label is None:
            first_label = label
        scale  = label['DERIVED_IMAGE_PARMS']['RADIANCE_SCALING_FACTOR']
        offset = label['DERIVED_IMAGE_PARMS']['RADIANCE_OFFSET']
        dn     = bandset.get_band(band).copy().astype(np.float32)
        dn     = np.where((dn == 0) | (dn == 4095), np.nan, dn)
        bands[band] = dn * scale + offset

    bandset._sparc_label = first_label

    left_keys  = sorted(b for b in bands if b.startswith('L'))
    right_keys = sorted(b for b in bands if b.startswith('R'))

    left_cube  = np.array([bands[b] for b in left_keys])
    right_cube = np.array([bands[b] for b in right_keys])

    left_safe  = np.where(np.isfinite(left_cube),  left_cube,  0.0)
    right_safe = np.where(np.isfinite(right_cube), right_cube, 0.0)

    # Align left camera onto right frame using L7/R1 as the anchor pair.
    homography_matrix = compute_homography(
        np.where(np.isfinite(bands['L7']), bands['L7'], 0.0),
        np.where(np.isfinite(bands['R1']), bands['R1'], 0.0),
    )
    left_cube_aligned = apply_homography(left_safe, homography_matrix, right_cube[0].shape)
    homography_mask   = (left_cube_aligned[left_keys.index('L7')] == 0)
    aligned           = {b: left_cube_aligned[i] for i, b in enumerate(left_keys)}

    # Merge stereo pairs then append remaining singles.
    merged_bands = {}
    merged_wavelengths = []

    for l_band, r_band in STEREO_PAIRS:
        key = f"{l_band}+{r_band}"
        merged_bands[key] = np.nanmean(
            np.stack([aligned[l_band], bands[r_band]], axis=0), axis=0
        )
        merged_wavelengths.append((wl_lookup[l_band] + wl_lookup[r_band]) / 2)

    for b in (k for k in left_keys if k not in stereo_left):
        merged_bands[b] = aligned[b]
        merged_wavelengths.append(wl_lookup[b])

    for b in (k for k in right_keys if k not in stereo_right):
        merged_bands[b] = bands[b]
        merged_wavelengths.append(wl_lookup[b])

    bandset._sparc_wavelengths = merged_wavelengths
    cube = np.array(list(merged_bands.values()))

    left_rgb_img  = _pcam_rgb(bands['L2'], bands['L5'], bands['L6'])
    right_rgb_img = _pcam_rgb(bands['R2'], bands['R1'], bands['R1'])

    return {
        'cube':              cube,
        'left_cube':         left_safe,
        'right_cube':        right_safe,
        'left_cube_aligned': left_cube_aligned,
        'base_bands':        bands,
        'bandset':           bandset,
        'homography_mask':   homography_mask,
        'homography_matrix': homography_matrix,
        'rgb_img':           left_rgb_img,
        'left_rgb_img':      left_rgb_img,
        'right_rgb_img':     right_rgb_img,
        'id':                scene_id,
        'instrument':        'PCAM',
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def merge_left_right_cubes(left_cube: np.ndarray,
                           right_cube: np.ndarray,
                           last_shared_index: int) -> np.ndarray:
    """Average shared bands and concatenate unique left and right bands."""
    shared = [(left_cube[i] + right_cube[i]) / 2 for i in range(last_shared_index + 1)]
    unique_left  = [left_cube[i]  for i in range(last_shared_index + 1, left_cube.shape[0])]
    unique_right = [right_cube[i] for i in range(last_shared_index + 1, right_cube.shape[0])]
    return np.array(shared + unique_left + unique_right)


def apply_homography(source_cube: np.ndarray,
                     homography_matrix: np.ndarray,
                     target_shape: tuple) -> np.ndarray:
    """Warp every band in source_cube into the target frame."""
    return np.array([
        cv2.warpPerspective(source_cube[i], homography_matrix,
                            (target_shape[1], target_shape[0]))
        for i in range(source_cube.shape[0])
    ])


def compute_homography(source: np.ndarray,
                       destination: np.ndarray,
                       prestretch: int = 1) -> np.ndarray:
    """Compute a homography matrix via SIFT feature matching."""
    sift = cv2.SIFT_create()
    src_kp,  src_desc = sift.detectAndCompute(eightbit(source,      prestretch), None)
    dst_kp,  dst_desc = sift.detectAndCompute(eightbit(destination, prestretch), None)

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = sorted(matcher.match(src_desc, dst_desc), key=lambda x: x.distance)

    src_pts = np.float32([src_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([dst_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return homography


def create_bad_pixel_mask(pixmaps: Dict[str, np.ndarray], camera: str) -> np.ndarray:
    """Union of bad-pixel masks across all bands for a given camera."""
    masks = [np.isin(v, BAD_PIXEL_VALUES) for k, v in pixmaps.items() if k.startswith(camera)]
    return np.any(np.dstack(masks), axis=2)


def apply_pixel_masks(bands: Dict[str, np.ndarray],
                      pixmaps: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Replace bad pixels with NaN using per-camera pixmap masks."""
    left_mask  = create_bad_pixel_mask(pixmaps, "L")
    right_mask = create_bad_pixel_mask(pixmaps, "R")
    return {
        b: np.where(left_mask if b.startswith("L") else right_mask, np.nan, a)
        for b, a in bands.items()
    }


def create_rgb_stretch(cube: np.ndarray) -> np.ndarray:
    """Percentile-stretched RGB from a hyperspectral cube."""
    from marslab.imgops.imgutils import enhance_color
    rgb = np.stack([cube[2], cube[1], cube[0]], axis=-1)
    return enhance_color(np.ma.masked_invalid(rgb), bounds=(0, 1), stretch=0.1)