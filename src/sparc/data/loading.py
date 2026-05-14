"""Data loading for SPARC - ZCAM and Pancam hyperspectral cubes."""

import numpy as np
import cv2
import pandas as pd
from functools import reduce
from operator import mul
from pathlib import Path
from typing import TypedDict, Optional, Dict
from marslab.imgops.imgutils import crop, eightbit
import asdf_settings.metadata
from asdf_settings import rapidlooks

from ..core.constants import SHARED_BANDS, BAD_PIXEL_FLAGS


class LoadResult(TypedDict):
    cube:               np.ndarray
    left_cube:          np.ndarray
    right_cube:         np.ndarray
    left_cube_aligned:  np.ndarray
    base_bands:         Dict[str, np.ndarray]
    bandset:            object
    homography_mask:    np.ndarray
    homography_matrix:  np.ndarray
    rgb_img:            np.ndarray
    left_rgb_img:       np.ndarray
    right_rgb_img:      np.ndarray
    id:                 str
    instrument:         str
    left_band_keys:     list
    right_band_keys:    list
    merged_band_recipe: list


BAD_PIXEL_VALUES = tuple(
    i + 1
    for i, flag in enumerate(asdf_settings.metadata.PIXEL_FLAG_NAMES)
    if flag in BAD_PIXEL_FLAGS
)

ZCAM_CROP = rapidlooks.CROP_SETTINGS["crop"]


def _fwd(path) -> str:
    """Normalize and forward-slash a path string.

    os.path.normpath resolves virtual filesystem paths (e.g. Dropbox on macOS)
    that QFileDialog returns but low-level file scanners can't traverse directly.
    The replace converts backslashes for asdf's Windows stem-parsing.
    """
    import os
    return os.path.normpath(os.path.expanduser(str(path))).replace('\\', '/')


def _scan_and_split(iof_path, seq_id=None):
    """
    Scan a folder and split files into per-pointing groups using RSM.

    Each pointing consists of a left/right camera pair with consecutive RSM
    values (e.g. 460/462). Sorting unique RSMs and chunking into pairs of two
    correctly groups all pointings regardless of scene count.

    Returns a list of DataFrames, one per pointing, each containing all filters
    for that pointing.
    """
    from asdf.scan import scan_zcam_files

    all_obs = scan_zcam_files(_fwd(iof_path))
    if seq_id:
        all_obs = all_obs[
            all_obs['SEQ_ID'].str.lower().str.contains(str(seq_id).lower())
        ]

    # Drop focus/context subframes — keep only the largest frame size.
    frame_areas = all_obs['SUBFRAME'].map(lambda s: reduce(mul, s[2:]))
    all_obs     = all_obs[frame_areas == frame_areas.max()]

    rsm_vals = sorted(all_obs['RSM'].unique())
    groups   = []
    for i in range(0, len(rsm_vals), 2):
        pair  = rsm_vals[i:i + 2]
        group = all_obs[all_obs['RSM'].isin(pair)].copy().reset_index(drop=True)
        if len(group) >= 3:
            groups.append(group)

    return groups


def _bandset_from_group(group):
    """Build and format a ZcamBandSet from a pre-filtered metadata DataFrame."""
    from asdf.scan import rate_cal_offset
    from asdf.zcam_bandset import ZcamBandSet

    # Deduplicate: one file per filter, best cal_offset score wins.
    keep_rows = []
    for _filt, fgroup in group.groupby('FILTER'):
        if len(fgroup) == 1:
            keep_rows.append(fgroup.iloc[0])
        else:
            scores = rate_cal_offset(fgroup)
            keep_rows.append(fgroup.loc[scores[scores].index[0]])

    deduped = pd.DataFrame(keep_rows).reset_index(drop=True)
    bs = ZcamBandSet(deduped)
    bs.format_metadata()
    return bs


def load_cube(
    iof_path: str,
    instrument: str,
    seq_id: Optional[str],
    obs_ix: int,
    do_apply_pixmaps: bool,
    ignore_bayers: bool,
    rgb_bands: Optional[tuple] = None,
) -> LoadResult:
    """Load and align a hyperspectral cube from ZCAM or Pancam."""
    if instrument == "PCAM":
        return _load_pcam_cube(iof_path, seq_id, obs_ix, rgb_bands)
    return _load_zcam_cube(
        iof_path, seq_id, obs_ix, do_apply_pixmaps, ignore_bayers, rgb_bands
    )


# ---------------------------------------------------------------------------
# ZCAM
# ---------------------------------------------------------------------------

def _load_zcam_cube(
    iof_path, seq_id, obs_ix, do_apply_pixmaps, ignore_bayers, rgb_bands
):
    groups = _scan_and_split(iof_path, seq_id)
    if obs_ix >= len(groups):
        raise ValueError(
            f"obs_ix={obs_ix} out of range — found {len(groups)} pointing(s) in {iof_path}"
        )

    bandset = _bandset_from_group(groups[obs_ix])

    scene_id = bandset.name
    filters  = bandset.metadata["BAND"].sort_values()
    if ignore_bayers:
        filters = filters.loc[~filters.str.contains("0")].reset_index()

    bandset.load("all")
    bandset.bulk_debayer("all")
    base_bands = {b: crop(bandset.get_band(b), ZCAM_CROP).copy() for b in filters}

    if do_apply_pixmaps:
        pixmaps = {}
        for b in sorted(bandset.metadata["FILTER"].unique()):
            try:
                pixmaps[b] = crop(bandset.pixmaps[b], ZCAM_CROP).copy()
            except (KeyError, Exception):
                pass
        bands = apply_pixel_masks(base_bands, pixmaps)
    else:
        bands = base_bands

    left_band_keys  = [b for b in bands if b.startswith("L")]
    right_band_keys = [b for b in bands if b.startswith("R")]

    left_cube  = np.array([bands[b] for b in left_band_keys])
    right_cube = np.array([bands[b] for b in right_band_keys])

    left_raw      = np.array([base_bands[b] for b in left_band_keys])
    right_raw     = np.array([base_bands[b] for b in right_band_keys])
    left_rgb_img  = create_rgb_stretch(left_raw)
    right_rgb_img = create_rgb_stretch(right_raw)

    homography_matrix = compute_homography(
        base_bands[SHARED_BANDS["L"]], base_bands[SHARED_BANDS["R"]]
    )
    left_cube_aligned = apply_homography(
        left_cube, homography_matrix, right_cube[0].shape
    )

    last_shared_index = left_band_keys.index(SHARED_BANDS["L"])
    homography_mask   = np.array(left_cube_aligned[last_shared_index] == 0)
    aligned_cube      = merge_left_right_cubes(
        left_cube_aligned, right_cube, last_shared_index
    )

    wl_lookup         = bandset.metadata.set_index("BAND")["WAVELENGTH"].to_dict()
    shared_keys       = left_band_keys[:last_shared_index + 1]
    unique_left_keys  = left_band_keys[last_shared_index + 1:]
    unique_right_keys = right_band_keys[last_shared_index + 1:]

    bandset._sparc_wavelengths = (
        [wl_lookup[b] for b in shared_keys]
        + [wl_lookup[b] for b in unique_left_keys]
        + [wl_lookup[b] for b in unique_right_keys]
    )
    merged_band_recipe = (
        [('stereo',     b, b,    right_band_keys[i]) for i, b in enumerate(shared_keys)]
        + [('left_only',  b, b,    None) for b in unique_left_keys]
        + [('right_only', b, None, b)    for b in unique_right_keys]
    )

    return {
        "cube":               aligned_cube,
        "left_cube":          left_cube,
        "right_cube":         right_cube,
        "left_cube_aligned":  left_cube_aligned,
        "base_bands":         base_bands,
        "bandset":            bandset,
        "homography_mask":    homography_mask,
        "homography_matrix":  homography_matrix,
        "rgb_img":            right_rgb_img,
        "left_rgb_img":       left_rgb_img,
        "right_rgb_img":      right_rgb_img,
        "id":                 scene_id,
        "instrument":         "ZCAM",
        "left_band_keys":     left_band_keys,
        "right_band_keys":    right_band_keys,
        "merged_band_recipe": merged_band_recipe,
    }


# ---------------------------------------------------------------------------
# Pancam
# ---------------------------------------------------------------------------

def _pcam_rgb(r_b, g_b, b_b):
    """Per-channel percentile-stretched RGB from IOF bands."""
    channels = []
    for ch in (r_b, g_b, b_b):
        ch    = np.nan_to_num(ch, nan=0.0)
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

    STEREO_PAIRS  = [("L2", "R2"), ("L7", "R1")]
    stereo_left   = {l for l, r in STEREO_PAIRS}
    stereo_right  = {r for l, r in STEREO_PAIRS}
    wl_lookup     = bandset.metadata.set_index("BAND")["WAVELENGTH"].to_dict()

    bands       = {}
    first_label = None
    for _, row in bandset.metadata.iterrows():
        band  = row["BAND"]
        label = pdr.Data(row["PATH"]).metadata
        if first_label is None:
            first_label = label
        scale  = label["DERIVED_IMAGE_PARMS"]["RADIANCE_SCALING_FACTOR"]
        offset = label["DERIVED_IMAGE_PARMS"]["RADIANCE_OFFSET"]
        dn = bandset.get_band(band).copy().astype(np.float32)
        dn = np.where((dn == 0) | (dn == 4095), np.nan, dn)
        bands[band] = dn * scale + offset

    bandset._sparc_label = first_label

    left_band_keys  = sorted(b for b in bands if b.startswith("L"))
    right_band_keys = sorted(b for b in bands if b.startswith("R"))

    left_cube  = np.array([bands[b] for b in left_band_keys])
    right_cube = np.array([bands[b] for b in right_band_keys])
    left_safe  = np.where(np.isfinite(left_cube),  left_cube,  0.0)
    right_safe = np.where(np.isfinite(right_cube), right_cube, 0.0)

    homography_matrix = compute_homography(
        np.where(np.isfinite(bands["L7"]), bands["L7"], 0.0),
        np.where(np.isfinite(bands["R1"]), bands["R1"], 0.0),
    )
    left_cube_aligned = apply_homography(left_safe, homography_matrix, right_cube[0].shape)
    homography_mask   = left_cube_aligned[left_band_keys.index("L7")] == 0
    aligned           = {b: left_cube_aligned[i] for i, b in enumerate(left_band_keys)}

    merged_arrays      = []
    merged_wavelengths = []
    merged_band_recipe = []

    for l_band, r_band in STEREO_PAIRS:
        merged_arrays.append(
            np.nanmean(np.stack([aligned[l_band], bands[r_band]]), axis=0)
        )
        merged_wavelengths.append((wl_lookup[l_band] + wl_lookup[r_band]) / 2)
        merged_band_recipe.append(('stereo', f"{l_band}+{r_band}", l_band, r_band))

    for b in (k for k in left_band_keys  if k not in stereo_left):
        merged_arrays.append(aligned[b])
        merged_wavelengths.append(wl_lookup[b])
        merged_band_recipe.append(('left_only', b, b, None))

    for b in (k for k in right_band_keys if k not in stereo_right):
        merged_arrays.append(bands[b])
        merged_wavelengths.append(wl_lookup[b])
        merged_band_recipe.append(('right_only', b, None, b))

    bandset._sparc_wavelengths = merged_wavelengths

    left_rgb_img  = _pcam_rgb(bands["L2"], bands["L5"], bands["L6"])
    right_rgb_img = _pcam_rgb(bands["R2"], bands["R1"], bands["R1"])

    return {
        "cube":               np.array(merged_arrays),
        "left_cube":          left_safe,
        "right_cube":         right_safe,
        "left_cube_aligned":  left_cube_aligned,
        "base_bands":         bands,
        "bandset":            bandset,
        "homography_mask":    homography_mask,
        "homography_matrix":  homography_matrix,
        "rgb_img":            left_rgb_img,
        "left_rgb_img":       left_rgb_img,
        "right_rgb_img":      right_rgb_img,
        "id":                 scene_id,
        "instrument":         "PCAM",
        "left_band_keys":     left_band_keys,
        "right_band_keys":    right_band_keys,
        "merged_band_recipe": merged_band_recipe,
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def merge_left_right_cubes(left_cube, right_cube, last_shared_index):
    """Average shared bands and concatenate unique left and right bands."""
    shared       = [(left_cube[i] + right_cube[i]) / 2
                    for i in range(last_shared_index + 1)]
    unique_left  = [left_cube[i]
                    for i in range(last_shared_index + 1, left_cube.shape[0])]
    unique_right = [right_cube[i]
                    for i in range(last_shared_index + 1, right_cube.shape[0])]
    return np.array(shared + unique_left + unique_right)


def apply_homography(source_cube, homography_matrix, target_shape):
    """Warp every band in source_cube into the target frame."""
    return np.array([
        cv2.warpPerspective(
            source_cube[i], homography_matrix,
            (target_shape[1], target_shape[0])
        )
        for i in range(source_cube.shape[0])
    ])


def compute_homography(source, destination, prestretch=1):
    """Compute a homography matrix via SIFT feature matching."""
    sift = cv2.SIFT_create()
    src_kp, src_desc = sift.detectAndCompute(eightbit(source,      prestretch), None)
    dst_kp, dst_desc = sift.detectAndCompute(eightbit(destination, prestretch), None)

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = sorted(matcher.match(src_desc, dst_desc), key=lambda x: x.distance)

    src_pts = np.float32([src_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([dst_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return homography


def create_bad_pixel_mask(pixmaps, camera, shape=None):
    """Union of bad-pixel masks across all bands for a given camera."""
    masks = [np.isin(v, BAD_PIXEL_VALUES) for k, v in pixmaps.items()
             if k.startswith(camera)]
    if not masks:
        return np.zeros(shape or (1, 1), dtype=bool)
    return np.any(np.dstack(masks), axis=2)


def apply_pixel_masks(bands, pixmaps):
    """Replace bad pixels with NaN using per-camera pixmap masks."""
    shape      = next(iter(bands.values())).shape
    left_mask  = create_bad_pixel_mask(pixmaps, "L", shape)
    right_mask = create_bad_pixel_mask(pixmaps, "R", shape)
    return {
        b: np.where(left_mask if b.startswith("L") else right_mask, np.nan, a)
        for b, a in bands.items()
    }


def create_rgb_stretch(cube):
    from marslab.imgops.imgutils import enhance_color
    rgb    = np.stack([cube[2], cube[1], cube[0]], axis=-1)
    result = enhance_color(np.ma.masked_invalid(rgb), bounds=(0, 1), stretch=0.1)
    return np.ascontiguousarray(np.ma.filled(result, 0) * 255, dtype=np.uint8)