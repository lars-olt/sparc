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

from ..core.constants import SHARED_BANDS, BAD_PIXEL_FLAGS, RGB_ENHANCE_KWARGS


_RATIO_THRESH       = 0.75  # Lowe ratio test
_RANSAC_THRESH      = 2.5   # reprojection threshold in pixels
_MIN_AFFINE_INLIERS = 20    # fall back to homography below this count

# Fixed RANSAC settings so stereo alignment is reproducible run to run and across
# machines - a drifting homography shifts every ROI derived from it.
_HOMOGRAPHY_SEED    = 42
_RANSAC_MAX_ITERS   = 5000
_RANSAC_CONFIDENCE  = 0.999

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


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
    stretch_bands:      Dict[str, bool]  # per-camera stretch availability


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

    # Drop focus/context subframes - keep only the largest frame size.
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


def _rgb_from_keys(keys, bands, shape, stretch_fn):
    """Build an RGB display image from preferred band keys.

    If all three preferred bands are present, uses stretch_fn. If not,
    falls back to the three lowest-index available bands for that camera.
    Returns (image, bands_available) where bands_available signals whether
    the preferred set was used - False means stretch overlays should be disabled.
    """
    prefix   = keys[0][0]  # 'L' or 'R'
    present  = [k for k in keys if k in bands]
    fallback = sorted(b for b in bands if b.startswith(prefix))

    if len(present) == 3:
        return stretch_fn(bands[keys[0]], bands[keys[1]], bands[keys[2]]), True

    candidates = fallback[:3]
    if len(candidates) == 3:
        return stretch_fn(bands[candidates[0]], bands[candidates[1]], bands[candidates[2]]), False

    return np.zeros((*shape, 3), dtype=np.uint8), False


# ---------------------------------------------------------------------------
# ZCAM
# ---------------------------------------------------------------------------

def _load_zcam_cube(
    iof_path, seq_id, obs_ix, do_apply_pixmaps, ignore_bayers, rgb_bands
):
    groups = _scan_and_split(iof_path, seq_id)
    if obs_ix >= len(groups):
        raise ValueError(
            f"obs_ix={obs_ix} out of range - found {len(groups)} pointing(s) in {iof_path}"
        )

    bandset = _bandset_from_group(groups[obs_ix])

    try:
        sol      = int(bandset.metadata['SOL'].iloc[0])
        seq      = str(bandset.metadata['SEQ_ID'].iloc[0]).strip()
        rsm      = int(bandset.metadata['RSM'].min())
        scene_id = f"Sol{sol:04d}_{seq}_RSM{rsm}"
    except Exception:
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

    if not left_band_keys and not right_band_keys:
        raise ValueError(f"No usable bands found in {iof_path}")
    shape = next(iter(bands.values())).shape

    left_cube  = np.array([bands[b] for b in left_band_keys])
    right_cube = np.array([bands[b] for b in right_band_keys])

    left_raw  = np.array([base_bands[b] for b in left_band_keys])
    right_raw = np.array([base_bands[b] for b in right_band_keys])

    left_rgb_img,  left_stretch  = _rgb_from_keys(('L2', 'L5', 'L7'), bands, shape, _pcam_rgb)
    right_rgb_img, right_stretch = _rgb_from_keys(('R2', 'R1', 'R1'), bands, shape, _pcam_rgb)

    # homography requires both shared bands - fall back to identity if either is missing
    l_shared = SHARED_BANDS["L"]
    r_shared = SHARED_BANDS["R"]
    if l_shared in base_bands and r_shared in base_bands:
        homography_matrix = compute_homography(base_bands[l_shared], base_bands[r_shared])
    else:
        homography_matrix = np.eye(3, dtype=np.float64)

    left_cube_aligned = apply_homography(left_cube, homography_matrix, shape) if left_cube.size else left_cube

    if l_shared in left_band_keys:
        last_shared_index = left_band_keys.index(l_shared)
        homography_mask   = np.array(left_cube_aligned[last_shared_index] == 0)
    else:
        last_shared_index = -1
        homography_mask   = np.zeros(shape, dtype=bool)

    if last_shared_index >= 0:
        aligned_cube = merge_left_right_cubes(left_cube_aligned, right_cube, last_shared_index)
    elif left_cube_aligned.size and right_cube.size:
        aligned_cube = np.concatenate([left_cube_aligned, right_cube], axis=0)
    else:
        aligned_cube = left_cube_aligned if left_cube_aligned.size else right_cube

    wl_lookup        = bandset.metadata.set_index("BAND")["WAVELENGTH"].to_dict()
    shared_keys      = left_band_keys[:last_shared_index + 1] if last_shared_index >= 0 else []
    unique_left_keys = left_band_keys[last_shared_index + 1:]
    unique_right_keys = right_band_keys[last_shared_index + 1:] if last_shared_index >= 0 else right_band_keys

    bandset._sparc_wavelengths = (
        [wl_lookup[b] for b in shared_keys]
        + [wl_lookup[b] for b in unique_left_keys]
        + [wl_lookup[b] for b in unique_right_keys]
    )
    merged_band_recipe = (
        [('stereo',    b, b,    right_band_keys[i]) for i, b in enumerate(shared_keys)]
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
        "stretch_bands":      {"left": left_stretch, "right": right_stretch},
    }


def _zcam_rgb(r_b, g_b, b_b):
    """enhance_color stretch of three ZCAM bands to uint8 RGB."""
    cube   = np.array([r_b, g_b, b_b])
    result = create_rgb_stretch(cube)
    return result


# ---------------------------------------------------------------------------
# Pancam
# ---------------------------------------------------------------------------

# Default Pancam display bands
_PCAM_LEFT_RGB  = ('L4', 'L5', 'L6')
_PCAM_RIGHT_RGB = ('R7', 'R5', 'R3')


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


def dcs_rgb(r_b, g_b, b_b):
    """Decorrelation stretch of three bands to uint8 RGB.

    Applies eigenspace rotation to remove inter-band correlation, giving
    enhanced spectral contrast for SAM segmentation and SIFT feature matching.
    Works with any three float band arrays (ZCAM or Pancam).
    """
    H, W    = r_b.shape
    invalid = ~np.isfinite(r_b) | ~np.isfinite(g_b) | ~np.isfinite(b_b)
    r = np.where(invalid, 0.0, np.nan_to_num(r_b)).astype(np.float32)
    g = np.where(invalid, 0.0, np.nan_to_num(g_b)).astype(np.float32)
    b = np.where(invalid, 0.0, np.nan_to_num(b_b)).astype(np.float32)

    vecs  = np.stack([r, g, b], axis=-1).reshape(-1, 3)
    valid = vecs[~invalid.ravel()]
    if valid.shape[0] < 4:
        return np.zeros((H, W, 3), dtype=np.uint8)

    cov        = np.cov(valid.T).astype(np.float32)
    eigvals, V = np.linalg.eig(cov)
    T          = (V @ np.diag(1.0 / np.sqrt(np.abs(eigvals))) @ V.T).astype(np.float32)
    means      = valid.mean(axis=0)
    dcs        = ((vecs - means) @ T + means + (means - means @ T)).reshape(H, W, 3)

    result   = np.zeros((H, W, 3), dtype=np.float32)
    valid_2d = ~invalid
    for c in range(3):
        ch     = dcs[:, :, c]
        v      = ch[valid_2d]
        if v.size == 0:
            continue
        lo, hi = np.percentile(v, [0.5, 99.5])
        result[:, :, c] = np.clip((ch - lo) / (hi - lo) if hi > lo else ch, 0.0, 1.0)
    result[invalid] = 0.0
    return (result * 255).astype(np.uint8)


def make_dcs_rgb(load_result: dict) -> np.ndarray:
    """Compute a DCS-stretched RGB from base_bands for use as segmentation input.

    ZCAM uses the right-camera DCS preset bands (R6/R3/R1).
    PCAM uses the right-camera visible bands (R7/R5/R3).
    Falls back to the existing rgb_img if any band is missing.
    """
    instrument = load_result.get('instrument', 'ZCAM')
    bands      = load_result['base_bands']

    keys = _PCAM_RIGHT_RGB if instrument == 'PCAM' else ('R6', 'R3', 'R1')

    r, g, b = (bands.get(k) for k in keys)
    if any(x is None for x in (r, g, b)):
        return load_result['rgb_img']

    return dcs_rgb(r, g, b)


def _is_pds4(label) -> bool:
    """Return True if this is a PDS4 pdr label (single top-level Product_Observational key)."""
    try:
        keys = list(label.keys())
        return keys == ['Product_Observational']
    except Exception:
        return False


def _pds4_metaget(label, *path):
    """Walk a chain of keys into a pdr MultiDict, returning None on any miss."""
    node = label['Product_Observational']
    for key in path:
        try:
            node = node[key]
        except (KeyError, TypeError):
            return None
    return node


def _normalise_pcam_label(label) -> dict:
    """Return a flat dict with the fields downstream code needs, regardless of PDS3/PDS4 format.

    Fields produced:
      PLANET_DAY_NUMBER, SEQUENCE_ID, ROVER_MOTION_COUNTER (list),
      SOLAR_ELEVATION, DERIVED_IMAGE_PARMS (dict with RADIANCE_SCALING_FACTOR / RADIANCE_OFFSET).

    Any field that can't be extracted is omitted - callers should use .get().
    """
    if not _is_pds4(label):
        # PDS3 - pdr.Metadata behaves like a dict; just mirror the fields we need.
        out = {}
        for key in ('PLANET_DAY_NUMBER', 'SEQUENCE_ID', 'SEQUENCE_VERSION_ID', 'SOLAR_ELEVATION'):
            try:
                out[key] = label[key]
            except (KeyError, TypeError):
                pass
        try:
            rmc = label['ROVER_MOTION_COUNTER']
            out['ROVER_MOTION_COUNTER'] = list(rmc) if hasattr(rmc, '__iter__') else rmc
        except (KeyError, TypeError):
            pass
        try:
            out['DERIVED_IMAGE_PARMS'] = label['DERIVED_IMAGE_PARMS']
        except (KeyError, TypeError):
            pass
        return out

    # PDS4 - navigate the MultiDict hierarchy.
    out = {}

    try:
        mer = _pds4_metaget(label, 'Observation_Area', 'Mission_Area', 'mer:MER_Parameters')
        out['PLANET_DAY_NUMBER'] = int(mer['mer:sol_number'])
    except Exception:
        pass

    try:
        cmd = _pds4_metaget(label, 'Observation_Area', 'Discipline_Area',
                            'msn_surface:Surface_Mission_Information',
                            'msn_surface:Command_Execution')
        out['SEQUENCE_ID'] = str(cmd['msn_surface:sequence_id']).strip()
        # separate try so a missing version doesn't take the sequence id down with it
        try:
            out['SEQUENCE_VERSION_ID'] = str(cmd['msn_surface:sequence_version_id']).strip()
        except Exception:
            pass
    except Exception:
        pass

    try:
        lander  = _pds4_metaget(label, 'Observation_Area', 'Discipline_Area',
                                'geom:Geometry', 'geom:Geometry_Lander')
        mc      = lander['geom:Motion_Counter']
        indices = mc.getall('geom:Motion_Counter_Index') if hasattr(mc, 'getall') else [mc]
        rmc     = [0, 0, 0, 0, 0]
        order   = ['Site', 'Drive', 'IDD', 'PMA', 'HGA']
        for entry in indices:
            name = entry.get('geom:index_name', '')
            if name in order:
                rmc[order.index(name)] = int(float(entry.get('geom:index_value_number', 0)))
        out['ROVER_MOTION_COUNTER'] = rmc
    except Exception:
        pass

    try:
        lander  = _pds4_metaget(label, 'Observation_Area', 'Discipline_Area',
                                'geom:Geometry', 'geom:Geometry_Lander')
        derived = lander['geom:Derived_Geometry']
        out['SOLAR_ELEVATION'] = float(derived['geom:solar_elevation'])
    except Exception:
        pass

    try:
        elem = _pds4_metaget(label, 'File_Area_Observational', 'Array_2D_Image', 'Element_Array')
        out['DERIVED_IMAGE_PARMS'] = {
            'RADIANCE_SCALING_FACTOR': float(elem['scaling_factor']),
            'RADIANCE_OFFSET':         float(elem['value_offset']),
        }
    except Exception:
        pass

    return out


def pcam_seq_token(norm: dict) -> str:
    """Fold sequence version onto the sequence id, e.g. 'p2530' + '1' -> 'p2530v1'.

    seq_ver only means anything paired with its sequence, so it rides along in the
    same token that identifies a pointing. Drops the suffix if the version is absent
    so PDS3 files with no version still get a usable id.
    """
    seq = str(norm.get('SEQUENCE_ID', '')).strip()
    ver = str(norm.get('SEQUENCE_VERSION_ID', '')).strip()
    return f"{seq}v{ver}" if ver else seq


def _pcam_calibration(label) -> tuple:
    """Return (scale, offset) from a raw pdr label, handling PDS3 and PDS4."""
    norm = _normalise_pcam_label(label)
    parms = norm.get('DERIVED_IMAGE_PARMS', {})
    scale  = float(parms.get('RADIANCE_SCALING_FACTOR', 1.0))
    offset = float(parms.get('RADIANCE_OFFSET',         0.0))
    return scale, offset




def _load_pcam_cube(iof_path, seq_id, obs_ix, rgb_bands):
    import pdr
    from ..utils.pancam_helpers import get_pcam_bandset

    # load=False - we read every band ourselves via pdr below, so marslab never
    # touches the image data directly. This avoids breakage on files whose pdr
    # key is 'Image_Object' rather than 'IMAGE'.
    bandset = get_pcam_bandset(Path(iof_path), seq_id=seq_id, observation_ix=obs_ix, load=False)

    STEREO_PAIRS  = [("L2", "R2"), ("L7", "R1")]
    stereo_left   = {l for l, r in STEREO_PAIRS}
    stereo_right  = {r for l, r in STEREO_PAIRS}
    wl_lookup     = bandset.metadata.set_index("BAND")["WAVELENGTH"].to_dict()

    bands       = {}
    first_label = None
    for _, row in bandset.metadata.iterrows():
        band  = row["BAND"]
        fpath = row["PATH"]
        data  = pdr.Data(fpath)
        label = data.metadata
        if first_label is None:
            first_label = _normalise_pcam_label(label)

        scale, offset = _pcam_calibration(label)
        img_key  = 'IMAGE' if 'IMAGE' in data.keys() else 'Image_Object'
        dn       = np.array(data[img_key]).astype(np.float32)
        dn      = np.where((dn == 0) | (dn == 4095), np.nan, dn)
        bands[band] = dn * scale + offset

    bandset._sparc_label = first_label

    if not bands:
        raise ValueError(f"No usable bands loaded from {iof_path}")

    # Keep only bands at the largest frame size - smaller ones are thumbnails or subframes.
    all_shapes  = [bands[b].shape for b in bands]
    modal_shape = max(set(all_shapes), key=all_shapes.count)
    bands       = {b: a for b, a in bands.items() if a.shape == modal_shape}

    shape = modal_shape

    left_band_keys  = sorted(b for b in bands if b.startswith("L"))
    right_band_keys = sorted(b for b in bands if b.startswith("R"))

    left_cube  = np.array([bands[b] for b in left_band_keys])  if left_band_keys  else np.empty((0, *shape), dtype=np.float32)
    right_cube = np.array([bands[b] for b in right_band_keys]) if right_band_keys else np.empty((0, *shape), dtype=np.float32)
    left_safe  = np.where(np.isfinite(left_cube),  left_cube,  0.0)
    right_safe = np.where(np.isfinite(right_cube), right_cube, 0.0)

    left_rgb_img,  left_stretch  = _rgb_from_keys(_PCAM_LEFT_RGB, bands, shape, _pcam_rgb)
    right_rgb_img, right_stretch = _rgb_from_keys(_PCAM_RIGHT_RGB, bands, shape, _pcam_rgb)

    def _gray_for_homography(keys):
        available = [bands[k] for k in keys if k in bands]
        r, g, b   = (available + [np.zeros(shape, np.float32)] * 3)[:3]
        return cv2.cvtColor(dcs_rgb(r, g, b), cv2.COLOR_RGB2GRAY).astype(np.float32)

    left_gray  = _gray_for_homography(_PCAM_LEFT_RGB)
    right_gray = _gray_for_homography(('R3', 'R5', 'R7'))

    homography_matrix = compute_homography(left_gray, right_gray)
    left_cube_aligned = apply_homography(left_safe, homography_matrix, shape) if left_safe.size else left_safe

    mask_band_ix    = next((i for i, b in enumerate(left_band_keys) if b == "L7"), 0 if left_band_keys else None)
    homography_mask = (left_cube_aligned[mask_band_ix] == 0) if mask_band_ix is not None else np.zeros(shape, dtype=bool)

    aligned = {b: left_cube_aligned[i] for i, b in enumerate(left_band_keys)}

    merged_arrays      = []
    merged_wavelengths = []
    merged_band_recipe = []

    for l_band, r_band in STEREO_PAIRS:
        l_present = l_band in aligned
        r_present = r_band in bands
        if l_present and r_present:
            merged_arrays.append(np.nanmean(np.stack([aligned[l_band], bands[r_band]]), axis=0))
            merged_wavelengths.append((wl_lookup[l_band] + wl_lookup[r_band]) / 2)
            merged_band_recipe.append(('stereo', f"{l_band}+{r_band}", l_band, r_band))
        elif l_present:
            merged_arrays.append(aligned[l_band])
            merged_wavelengths.append(wl_lookup[l_band])
            merged_band_recipe.append(('left_only', l_band, l_band, None))
        elif r_present:
            merged_arrays.append(bands[r_band])
            merged_wavelengths.append(wl_lookup[r_band])
            merged_band_recipe.append(('right_only', r_band, None, r_band))
        # both missing - skip this wavelength entirely

    for b in (k for k in left_band_keys if k not in stereo_left):
        merged_arrays.append(aligned[b])
        merged_wavelengths.append(wl_lookup[b])
        merged_band_recipe.append(('left_only', b, b, None))

    for b in (k for k in right_band_keys if k not in stereo_right):
        merged_arrays.append(bands[b])
        merged_wavelengths.append(wl_lookup[b])
        merged_band_recipe.append(('right_only', b, None, b))

    bandset._sparc_wavelengths = merged_wavelengths

    try:
        sol      = int(first_label['PLANET_DAY_NUMBER'])
        seq      = pcam_seq_token(first_label)
        pma      = int(first_label['ROVER_MOTION_COUNTER'][3])  # PMA is index 3: (SITE, DRIVE, IDD, PMA, HGA)
        scene_id = f"Sol{sol:04d}_{seq}_PMA{pma}"
    except Exception:
        scene_id = bandset.metadata['SEQ_ID'].iloc[0] if 'SEQ_ID' in bandset.metadata.columns else "PCAM_scene"

    return {
        "cube":               np.array(merged_arrays) if merged_arrays else np.empty((0, *shape), dtype=np.float32),
        "left_cube":          left_safe,
        "right_cube":         right_safe,
        "left_cube_aligned":  left_cube_aligned,
        "base_bands":         bands,
        "bandset":            bandset,
        "homography_mask":    homography_mask,
        "homography_matrix":  homography_matrix,
        "rgb_img":            right_rgb_img,
        "left_rgb_img":       left_rgb_img,
        "right_rgb_img":      right_rgb_img,
        "id":                 scene_id,
        "instrument":         "PCAM",
        "left_band_keys":     left_band_keys,
        "right_band_keys":    right_band_keys,
        "merged_band_recipe": merged_band_recipe,
        "stretch_bands":      {"left": left_stretch, "right": right_stretch},
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def merge_left_right_cubes(left_cube, right_cube, last_shared_index):
    """Average shared bands and concatenate unique left and right bands."""
    shared        = [(left_cube[i] + right_cube[i]) / 2
                     for i in range(last_shared_index + 1)]
    unique_left   = [left_cube[i]  for i in range(last_shared_index + 1, left_cube.shape[0])]
    unique_right  = [right_cube[i] for i in range(last_shared_index + 1, right_cube.shape[0])]
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


def _prepare(img):
    """Convert to uint8 grayscale and apply CLAHE to normalize cross-camera tone."""
    gray = eightbit(img)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    return _CLAHE.apply(gray)


def compute_homography(source, destination, prestretch=1):
    """Compute a (3, 3) warp matrix from source to destination via SIFT feature matching.

    Both images are converted to CLAHE-normalized grayscale before descriptor
    extraction, removing tonal differences between cameras that would otherwise
    corrupt matches. Tries an affine model first (more constrained, better for
    wide-baseline stereo), falling back to a full homography if the affine inlier
    count is too low. Always returns a (3, 3) matrix so cv2.invert and
    warpPerspective work unchanged.
    """
    sift = cv2.SIFT_create()
    src_kp, src_desc = sift.detectAndCompute(_prepare(source),      None)
    dst_kp, dst_desc = sift.detectAndCompute(_prepare(destination), None)

    if src_desc is None or dst_desc is None:
        return np.eye(3, dtype=np.float64)

    # kNN with k=2 enables the ratio test, which is more robust than crossCheck
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn     = matcher.knnMatch(src_desc, dst_desc, k=2)
    good    = [m for m, n in knn if m.distance < _RATIO_THRESH * n.distance]

    if len(good) < 4:
        return np.eye(3, dtype=np.float64)

    src_pts = np.float32([src_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([dst_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Sort matches by (source x, y) so the point ordering fed to RANSAC is
    # independent of match-iteration order, which can vary across OpenCV builds.
    order   = np.lexsort((src_pts[:, 0, 1], src_pts[:, 0, 0]))
    src_pts = src_pts[order]
    dst_pts = dst_pts[order]

    # affine first - 6 DOF is less likely to overfit noise on a wide-baseline pair.
    # Seed OpenCV's RNG and pin the iteration count so RANSAC is reproducible -
    # otherwise the same scene warps slightly differently run to run and machine
    # to machine, shifting every homography-derived ROI.
    cv2.setRNGSeed(_HOMOGRAPHY_SEED)
    affine, inliers = cv2.estimateAffine2D(
        src_pts, dst_pts,
        method                = cv2.RANSAC,
        ransacReprojThreshold = _RANSAC_THRESH,
        maxIters              = _RANSAC_MAX_ITERS,
        confidence            = _RANSAC_CONFIDENCE,
    )

    n_inliers = int(inliers.sum()) if inliers is not None else 0

    if affine is not None and n_inliers >= _MIN_AFFINE_INLIERS:
        # pad to (3, 3) so cv2.invert and warpPerspective work everywhere downstream
        return np.vstack([affine, [0.0, 0.0, 1.0]])

    # fall back to full homography when affine doesn't have enough support
    cv2.setRNGSeed(_HOMOGRAPHY_SEED)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, _RANSAC_THRESH,
                              maxIters=_RANSAC_MAX_ITERS, confidence=_RANSAC_CONFIDENCE)
    return H if H is not None else np.eye(3, dtype=np.float64)


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
    rgb    = np.stack([cube[0], cube[1], cube[2]], axis=-1)
    result = enhance_color(np.ma.masked_invalid(rgb), **RGB_ENHANCE_KWARGS)
    return np.ascontiguousarray(np.ma.filled(result, 0) * 255, dtype=np.uint8)