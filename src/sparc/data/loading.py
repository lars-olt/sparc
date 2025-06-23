"""Data loading functionality for SPARC."""

import numpy as np
import cv2
from pathlib import Path
from typing import TypedDict, Optional, Dict
from rapid.helpers import get_zcam_bandset
from marslab.imgops.imgutils import crop, eightbit
from asdf.zcam_bandset import ZcamBandSet
import asdf_settings.metadata
from asdf_settings import rapidlooks

from ..core.constants import SHARED_BANDS, BAD_FLAGS
from ..utils.geometry import get_rgb_stretch


# -----------------------------------------------------------------------------
# Huge thanks to Michael St. Clair (@m-stclair) for loading functionality!
# -----------------------------------------------------------------------------


class LoadResult(TypedDict):
    """Type definition for cube loading result."""

    cube: np.ndarray
    base_bands: Dict[str, np.ndarray]
    bandset: ZcamBandSet
    homography_tmask: np.ndarray
    rgb_img: np.ndarray
    id: str


# Constants for pixel mapping
BAD_PIXMAP_VALUES = tuple(
    i + 1
    for i, f in enumerate(asdf_settings.metadata.PIXEL_FLAG_NAMES)
    if f in BAD_FLAGS
)
ZCAM_CROP = rapidlooks.CROP_SETTINGS["crop"]


def load_cube(
    iof_path: str,
    seq_id: Optional[str],
    obs_ix: int,
    do_apply_pixmaps: bool,
    ignore_bayers: bool,
) -> LoadResult:
    """
    Load hyperspectral data cube from IOF files.

    Args:
        iof_path: Path to IOF data directory
        seq_id: Sequence ID (optional)
        obs_ix: Observation index
        do_apply_pixmaps: Whether to apply pixel maps for bad pixel correction
        ignore_bayers: Whether to ignore Bayer filter bands

    Returns:
        LoadResult containing loaded data and metadata
    """
    cube = []

    # Load left and right data cubes
    search_path = Path(iof_path)

    bs = get_zcam_bandset(search_path, seq_id=seq_id, observation_ix=obs_ix, load=False)

    # Get unique id of observation
    scene_id = bs.name

    filts = bs.metadata["BAND"].sort_values()
    if ignore_bayers:
        filts = filts.loc[~filts.str.contains("0")].reset_index()

    bs.load("all")
    bs.bulk_debayer("all")
    base_bands = {b: crop(bs.get_band(b), ZCAM_CROP).copy() for b in filts}

    if do_apply_pixmaps:
        pixmaps = {
            b: crop(bs.pixmaps[b], ZCAM_CROP).copy()
            for b in sorted(bs.metadata["FILTER"].unique())
        }
        # Apply pixel maps to mask bad pixels
        bands = apply_pixmaps(base_bands, pixmaps)
    else:
        bands = base_bands

    l_cube = np.array([a for b, a in bands.items() if b.startswith("L")])
    r_cube = np.array([a for b, a in bands.items() if b.startswith("R")])

    # Store RGB image of scene (used for segmentation)
    rgb_img = get_rgb_stretch(r_cube)

    # Compute homography using original bands (NaNs make SIFT unhappy)
    h_matrix = compute_homography(
        base_bands[SHARED_BANDS["L"]], base_bands[SHARED_BANDS["R"]]
    )
    l_cube_mapped = apply_homography(l_cube, h_matrix, r_cube[0].shape)

    # Get index of last shared band between left/right cameras
    last_shared_band_index = sorted(bs.raw).index(SHARED_BANDS["L"])

    # Create mask for homography transformation areas
    homography_tmask = np.array(l_cube_mapped[last_shared_band_index] == 0)

    # Average bands shared between left/right cameras (Bayer + 800nm)
    for band in range(last_shared_band_index + 1):
        band_avg = (l_cube_mapped[band] + r_cube[band]) / 2
        cube.append(band_avg)

    # Store left bands
    l_num_bands = l_cube.shape[0]
    for band in range(last_shared_band_index + 1, l_num_bands):
        cube.append(l_cube_mapped[band])

    # Store right bands
    r_num_bands = r_cube.shape[0]
    for band in range(last_shared_band_index + 1, r_num_bands):
        cube.append(r_cube[band])

    return {
        "cube": np.array(cube),
        "base_bands": base_bands,
        "bandset": bs,
        "homography_tmask": homography_tmask,
        "rgb_img": rgb_img,
        "id": scene_id,
    }


def apply_homography(
    src_cube: np.ndarray, hmat: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """
    Apply homography transformation to align left camera with right camera.

    Note: This approach is not robust to parallax.
    TODO: really need to fix this... eventually.

    Args:
        src_cube: Source cube to transform
        hmat: Homography matrix
        shape: Target shape

    Returns:
        Transformed cube
    """
    cube_transformed = []
    for band in range(src_cube.shape[0]):
        spec_slice = src_cube[band]
        warped_img = cv2.warpPerspective(spec_slice, hmat, (shape[1], shape[0]))
        cube_transformed.append(warped_img)
    return np.array(cube_transformed)


def compute_homography(
    src: np.ndarray, dst: np.ndarray, prestretch: int = 1
) -> np.ndarray:
    """
    Compute homography matrix that maps src to dst using SIFT features.

    Args:
        src: Source 2D array
        dst: Destination 2D array
        prestretch: Pre-stretch factor for 8-bit conversion

    Returns:
        Homography matrix
    """
    src, dst = map(lambda a: eightbit(a, prestretch), (src, dst))

    # Detect features and compute descriptors
    sift = cv2.SIFT_create()
    src_keypoints, src_descriptors = sift.detectAndCompute(src, None)
    dst_keypoints, dst_descriptors = sift.detectAndCompute(dst, None)

    # Match features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(src_descriptors, dst_descriptors)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    src_pts = np.float32([src_keypoints[m.queryIdx].pt for m in matches]).reshape(
        -1, 1, 2
    )
    dst_pts = np.float32([dst_keypoints[m.trainIdx].pt for m in matches]).reshape(
        -1, 1, 2
    )

    # Compute homography matrix
    return cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]


def make_eye_mask(pixmaps: Dict[str, np.ndarray], eye: str) -> np.ndarray:
    """
    Create bad pixel mask for specified camera eye.

    Args:
        pixmaps: Dictionary of pixel maps
        eye: Camera eye ('L' or 'R')

    Returns:
        Boolean mask of bad pixels
    """
    pixmaps = {k: v for k, v in pixmaps.items() if k.startswith(eye)}
    pixmaps = [np.isin(v, BAD_PIXMAP_VALUES) for v in pixmaps.values()]
    return np.any(np.dstack(pixmaps), axis=2)


def apply_pixmaps(
    bands: Dict[str, np.ndarray], pixmaps: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Apply pixel maps to mask bad pixels with NaN values.

    Args:
        bands: Dictionary of band data
        pixmaps: Dictionary of pixel maps

    Returns:
        Dictionary of bands with bad pixels masked as NaN
    """
    l_pix_mask = make_eye_mask(pixmaps, "L")
    r_pix_mask = make_eye_mask(pixmaps, "R")

    outbands = {}
    for k, v in bands.items():
        mask = l_pix_mask if k.startswith("L") else r_pix_mask
        outbands[k] = np.where(mask, np.nan, bands[k])

    return outbands
