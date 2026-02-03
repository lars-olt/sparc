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

from ..core.constants import SHARED_BANDS, BAD_PIXEL_FLAGS


class LoadResult(TypedDict):
    """Result from loading hyperspectral cube."""
    cube: np.ndarray
    left_cube: np.ndarray
    right_cube: np.ndarray
    left_cube_aligned: np.ndarray
    base_bands: Dict[str, np.ndarray]
    bandset: ZcamBandSet
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
              seq_id: Optional[str],
              obs_ix: int,
              do_apply_pixmaps: bool,
              ignore_bayers: bool) -> LoadResult:
    """
    Load and align hyperspectral data from left and right ZCAM cameras.
    
    Args:
        iof_path: Path to IOF data directory
        seq_id: Sequence ID (optional)
        obs_ix: Observation index
        do_apply_pixmaps: Apply pixel maps for bad pixel correction
        ignore_bayers: Ignore Bayer filter bands
        
    Returns:
        LoadResult with aligned cube and metadata
    """
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
    
    left_cube = np.array([a for b, a in bands.items() if b.startswith("L")])
    right_cube = np.array([a for b, a in bands.items() if b.startswith("R")])
    
    # Create RGB images for left and right separately
    left_rgb_img = create_rgb_stretch(left_cube)
    right_rgb_img = create_rgb_stretch(right_cube)
    
    homography_matrix = compute_homography(
        base_bands[SHARED_BANDS["L"]],
        base_bands[SHARED_BANDS["R"]]
    )
    left_cube_aligned = apply_homography(left_cube, homography_matrix, right_cube[0].shape)
    
    last_shared_index = sorted(bandset.raw).index(SHARED_BANDS['L'])
    homography_mask = np.array(left_cube_aligned[last_shared_index] == 0)
    
    aligned_cube = merge_left_right_cubes(left_cube_aligned, right_cube, last_shared_index)
    
    return {
        'cube': aligned_cube,
        'left_cube': left_cube,
        'right_cube': right_cube,
        'left_cube_aligned': left_cube_aligned,
        'base_bands': base_bands,
        'bandset': bandset,
        'homography_mask': homography_mask,
        'homography_matrix': homography_matrix,
        'rgb_img': right_rgb_img,  # Use right as default
        'left_rgb_img': left_rgb_img,
        'right_rgb_img': right_rgb_img,
        'id': scene_id
    }


def merge_left_right_cubes(left_cube: np.ndarray,
                           right_cube: np.ndarray,
                           last_shared_index: int) -> np.ndarray:
    """
    Merge left and right camera cubes, averaging shared bands.
    
    Args:
        left_cube: Aligned left camera cube
        right_cube: Right camera cube
        last_shared_index: Index of last band shared between cameras
        
    Returns:
        Merged hyperspectral cube
    """
    cube = []
    
    # Average shared bands
    for band in range(last_shared_index + 1):
        band_avg = (left_cube[band] + right_cube[band]) / 2
        cube.append(band_avg)
    
    # Add unique left bands
    for band in range(last_shared_index + 1, left_cube.shape[0]):
        cube.append(left_cube[band])
    
    # Add unique right bands
    for band in range(last_shared_index + 1, right_cube.shape[0]):
        cube.append(right_cube[band])
    
    return np.array(cube)


def apply_homography(source_cube: np.ndarray,
                    homography_matrix: np.ndarray,
                    target_shape: tuple) -> np.ndarray:
    """
    Apply homography transformation to align left camera with right.
    
    Note: This approach is not robust to parallax.
    
    Args:
        source_cube: Cube to transform
        homography_matrix: Homography transformation matrix
        target_shape: Target image shape (height, width)
        
    Returns:
        Transformed cube
    """
    transformed = []
    for band in range(source_cube.shape[0]):
        warped = cv2.warpPerspective(
            source_cube[band],
            homography_matrix,
            (target_shape[1], target_shape[0])
        )
        transformed.append(warped)
    return np.array(transformed)


def compute_homography(source: np.ndarray,
                      destination: np.ndarray,
                      prestretch: int = 1) -> np.ndarray:
    """
    Compute homography matrix using SIFT feature matching.
    
    Args:
        source: Source 2D image
        destination: Destination 2D image
        prestretch: Pre-stretch factor for 8-bit conversion
        
    Returns:
        3x3 homography matrix
    """
    source_8bit = eightbit(source, prestretch)
    dest_8bit = eightbit(destination, prestretch)
    
    sift = cv2.SIFT_create()
    src_keypoints, src_descriptors = sift.detectAndCompute(source_8bit, None)
    dst_keypoints, dst_descriptors = sift.detectAndCompute(dest_8bit, None)
    
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = sorted(matcher.match(src_descriptors, dst_descriptors), key=lambda x: x.distance)
    
    src_points = np.float32([src_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_points = np.float32([dst_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    homography, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    return homography


def create_bad_pixel_mask(pixmaps: Dict[str, np.ndarray], camera: str) -> np.ndarray:
    """
    Create mask of bad pixels for specified camera.
    
    Args:
        pixmaps: Dictionary of pixel maps
        camera: Camera identifier ('L' or 'R')
        
    Returns:
        Boolean mask of bad pixels
    """
    camera_pixmaps = {k: v for k, v in pixmaps.items() if k.startswith(camera)}
    bad_pixel_masks = [np.isin(v, BAD_PIXEL_VALUES) for v in camera_pixmaps.values()]
    return np.any(np.dstack(bad_pixel_masks), axis=2)


def apply_pixel_masks(bands: Dict[str, np.ndarray],
                     pixmaps: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Apply pixel maps to mask bad pixels with NaN.
    
    Args:
        bands: Dictionary of band data
        pixmaps: Dictionary of pixel maps
        
    Returns:
        Bands with bad pixels masked as NaN
    """
    left_mask = create_bad_pixel_mask(pixmaps, "L")
    right_mask = create_bad_pixel_mask(pixmaps, "R")
    
    masked_bands = {}
    for band_name, band_data in bands.items():
        mask = left_mask if band_name.startswith("L") else right_mask
        masked_bands[band_name] = np.where(mask, np.nan, band_data)
    
    return masked_bands


def create_rgb_stretch(cube: np.ndarray) -> np.ndarray:
    """
    Create RGB stretched image from hyperspectral cube.
    
    Args:
        cube: Hyperspectral data cube
        
    Returns:
        RGB stretched image for visualization
    """
    from marslab.imgops.imgutils import enhance_color
    
    rgb_dict = {'R': cube[2], 'G': cube[1], 'B': cube[0]}
    rgb_stack = np.stack([rgb_dict['R'], rgb_dict['G'], rgb_dict['B']], axis=-1)
    rgb_masked = np.ma.masked_invalid(rgb_stack)
    return enhance_color(rgb_masked, bounds=(0, 1), stretch=0.1)