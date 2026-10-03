"""Optional RoMa adapter. Imported only when explicitly selected in a source run."""

from functools import lru_cache
from threading import Lock
import sys
import warnings

import cv2
import numpy as np

from .stereo import DenseStereoMapping
from ..core.config import AlignmentConfig
from ..utils.device import prepare_accelerators, resolve_device
from ..utils.memory import release_accelerator_memory

_INFERENCE_LOCK = Lock()
CERTAINTY_THRESHOLD = 0.5


@lru_cache(maxsize=1)
def _model(device):
    import torch
    try:
        from romatch import roma_outdoor
    except ImportError as exc:
        raise RuntimeError(
            'RoMa is not installed in this Python environment. Install SPARC requirements-roma.txt '
            'in the experimental environment, or select homography.'
        ) from exc
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', message='Local correlation is not supported on non-Linux platforms.*',
            category=UserWarning, module=r'romatch\.models\..*',
        )
        # Portable PyTorch correlation; no fused extension required, including Linux.
        # RoMa disables autocast outside CUDA; float32 avoids mixed dtypes on MPS.
        dtype = torch.float16 if device == 'cuda' else torch.float32
        return roma_outdoor(device=device, symmetric=True, use_custom_corr=False, amp_dtype=dtype)


def _offload_model(model):
    model.to('cpu')
    # RoMa 0.1.2 keeps DINO in a plain list, outside model.parameters().
    backbones = getattr(model.encoder, 'dinov2_vitl14', None)
    if isinstance(backbones, list):
        for backbone in backbones:
            backbone.to('cpu')


def _rgb(image):
    from PIL import Image
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if not finite.size:
        raise ValueError('RoMa input has no finite pixels.')
    scale = 255.0 if finite.max() <= 1 else 1.0
    pixels = np.nan_to_num(image * scale, nan=0.0, posinf=255.0, neginf=0.0)
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)).convert('RGB')


def _pixel_field(normalized, certainty, source, target, certainty_threshold=CERTAINTY_THRESHOLD):
    h, w = source.shape
    grid = cv2.resize(normalized, (w, h), interpolation=cv2.INTER_LINEAR)
    grid = ((grid + 1) * np.array([target.shape[1], target.shape[0]], dtype=np.float32) - 1) / 2
    confidence = cv2.resize(certainty, (w, h), interpolation=cv2.INTER_LINEAR)
    finite_target = cv2.remap(
        np.isfinite(target).astype(np.float32), grid[..., 0], grid[..., 1],
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    valid = ((confidence > certainty_threshold) & np.isfinite(source)
             & np.isfinite(grid).all(axis=-1) & (finite_target >= 1)
             & (grid[..., 0] >= 0) & (grid[..., 0] <= target.shape[1] - 1)
             & (grid[..., 1] >= 0) & (grid[..., 1] <= target.shape[0] - 1))
    return np.ascontiguousarray(grid, dtype=np.float32), valid


def compute_mapping(left, right, config=None):
    if getattr(sys, 'frozen', False):
        raise RuntimeError('Experimental RoMa is available only when running from source.')
    prepare_accelerators()
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('RoMa requires a separate PyTorch environment. See README.md.') from exc
    config = config or AlignmentConfig(method='roma')
    config.validate()
    device = resolve_device(config.device)
    with _INFERENCE_LOCK:
        model = _model(device.type)
        warp = certainty = None
        try:
            model.to(device)
            with torch.inference_mode():
                warp, certainty = model.match(_rgb(left), _rgb(right), device=device)
            warp = warp.detach().float().cpu().numpy()
            certainty = certainty.detach().float().cpu().numpy()
        finally:
            # Reuse weights on subsequent scenes without keeping SAM's GPU memory occupied.
            _offload_model(model)
            release_accelerator_memory()
    if warp.ndim == 4:
        warp, certainty = warp[0], certainty[0]
    if warp.ndim != 3 or warp.shape[-1] != 4 or warp.shape[1] % 2:
        raise ValueError(f'Unexpected symmetric RoMa output shape: {warp.shape}')
    mid = warp.shape[1] // 2
    lr, lv = _pixel_field(warp[:, :mid, 2:], certainty[:, :mid], left, right, config.certainty_threshold)
    rl, rv = _pixel_field(warp[:, mid:, :2], certainty[:, mid:], right, left, config.certainty_threshold)
    # Reject inconsistent matches and correspondences into low-confidence regions.
    for forward, backward, own_valid, other_valid in ((lr, rl, lv, rv.copy()), (rl, lr, rv, lv.copy())):
        back = cv2.remap(backward, forward[..., 0], forward[..., 1], cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=float('nan'))
        support = cv2.remap(other_valid.astype(np.float32), forward[..., 0], forward[..., 1],
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        yy, xx = np.indices(own_valid.shape)
        own_valid &= (support >= 1) & (np.hypot(back[..., 0] - xx, back[..., 1] - yy) <= config.cycle_tolerance)
    if not lv.any() or not rv.any():
        raise ValueError('RoMa found no confident, consistent stereo overlap. Try Homography.')
    return DenseStereoMapping(lr, rl, lv, rv)


def align_scene(scene, config=None):
    """Replace only aligned products; retain raw per-eye cubes for ROI spectra."""
    if getattr(sys, 'frozen', False):
        raise RuntimeError('Experimental RoMa is unavailable in packaged builds.')
    # ZCAM shared L1/R1 bands are the pair evaluated in the playground.
    # Pancam uses the existing RGB composites used for stereo display.
    if not scene.get('left_band_keys') or not scene.get('right_band_keys'):
        raise ValueError('RoMa requires a scene with both cameras. Use Homography for a single eye.')
    if scene.get('instrument') == 'PCAM':
        left = cv2.cvtColor(scene['left_rgb_img'], cv2.COLOR_RGB2GRAY)
        right = cv2.cvtColor(scene['right_rgb_img'], cv2.COLOR_RGB2GRAY)
    else:
        bands = scene['base_bands']
        if 'L1' not in bands or 'R1' not in bands:
            raise ValueError('RoMa requires the ZCAM L1 and R1 bands.')
        left, right = bands['L1'], bands['R1']
    config = config or AlignmentConfig(method='roma')
    mapping = compute_mapping(left, right, config)
    aligned = mapping.warp_left(scene['left_cube'])
    left_indices = {name: i for i, name in enumerate(scene['left_band_keys'])}
    right_indices = {name: i for i, name in enumerate(scene['right_band_keys'])}
    merged = []
    for source, _, lkey, rkey in scene['merged_band_recipe']:
        if source == 'stereo':
            band = (aligned[left_indices[lkey]] + scene['right_cube'][right_indices[rkey]]) / 2
        elif source == 'left_only':
            band = aligned[left_indices[lkey]]
        else:
            band = scene['right_cube'][right_indices[rkey]]
        merged.append(band)
    from dataclasses import asdict
    return {**scene, 'stereo_mapping': mapping, 'alignment_method': 'roma',
            'alignment_config': asdict(config),
            'homography_matrix': None, 'left_cube_aligned': aligned,
            'cube': np.array(merged), 'homography_mask': ~mapping.right_valid,
            'alignment_coverage': float(mapping.right_valid.mean())}
