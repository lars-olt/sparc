"""
Pre-segment all scenes in a directory using SAM.

Scans the given folder and one level of subdirectories for ZCAM and Pancam scenes,
runs SAM segmentation on each, and saves the result as a compressed NPZ file
alongside the source images. Already-processed scenes are skipped.

Use --dcs to segment from a decorrelation-stretched image; files get a _dcs suffix.
Only the DCS flag at pre-segmentation time matters for matching - ROIStudio will use
a pre-segmented file only when the DCS toggle matches the suffix.

Usage:
    python presegment.py <folder> --sam-path <path> [--dcs]
    python presegment.py <folder> --sam-path <path> --points-per-side 64
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

# asdf must be in sys.modules before asdf_settings.rapidlooks is imported.
# loading.py imports asdf_settings at module level, so any sparc import that
# triggers the package init chain will fail unless asdf is already loaded.
import asdf  # noqa: F401 - primes sys.modules for asdf_settings


# Inlined from src.sparc.utils.pancam_helpers to avoid triggering sparc/__init__
# (and therefore the full loading.py → asdf_settings chain) during scanning.
_PCAM_FILENAME_RE = re.compile(
    r'^(?P<ROVER>\d)'
    r'P'
    r'(?P<SCLK>\d{9})'
    r'(?P<PRODUCT_TYPE>[A-Z]{3})'
    r'[A-Z0-9]{4}'
    r'(?P<SEQ_ID>[A-Z]\d{4})'
    r'(?P<FILTER>[LR]\d)'
    r'(?P<VERSION>[A-Z0-9])'
    r'.+$',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Scene discovery
# ---------------------------------------------------------------------------

def _is_pcam_folder(folder: Path) -> bool:
    """Return True if the folder contains parseable Pancam IOF files."""
    for f in folder.iterdir():
        if f.suffix.upper() in ('.IMG', '.IMQ') and _PCAM_FILENAME_RE.match(f.name):
            return True
    return False


def _pcam_scene_id(observation) -> str:
    """Build a PCAM scene ID from the first file's pdr label."""
    import pdr
    try:
        label = pdr.Data(observation.iloc[0]['PATH']).metadata
        sol   = int(label['PLANET_DAY_NUMBER'])
        seq   = str(label['SEQUENCE_ID']).strip()
        pma   = int(label['ROVER_MOTION_COUNTER'][3])
        return f"Sol{sol:04d}_{seq}_PMA{pma}"
    except Exception:
        return str(observation['SEQ_ID'].iloc[0])


def _find_pcam_scenes(folder: Path):
    """Yield (seq_id, obs_ix, scene_id) for each PCAM pointing in folder."""
    from src.sparc.utils.pancam_helpers import scan_pcam_files, split_pcam_observations
    try:
        products = scan_pcam_files(folder)
    except Exception:
        return
    obs_ix = 0
    for _, group in products.groupby('SEQ_ID'):
        for obs in split_pcam_observations(group):
            yield obs['SEQ_ID'].iloc[0], obs_ix, _pcam_scene_id(obs)
            obs_ix += 1


def _find_zcam_scenes(folder: Path):
    """Yield (seq_id=None, obs_ix, scene_id) for each ZCAM pointing in folder."""
    from src.sparc.data.loading import _scan_and_split, _bandset_from_group
    try:
        groups = _scan_and_split(folder)
    except Exception:
        return
    for obs_ix, group in enumerate(groups):
        try:
            bs = _bandset_from_group(group)
            if len(bs.metadata) >= 3 and bs.name:
                yield None, obs_ix, bs.name
        except Exception as e:
            print(f"  Warning: skipping obs {obs_ix} in {folder.name}: {e}")


def find_scenes(root: Path):
    """
    Find all scenes up to one level deep.

    Yields (folder, instrument, seq_id, obs_ix, scene_id).
    """
    candidates = [root] + sorted(d for d in root.iterdir() if d.is_dir())
    seen = set()

    for folder in candidates:
        img_files = [f for f in folder.iterdir()
                     if f.is_file() and f.suffix.upper() in ('.IMG', '.IMQ')]
        if not img_files:
            continue

        instrument = 'PCAM' if _is_pcam_folder(folder) else 'ZCAM'
        finder     = _find_pcam_scenes if instrument == 'PCAM' else _find_zcam_scenes

        for seq_id, obs_ix, scene_id in finder(folder):
            key = (str(folder), obs_ix)
            if key not in seen:
                seen.add(key)
                yield folder, instrument, seq_id, obs_ix, scene_id


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def _segment(rgb_img: np.ndarray, sam_path: str, params: dict) -> np.ndarray:
    """Run SAM on rgb_img and return a 2D int32 segment label array."""
    from src.sparc.segmentation.sam_segmentation import segment_image
    return segment_image(
        model_path          = sam_path,
        img                 = rgb_img,
        preserve_background = params['preserve_background'],
        points_per_side     = params['points_per_side'],
        pred_iou_thresh     = params['pred_iou_thresh'],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('folder',
        help='Root directory to scan for scenes (checked one level deep)')
    parser.add_argument('--sam-path', required=True,
        help='Path to SAM model checkpoint (.pth)')
    parser.add_argument('--dcs', action='store_true',
        help='Segment using the decorrelation stretch (files saved with _dcs suffix)')
    parser.add_argument('--points-per-side', type=int, default=32,
        help='SAM sampling density (default: 32)')
    parser.add_argument('--pred-iou-thresh', type=float, default=0.88,
        help='SAM IoU confidence threshold (default: 0.88)')
    parser.add_argument('--preserve-background', action='store_true',
        help='Assign unclassified pixels to segment 0 instead of the last segment')
    args = parser.parse_args()

    root = Path(args.folder)
    if not root.is_dir():
        print(f"Error: {root} is not a directory.", file=sys.stderr)
        sys.exit(1)

    suffix = '_dcs' if args.dcs else ''
    params = {
        'preserve_background': args.preserve_background,
        'points_per_side':     args.points_per_side,
        'pred_iou_thresh':     args.pred_iou_thresh,
    }

    print(f"Scanning {root} for scenes...")
    scenes = list(find_scenes(root))

    if not scenes:
        print("No scenes found.")
        return

    print(f"Found {len(scenes)} scene(s).")
    if args.dcs:
        print("DCS mode: segmenting decorrelation-stretched images (_dcs suffix).")
    print()

    from src.sparc.data.loading import load_cube, make_dcs_rgb

    for i, (folder, instrument, seq_id, obs_ix, scene_id) in enumerate(scenes, 1):
        npz_path = folder / f"{scene_id}{suffix}.npz"
        print(f"[{i}/{len(scenes)}] {scene_id}  ({instrument})")

        if npz_path.exists():
            print(f"  Skipping - {npz_path.name} already exists.\n")
            continue

        print(f"  Loading scene from {folder.name}...")
        try:
            load_result = load_cube(
                iof_path         = str(folder),
                instrument       = instrument,
                seq_id           = seq_id,
                obs_ix           = obs_ix,
                do_apply_pixmaps = True,
                ignore_bayers    = False,
            )
        except Exception as e:
            print(f"  Load failed: {e}\n")
            continue

        rgb_img = load_result['rgb_img']
        if args.dcs:
            rgb_img = make_dcs_rgb(load_result)

        H, W = rgb_img.shape[:2]
        print(f"  Segmenting {W}x{H} image...")
        try:
            segments = _segment(rgb_img, args.sam_path, params)
        except Exception as e:
            print(f"  Segmentation failed: {e}\n")
            continue

        n_segs = len(np.unique(segments)) - (1 if 0 in segments else 0)
        np.savez_compressed(str(npz_path), segments=segments, dcs=np.bool_(args.dcs))
        print(f"  {n_segs} segments → {npz_path.name}\n")

    print("Done.")


if __name__ == '__main__':
    main()