"""Pre-segment ZCAM and Pancam scenes with SAM and save compressed label arrays."""

import argparse
import queue
import re
import sys
import threading
import warnings
from pathlib import Path

import numpy as np

# Suppress known harmless warnings from DCS computation and marslab stretching.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="marslab")

# asdf must be in sys.modules before asdf_settings.rapidlooks is imported.
# loading.py imports asdf_settings at module level, so the sparc imports below
# would fail unless asdf is already loaded.
import asdf
import pdr

from src.sparc.data.loading import (_scan_and_split, _bandset_from_group,
                                    _normalise_pcam_label, load_cube,
                                    make_dcs_rgb, pcam_seq_token)
from src.sparc.segmentation.sam_segmentation import segment_image
from src.sparc.utils.pancam_helpers import scan_pcam_files, split_pcam_observations


# Local copy of the Pancam filename pattern - _is_pcam_folder only needs a
# cheap match, not the full parse that pancam_helpers does.
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

_SENTINEL = None  # signals the GPU worker that all loaders are done


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
    """Build a PCAM scene ID (Sol{sol}_seqVver_PMA{pma}) from the first file's pdr label."""
    try:
        label = pdr.Data(observation.iloc[0]['PATH']).metadata
        norm  = _normalise_pcam_label(label)
        sol   = int(norm['PLANET_DAY_NUMBER'])
        seq   = pcam_seq_token(norm)
        pma   = int(norm['ROVER_MOTION_COUNTER'][3])
        return f"Sol{sol:04d}_{seq}_PMA{pma}"
    except Exception:
        return str(observation['SEQ_ID'].iloc[0])


def _zcam_scene_id(bs) -> str:
    """Build a ZCAM scene ID (Sol{sol}_seq_RSM{rsm}) from bandset metadata."""
    try:
        sol = int(bs.metadata['SOL'].iloc[0])
        seq = str(bs.metadata['SEQ_ID'].iloc[0]).strip()
        rsm = int(bs.metadata['RSM'].min())
        return f"Sol{sol:04d}_{seq}_RSM{rsm}"
    except Exception:
        return bs.name


def _find_pcam_scenes(folder: Path):
    """Yield (seq_id, obs_ix, scene_id) for each PCAM pointing in folder."""
    try:
        products = scan_pcam_files(folder)
    except Exception:
        return

    def _band_area(path):
        try:
            d       = pdr.Data(path)
            img_key = 'IMAGE' if 'IMAGE' in d.keys() else 'Image_Object'
            arr     = d[img_key]
            return arr.shape[0] * arr.shape[1]
        except Exception:
            return 0

    def _frame_areas(group):
        """Return areas per row, reading only one file per unique SCLK to minimise IO."""
        representatives = group.drop_duplicates('SCLK')
        areas = representatives['PATH'].map(_band_area)
        areas.index = representatives.index
        return group['PATH'].map(lambda _: areas.max())

    for seq_id, group in products.groupby('SEQ_ID'):
        areas    = _frame_areas(group)
        max_area = areas.max()
        if max_area == 0:
            continue
        full_frame = group[areas == max_area]
        for obs_ix, obs in enumerate(split_pcam_observations(full_frame)):
            yield seq_id, obs_ix, _pcam_scene_id(obs)


def _find_zcam_scenes(folder: Path):
    """Yield (seq_id=None, obs_ix, scene_id) for each ZCAM pointing in folder."""
    try:
        groups = _scan_and_split(folder)
    except Exception:
        return
    for obs_ix, group in enumerate(groups):
        try:
            bs = _bandset_from_group(group)
            if len(bs.metadata) >= 3:
                yield None, obs_ix, _zcam_scene_id(bs)
        except Exception as e:
            print(f"  Warning: skipping obs {obs_ix} in {folder.name}: {e}")


def find_scenes(root: Path, suffix: str, dcs_with_fallback: bool = False):
    """
    Find all unprocessed scenes up to one level deep.

    Skips scenes whose NPZ already exists before any IO-heavy work.
    In dcs-with-fallback mode, skips if either the _dcs or plain variant exists.
    Yields (folder, instrument, seq_id, obs_ix, scene_id).
    """
    candidates = [root] + sorted(root / d.name for d in root.iterdir() if d.is_dir())
    seen = set()

    for folder in candidates:
        img_files = [f for f in folder.iterdir()
                     if f.is_file() and f.suffix.upper() in ('.IMG', '.IMQ')]
        if not img_files:
            continue

        instrument = 'PCAM' if _is_pcam_folder(folder) else 'ZCAM'
        finder     = _find_pcam_scenes if instrument == 'PCAM' else _find_zcam_scenes

        for seq_id, obs_ix, scene_id in finder(folder):
            key = (str(folder), seq_id, obs_ix)
            if key in seen:
                continue
            seen.add(key)

            if dcs_with_fallback:
                already_done = (
                    (Path(str(folder)) / f"{scene_id}_dcs.npz").exists() or
                    (Path(str(folder)) / f"{scene_id}.npz").exists()
                )
            else:
                already_done = (Path(str(folder)) / f"{scene_id}{suffix}.npz").exists()

            if already_done:
                print(f"  Skipping {scene_id} - already segmented.")
                continue

            yield folder, instrument, seq_id, obs_ix, scene_id


# ---------------------------------------------------------------------------
# Pipeline workers
# ---------------------------------------------------------------------------

def _loader_worker(scenes_chunk, suffix, use_dcs, use_dcs_with_fallback, load_queue, print_lock):
    """Load scenes from disk and push RGB images onto the queue."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    for folder, instrument, seq_id, obs_ix, scene_id in scenes_chunk:
        try:
            load_result = load_cube(
                iof_path         = str(folder),
                instrument       = instrument,
                seq_id           = seq_id,
                obs_ix           = obs_ix,
                do_apply_pixmaps = True,
                ignore_bayers    = False,
            )

            if use_dcs_with_fallback:
                # use DCS only if the preferred bands were all present
                stretch = load_result.get('stretch_bands', {})
                camera  = 'right' if instrument == 'ZCAM' else 'right'
                dcs_available = stretch.get(camera, False)
                actual_suffix = '_dcs' if dcs_available else ''
                rgb_img = make_dcs_rgb(load_result) if dcs_available else load_result['rgb_img']
            elif use_dcs:
                actual_suffix = '_dcs'
                rgb_img = make_dcs_rgb(load_result)
            else:
                actual_suffix = ''
                rgb_img = load_result['rgb_img']

            npz_path = Path(str(folder)) / f"{scene_id}{actual_suffix}.npz"
            if npz_path.exists():
                with print_lock:
                    print(f"  Skipping {scene_id} - already segmented.")
                continue

            load_queue.put((scene_id, rgb_img, npz_path))
        except Exception as e:
            with print_lock:
                print(f"  Load failed [{scene_id}]: {e}")

    load_queue.put(_SENTINEL)


def _segment(rgb_img: np.ndarray, sam_path: str, params: dict) -> np.ndarray:
    """Run SAM on rgb_img and return a 2D int32 segment label array."""
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
    parser.add_argument('--dcs-with-fallback', action='store_true',
        help='Use DCS if preferred bands exist, otherwise fall back to regular RGB. '
             'Files are saved with _dcs suffix only when DCS was actually used.')
    parser.add_argument('--points-per-side', type=int, default=32,
        help='SAM sampling density (default: 32)')
    parser.add_argument('--pred-iou-thresh', type=float, default=0.88,
        help='SAM IoU confidence threshold (default: 0.88)')
    parser.add_argument('--preserve-background', action='store_true',
        help='Assign unclassified pixels to segment 0 instead of the last segment')
    parser.add_argument('--workers', type=int, default=4,
        help='Number of parallel loader threads (default: 4)')
    args = parser.parse_args()

    root = Path(args.folder)
    if not root.is_dir():
        print(f"Error: {root} is not a directory.", file=sys.stderr)
        sys.exit(1)

    if args.dcs and args.dcs_with_fallback:
        print("Error: --dcs and --dcs-with-fallback are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    use_dcs_with_fallback = args.dcs_with_fallback
    suffix = '_dcs' if args.dcs else ''
    params = {
        'preserve_background': args.preserve_background,
        'points_per_side':     args.points_per_side,
        'pred_iou_thresh':     args.pred_iou_thresh,
    }

    print(f"Scanning {root} for scenes...")
    scenes = list(find_scenes(root, suffix, dcs_with_fallback=use_dcs_with_fallback))

    if not scenes:
        print("No scenes to process.")
        return

    total = len(scenes)
    print(f"\n{total} scene(s) to process. Loading with {args.workers} threads.\n")
    if args.dcs:
        print("DCS mode: segmenting decorrelation-stretched images (_dcs suffix).\n")
    elif use_dcs_with_fallback:
        print("DCS-with-fallback mode: uses DCS when available, regular RGB otherwise.\n")

    # Distribute scenes evenly across loader threads.
    chunk_size   = max(1, (total + args.workers - 1) // args.workers)
    chunks       = [scenes[i:i + chunk_size] for i in range(0, total, chunk_size)]
    load_queue   = queue.Queue(maxsize=args.workers * 2)  # bound to limit memory
    print_lock   = threading.Lock()

    # Start loader threads - they fill the queue as fast as disk allows.
    loaders = []
    for chunk in chunks:
        t = threading.Thread(
            target=_loader_worker,
            args=(chunk, suffix, args.dcs, use_dcs_with_fallback, load_queue, print_lock),
            daemon=True,
        )
        t.start()
        loaders.append(t)

    # Run SAM sequentially on the main thread while loader threads fill the queue.
    # Stop after receiving one sentinel from each loader.
    n_loaders  = len(loaders)
    done_count = 0
    attempted  = 0
    completed  = 0

    while done_count < n_loaders:
        item = load_queue.get()

        if item is _SENTINEL:
            done_count += 1
            continue

        scene_id, rgb_img, npz_path = item
        H, W      = rgb_img.shape[:2]
        attempted += 1

        with print_lock:
            print(f"[{attempted}/{total}] {scene_id}  {W}x{H}")

        try:
            segments = _segment(rgb_img, args.sam_path, params)
        except Exception as e:
            with print_lock:
                print(f"  Segmentation failed: {e}\n")
            continue

        n_segs = len(np.unique(segments)) - (1 if 0 in segments else 0)
        if n_segs == 0:
            with print_lock:
                print(f"  0 segments - skipping.\n")
            continue

        np.savez_compressed(str(npz_path), segments=segments, dcs=np.bool_(args.dcs))
        completed += 1
        with print_lock:
            print(f"  {n_segs} segments -> {npz_path.name}\n")

    for t in loaders:
        t.join()

    print(f"Done. {completed}/{total} scene(s) segmented.")


if __name__ == '__main__':
    main()
