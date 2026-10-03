"""Terminal entry point for the same pipeline used by Python and ROIStudio."""

import argparse
from dataclasses import asdict
from enum import Enum
import json
from pathlib import Path
from .utils.device import DEVICES, prepare_accelerators, resolve_device

from .core.config import (
    AlignmentConfig, LoadConfig, SegmentConfig, PreprocessConfig, ROIConfig,
    SpectralConfig, PerformanceConfig, SparcConfig, SegmentationBackend, ROIBackend,
)


def parser():
    result = argparse.ArgumentParser(description='Run SPARC on one multispectral scene.')
    result.add_argument('--config', type=Path, help='YAML pipeline settings; command-line flags override these')
    result.add_argument('--input', type=Path, help='Folder containing the calibrated IOF images')
    result.add_argument('--sam-path', type=Path, help='SAM checkpoint')
    result.add_argument('--instrument', choices=('ZCAM', 'PCAM'))
    result.add_argument('--seq-id')
    result.add_argument('--obs-index', type=int, help='Zero-based pointing index within the selected sequence/folder')
    result.add_argument('--alignment', choices=('homography', 'roma'))
    result.add_argument('--device', choices=DEVICES, help='Device for SAM and RoMa; auto prefers CUDA, then MPS, then CPU')
    result.add_argument('--roma-certainty', type=float, help='Minimum correspondence certainty, 0–1 (default .5)')
    result.add_argument('--roma-cycle-tolerance', type=float, help='Maximum round-trip error in pixels (default 3)')
    result.add_argument('--output', type=Path, help='New output directory (default: sparc-results/<timestamp>)')
    result.add_argument('--print-config', action='store_true', help='Print resolved settings and exit without loading data or models')
    result.add_argument('--verbose', action='store_true')
    return result


def make_config(args):
    """Parse the dataclass sections without importing optional ML packages."""
    data = {}
    if args.config:
        import yaml
        data = yaml.safe_load(args.config.read_text(encoding='utf-8')) or {}
        if not isinstance(data, dict):
            raise ValueError('Config must be a YAML mapping of pipeline sections')
    allowed = {'load', 'segment', 'preprocess', 'roi', 'spectral', 'performance'}
    if data.keys() - allowed:
        raise ValueError(f'Unknown config sections: {sorted(data.keys() - allowed)}')
    sections = {}
    for name in allowed:
        value = data.get(name, {})
        if not isinstance(value, dict):
            raise ValueError(f'{name} must be a mapping')
        sections[name] = dict(value)
    load, segment = sections['load'], sections['segment']
    alignment = load.pop('alignment', {})
    if not isinstance(alignment, dict):
        raise ValueError('load.alignment must be a mapping')
    alignment = dict(alignment)
    for key, value in (('iof_path', args.input), ('instrument', args.instrument),
                       ('seq_id', args.seq_id), ('obs_ix', args.obs_index)):
        if value is not None:
            load[key] = str(value) if isinstance(value, Path) else value
    if args.sam_path is not None:
        segment['sam_model_path'] = str(args.sam_path)
    for key, value in (('method', args.alignment), ('device', args.device),
                       ('certainty_threshold', args.roma_certainty), ('cycle_tolerance', args.roma_cycle_tolerance)):
        if value is not None:
            alignment[key] = value
    if args.device is not None:
        # GPU includes both CUDA and auto selection; resolve availability only
        # when actually running, so --print-config never imports PyTorch.
        segment['backend'] = 'cpu' if args.device == 'cpu' else 'gpu'
        segment['device'] = args.device
    load.setdefault('iof_path', '')
    segment.setdefault('sam_model_path', '')
    load['alignment'] = AlignmentConfig(**alignment)
    load['alignment'].validate()
    if 'backend' in segment:
        segment['backend'] = SegmentationBackend(segment['backend'])
    if 'backend' in sections['roi']:
        sections['roi']['backend'] = ROIBackend(sections['roi']['backend'])
    config = SparcConfig(
        load=LoadConfig(**load), segment=SegmentConfig(**segment),
        preprocess=PreprocessConfig(**sections['preprocess']), roi=ROIConfig(**sections['roi']),
        spectral=SpectralConfig(**sections['spectral']), performance=PerformanceConfig(**sections['performance']),
    )
    if config.load.instrument not in ('ZCAM', 'PCAM'):
        raise ValueError('instrument must be ZCAM or PCAM')
    if config.load.obs_ix < 0:
        raise ValueError('obs-index must be nonnegative')
    return config


def config_dict(config):
    return json.loads(json.dumps(asdict(config), default=lambda value: value.value if isinstance(value, Enum) else str(value)))


def run(config, output, verbose=False):
    """No Qt or ROIStudio imports, including during export."""
    if not config.load.iof_path or not Path(config.load.iof_path).is_dir():
        raise ValueError('Set --input (or load.iof_path) to an existing IOF folder')
    if not config.segment.sam_model_path or not Path(config.segment.sam_model_path).is_file():
        raise ValueError('Set --sam-path (or segment.sam_model_path) to an existing checkpoint')
    prepare_accelerators()
    if config.segment.device is not None:
        resolve_device(config.segment.device)
    if config.load.alignment.method == 'roma':
        resolve_device(config.load.alignment.device)
    import matplotlib
    matplotlib.use('Agg')
    from .core.functional import run_sparc
    from .core.result import export_spectra_csv, plot_result
    from .core.logging_utils import configure_logging
    import numpy as np
    configure_logging(verbose)
    # Refuse to overwrite an earlier run.
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(config_dict(config), indent=2), encoding='utf-8')
    result = run_sparc(config.load.iof_path, config.segment.sam_model_path, config)
    # validate() resolves automatic backends during the run; record the actual settings.
    (output / 'config.json').write_text(json.dumps(config_dict(config), indent=2), encoding='utf-8')
    export_spectra_csv(result, str(output / 'spectra.csv'))
    np.savez_compressed(output / 'result.npz', right_rois=result.final_rois,
                        left_rois=result.final_left_rois, spectra=result.final_spectra,
                        stds=result.final_stds, wavelengths=result.wavelengths, segments=result.segments)
    details = dict(scene_id=result.scene_id, instrument=result.instrument,
                   alignment=config.load.alignment.method,
                   right_rois=result.final_rois.tolist(), left_rois=result.final_left_rois.tolist(),
                   coordinate_system='loaded-image pixels: x, y, width, height',
                   alignment_coverage=(result._load_result or {}).get('alignment_coverage'))
    (output / 'rois.json').write_text(json.dumps(details, indent=2), encoding='utf-8')
    plot_result(result).savefig(output / 'overview.png', dpi=150, bbox_inches='tight')
    print(f'{result.scene_id}: {len(result.final_rois)} ROIs. Results: {output.resolve()}')
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        config = make_config(args)
        if args.print_config:
            print(json.dumps(config_dict(config), indent=2))
            return 0
        from datetime import datetime
        output = args.output or Path('sparc-results') / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        run(config, output, verbose=args.verbose)
    except (ValueError, TypeError, OSError, RuntimeError, ImportError) as exc:
        cli.exit(2, f'SPARC: {exc}\n')
    return 0
