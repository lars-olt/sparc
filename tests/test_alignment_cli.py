import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from sparc.cli import main, parser, make_config, run
from sparc.core.config import AlignmentConfig, LoadConfig, SegmentConfig, SparcConfig
from sparc.core.pipeline import selection_step
from sparc.core.state import SparcState
from sparc.data.alignment import align_loaded_scene
from sparc.data.loading import load_cube
from sparc.experimental import roma


class AlignmentPipelineTests(unittest.TestCase):
    def test_loader_dispatches_alignment_without_gui(self):
        config = AlignmentConfig(method='roma', device='cpu')
        scene, aligned = {}, {'alignment_method': 'roma'}
        with patch('sparc.data.loading._load_zcam_cube', return_value=scene), \
             patch.object(roma, 'align_scene', return_value=aligned) as align:
            result = load_cube('.', 'ZCAM', None, 0, True, False, alignment=config)
        self.assertIs(result, aligned)
        align.assert_called_once_with(scene, config)

    def test_default_alignment_never_imports_roma(self):
        scene = {}
        with patch('sparc.data.alignment.importlib.import_module') as load_module:
            self.assertIs(align_loaded_scene(scene), scene)
        load_module.assert_not_called()

    def test_explicit_cuda_does_not_fall_back(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, torch=torch), patch.object(roma, '_model') as model:
            with self.assertRaisesRegex(RuntimeError, 'CUDA'):
                roma.compute_mapping(None, None, AlignmentConfig(method='roma', device='cuda'))
        model.assert_not_called()

    def test_dense_selection_keeps_rectangles_spectra_and_indices_together(self):
        for mapped in ((1, 3, 3, 5), None):
            with self.subTest(mapped=mapped):
                mapping = Mock()
                mapping.map_rect.side_effect = [None, mapped]
                state = SparcState(
                    load_result={'stereo_mapping': mapping},
                    area_filtered_rois=np.array([[2, 3, 4, 5], [8, 3, 4, 5]]),
                    albedo_valid_indices=np.array([True, True]),
                    roi_spectra=np.array([[.2], [.4]]), roi_stds=np.array([[.01], [.02]]),
                    clustering_result={'labels': np.array([0, 1])},
                )
                config = SparcConfig(LoadConfig('.'), SegmentConfig('sam.pth'))
                with patch('sparc.core.pipeline.select_representative_rois', return_value=[1, 0]):
                    selection_step(state, config)
                if mapped is None:
                    self.assertEqual(state.final_left_rois.shape, (0, 4))
                    self.assertEqual(state.final_rois.shape, (0, 4))
                    self.assertEqual(state.final_spectra.shape, (0, 1))
                else:
                    np.testing.assert_array_equal(state.final_left_rois, [mapped])
                    np.testing.assert_array_equal(state.final_rois, [[2, 3, 4, 5]])
                    np.testing.assert_array_equal(state.final_spectra, [[.2]])
                    np.testing.assert_array_equal(state.final_stds, [[.01]])
                    np.testing.assert_array_equal(state.roi_indices, [0])

    def test_threshold_controls_valid_correspondences(self):
        y, x = np.indices((4, 5), dtype=np.float32)
        coords = np.stack((2 * (x + .5) / 5 - 1, 2 * (y + .5) / 4 - 1), -1)
        source = np.ones((4, 5), dtype=np.float32)
        certainty = np.full_like(source, .6)
        _, low = roma._pixel_field(coords, certainty, source, source, .5)
        _, high = roma._pixel_field(coords, certainty, source, source, .7)
        self.assertTrue(low.any())
        self.assertFalse(high.any())


class CommandLineTests(unittest.TestCase):
    def test_yaml_and_command_line_precedence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'settings.yml'
            path.write_text('load:\n  alignment:\n    method: roma\n    certainty_threshold: 0.7\n'
                            'segment:\n  preserve_background: true\nroi:\n  backend: threaded\n')
            config = make_config(parser().parse_args(['--config', str(path), '--device', 'cpu',
                                                     '--roma-certainty', '.8', '--obs-index', '2']))
        self.assertEqual(config.load.alignment.method, 'roma')
        self.assertEqual(config.load.alignment.certainty_threshold, .8)
        self.assertEqual(config.load.alignment.device, 'cpu')
        self.assertEqual(config.segment.backend.value, 'cpu')
        self.assertTrue(config.segment.preserve_background)
        self.assertEqual(config.roi.backend.value, 'threaded')
        self.assertEqual(config.load.obs_ix, 2)

    def test_unknown_config_keys_and_invalid_threshold_fail(self):
        with self.assertRaises(ValueError):
            make_config(parser().parse_args(['--roma-certainty', '1.1']))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'settings.yml'
            path.write_text('load:\n  alignmnt: roma\n')
            with self.assertRaises(TypeError):
                make_config(parser().parse_args(['--config', str(path)]))

    def test_print_config_runs_without_inference(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch('sparc.cli.run') as execute:
            self.assertEqual(main(['--alignment', 'roma', '--device', 'cuda', '--print-config']), 0)
        execute.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())['load']['alignment']['device'], 'cuda')

    def test_help_does_not_import_optional_stack_or_gui(self):
        script = '''
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'romatch', 'PyQt5', 'controllers', 'workers'}:
            raise AssertionError('unexpected import: ' + fullname)
sys.meta_path.insert(0, Block())
from sparc.cli import main
main(['--alignment', 'roma', '--print-config'])
'''
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_exports_sparc_result_and_refuses_overwrite(self):
        result = SimpleNamespace(
            scene_id='test', instrument='ZCAM', final_rois=np.array([[2, 3, 4, 5]]),
            final_left_rois=np.array([[1, 3, 3, 5]]), final_spectra=np.array([[.2]]),
            final_stds=np.array([[.01]]), wavelengths=[500], segments=np.zeros((8, 8)),
            _load_result={'alignment_coverage': .9},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'sam.pth'
            checkpoint.touch()
            output = root / 'results'
            config = SparcConfig(LoadConfig(str(root)), SegmentConfig(str(checkpoint)))
            with patch('sparc.core.functional.run_sparc', return_value=result) as pipeline, \
                 patch('sparc.core.result.plot_result'):
                run(config, output)
                pipeline.assert_called_once_with(str(root), str(checkpoint), config)
                with self.assertRaises(FileExistsError):
                    run(config, output)
                self.assertEqual(pipeline.call_count, 1)
            saved = json.loads((output / 'rois.json').read_text())
            self.assertEqual(saved['left_rois'], [[1, 3, 3, 5]])
            with np.load(output / 'result.npz') as arrays:
                np.testing.assert_array_equal(arrays['left_rois'], result.final_left_rois)
            self.assertTrue((output / 'spectra.csv').is_file())
