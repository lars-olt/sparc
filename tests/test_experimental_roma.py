import contextlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from sparc.experimental.stereo import DenseStereoMapping
from sparc.experimental import roma


def identity(shape=(12, 16), dx=0):
    y, x = np.indices(shape, dtype=np.float32)
    lr = np.stack((x + dx, y), -1)
    rl = np.stack((x - dx, y), -1)
    return DenseStereoMapping(lr, rl, (x + dx < shape[1]), (x >= dx))


class DenseGeometryTests(unittest.TestCase):
    def test_identity_and_translation_map_both_directions(self):
        mapping = identity(dx=2)
        self.assertEqual(mapping.map_rect((5, 3, 4, 5), 'right'), (3, 3, 4, 5))
        self.assertEqual(mapping.map_rect((3, 3, 4, 5), 'left'), (5, 3, 4, 5))
        self.assertEqual(mapping.map_point((5, 3), 'right'), (3., 3.))
        self.assertIsNone(mapping.map_point((0, 3), 'right'))

    def test_inscribed_rectangle_never_crosses_invalid_holes(self):
        mapping = identity()
        mapping.left_valid[4:7, 5:8] = False
        rect = mapping.map_rect((2, 2, 10, 8), 'right')
        self.assertIsNotNone(rect)
        x, y, w, h = rect
        self.assertTrue(mapping.left_valid[y:y+h, x:x+w].all())
        self.assertGreaterEqual(x, 2)
        self.assertLessEqual(x + w, 12)
        self.assertLessEqual(y + h, 10)
        mapping.left_valid[:] = False
        self.assertIsNone(mapping.map_rect((2, 2, 10, 8), 'right'))

    def test_nonlinear_footprint_is_inscribed_not_bounded(self):
        mapping = identity()
        mapping.left_to_right[6:, :, 0] += 4
        rect = mapping.map_rect((6, 2, 4, 8), 'right')
        x, y, w, h = rect
        points = mapping.left_to_right[y:y+h, x:x+w]
        self.assertTrue(((points[..., 0] >= 6) & (points[..., 0] < 10)).all())

    def test_crop_updates_fields_without_modifying_original(self):
        mapping = identity(dx=2)
        crop = mapping.cropped((3, 2, 8, 7))
        self.assertEqual(crop.map_rect((3, 1, 3, 3), 'right'), (1, 1, 3, 3))
        self.assertEqual(crop.left_valid.shape, (7, 8))
        self.assertFalse(crop.right_valid[:, :2].any())
        self.assertEqual(mapping.map_point((6, 3), 'right'), (4., 3.))

    def test_warp_preserves_intensities_and_masks_missing_support(self):
        mapping = identity(dx=2)
        cube = np.arange(12 * 16, dtype=np.float32).reshape(1, 12, 16)
        warped = mapping.warp_left(cube)
        np.testing.assert_allclose(warped[:, 1:-1, 3:-1], cube[:, 1:-1, 1:-3])
        self.assertTrue(np.isnan(warped[:, :, :2]).all())


class ArrayTensor:
    def __init__(self, array):
        self.array = array

    def detach(self):
        return self

    float = cpu = detach

    def numpy(self):
        return self.array


class RoMaAdapterTests(unittest.TestCase):
    def test_import_errors_identify_environment_and_underlying_failure(self):
        errors = (
            (ModuleNotFoundError("No module named 'romatch'", name='romatch'),
             'RoMa is not installed'),
            (ModuleNotFoundError("No module named 'kornia'", name='kornia'),
             'RoMa could not be imported'),
            (ImportError('incompatible dependency'), 'RoMa could not be imported'),
        )
        real_import = __import__
        for error, expected in errors:
            with self.subTest(error=error):
                def import_module(name, *args, **kwargs):
                    if name == 'romatch':
                        raise error
                    return real_import(name, *args, **kwargs)

                with patch.dict(sys.modules, {'torch': SimpleNamespace()}):
                    with patch('builtins.__import__', side_effect=import_module):
                        with self.assertRaises(RuntimeError) as raised:
                            roma._model.__wrapped__('cpu')
                self.assertIn(expected, str(raised.exception))
                self.assertIn(sys.executable, str(raised.exception))
                self.assertIn(str(error), str(raised.exception))
                self.assertIs(raised.exception.__cause__, error)

    def test_batched_symmetric_output_has_correct_directions(self):
        h, w = 12, 16
        y, x = np.indices((h, w), dtype=np.float32)
        coords = np.stack((2 * (x + .5) / w - 1, 2 * (y + .5) / h - 1), -1)
        shift = np.array([4 / w, 0], dtype=np.float32)
        warp = np.concatenate((np.concatenate((coords, coords + shift), -1),
                               np.concatenate((coords - shift, coords), -1)), axis=1)
        model = Mock()
        model.to.return_value = model
        model.match.return_value = (ArrayTensor(warp[None]), ArrayTensor(np.ones((1, h, w*2), np.float32)))
        torch = SimpleNamespace(
            device=lambda name: SimpleNamespace(type=name),
            cuda=SimpleNamespace(is_available=lambda: False),
            inference_mode=contextlib.nullcontext,
        )
        with patch.dict(sys.modules, {'torch': torch}), patch.object(roma, '_model', return_value=model):
            mapping = roma.compute_mapping(np.zeros((h, w)), np.zeros((h, w)))
        self.assertEqual(mapping.map_rect((5, 3, 4, 4), 'right'), (3, 3, 4, 4))
        self.assertEqual(mapping.map_rect((3, 3, 4, 4), 'left'), (5, 3, 4, 4))
        self.assertEqual(model.match.call_args.args[0].mode, 'RGB')
        model.to.assert_any_call('cpu')

    def test_scene_rebuilds_cube_recipe_but_preserves_raw_data(self):
        left = np.full((2, 12, 16), .2, np.float32)
        right = np.full((2, 12, 16), .4, np.float32)
        scene = dict(instrument='ZCAM', base_bands={'L1': left[0], 'R1': right[0]},
                     left_cube=left, right_cube=right,
                     left_band_keys=['L1', 'L2'], right_band_keys=['R1', 'R2'],
                     merged_band_recipe=[('stereo', 's', 'L1', 'R1'),
                                         ('left_only', 'l', 'L2', None),
                                         ('right_only', 'r', None, 'R2')])
        with patch.object(roma, 'compute_mapping', return_value=identity()):
            result = roma.align_scene(scene)
        np.testing.assert_allclose(result['cube'][:, 3, 3], [.3, .2, .4])
        self.assertIs(result['left_cube'], left)
        self.assertIsNone(result['homography_matrix'])
        self.assertNotIn('stereo_mapping', scene)

    def test_frozen_mode_rejects_before_importing_torch(self):
        with patch.object(sys, 'frozen', True, create=True):
            with self.assertRaisesRegex(RuntimeError, 'source'):
                roma.compute_mapping(None, None)


