"""Resource-lifecycle tests for SAM that do not require model weights."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).parents[1]


def _load_sam_segmentation_with_dependency_stubs():
    fake_sparc = types.ModuleType('sparc')
    fake_sparc.__path__ = []
    fake_segmentation = types.ModuleType('sparc.segmentation')
    fake_segmentation.__path__ = []
    fake_utils = types.ModuleType('sparc.utils')
    fake_utils.__path__ = []
    fake_memory = types.ModuleType('sparc.utils.memory')
    fake_memory.release_cuda_memory = Mock()
    fake_torch = types.ModuleType('torch')
    fake_torch.device = object
    fake_segment_anything = types.ModuleType('segment_anything')
    fake_segment_anything.sam_model_registry = {}
    fake_segment_anything.SamAutomaticMaskGenerator = object

    path = ROOT / 'src' / 'sparc' / 'segmentation' / 'sam_segmentation.py'
    spec = importlib.util.spec_from_file_location(
        'sparc.segmentation.sam_segmentation_under_test',
        path,
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            'sparc': fake_sparc,
            'sparc.segmentation': fake_segmentation,
            'sparc.utils': fake_utils,
            'sparc.utils.memory': fake_memory,
            'torch': fake_torch,
            'segment_anything': fake_segment_anything,
        },
    ):
        spec.loader.exec_module(module)
    return module


# A failed SAM run still has to give its GPU memory back before the next run.
class SamResourceLifecycleTests(unittest.TestCase):
    def test_failed_segmentation_releases_cuda_resources(self):
        segmentation = _load_sam_segmentation_with_dependency_stubs()
        cleanup = Mock()

        with (
            patch.object(segmentation, 'select_device', return_value='cuda:0'),
            patch.object(segmentation, 'detect_model_type', return_value='vit_h'),
            patch.object(segmentation, 'load_sam_model', return_value=object()),
            patch.object(
                segmentation,
                'generate_masks',
                side_effect=RuntimeError('CUDA out of memory'),
            ),
            patch.object(segmentation, 'release_cuda_memory', cleanup),
        ):
            with self.assertRaisesRegex(RuntimeError, 'CUDA out of memory'):
                segmentation.segment_image(
                    'sam_vit_h.pth',
                    np.zeros((4, 4, 3), dtype=np.uint8),
                )

        cleanup.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
