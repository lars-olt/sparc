import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from sparc.cli import make_config, parser
from sparc.core.config import LoadConfig, SegmentConfig, SparcConfig, SegmentationBackend
from sparc.utils.device import prepare_accelerators, resolve_device
from sparc.utils.memory import release_accelerator_memory
from sparc.experimental.roma import _model, _offload_model


def fake_torch(cuda=False, mps=False):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda, is_initialized=lambda: cuda, empty_cache=Mock()),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        mps=SimpleNamespace(empty_cache=Mock()),
        device=lambda name: SimpleNamespace(type=name),
        float16='float16', float32='float32',
    )


class DeviceTests(unittest.TestCase):
    def test_auto_prefers_cuda_then_mps_then_cpu(self):
        for cuda, mps, expected in ((True, True, 'cuda'), (False, True, 'mps'), (False, False, 'cpu')):
            with self.subTest(expected=expected), patch.dict(sys.modules, torch=fake_torch(cuda, mps)):
                self.assertEqual(resolve_device().type, expected)

    def test_cpu_override_and_unavailable_explicit_gpu(self):
        with patch.dict(sys.modules, torch=fake_torch(True, True)):
            self.assertEqual(resolve_device('cpu').type, 'cpu')
        for name in ('cuda', 'mps'):
            with self.subTest(name=name), patch.dict(sys.modules, torch=fake_torch()):
                with self.assertRaisesRegex(RuntimeError, 'unavailable'):
                    resolve_device(name)

    def test_mac_fallback_respects_user_setting(self):
        with patch('sys.platform', 'darwin'), patch.dict(os.environ, {}, clear=True):
            prepare_accelerators()
            self.assertEqual(os.environ['PYTORCH_ENABLE_MPS_FALLBACK'], '1')
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
            prepare_accelerators()
            self.assertEqual(os.environ['PYTORCH_ENABLE_MPS_FALLBACK'], '0')

    def test_mps_selection_applies_to_both_cli_models(self):
        config = make_config(parser().parse_args(['--alignment', 'roma', '--device', 'mps']))
        self.assertEqual(config.load.alignment.device, 'mps')
        self.assertEqual(config.segment.device, 'mps')
        with patch.dict(sys.modules, torch=fake_torch(mps=True)):
            config.validate()
        self.assertEqual(config.segment.backend, SegmentationBackend.GPU)

    def test_legacy_gpu_backend_uses_mps_instead_of_falling_back_to_cpu(self):
        config = SparcConfig(LoadConfig('.'), SegmentConfig('sam.pth', backend=SegmentationBackend.GPU))
        with patch.dict(sys.modules, torch=fake_torch(mps=True)):
            config.validate()
        self.assertEqual(config.segment.backend, SegmentationBackend.GPU)

    def test_cleanup_releases_mps_cache(self):
        torch = fake_torch(mps=True)
        with patch.dict(sys.modules, torch=torch):
            release_accelerator_memory()
        torch.mps.empty_cache.assert_called_once_with()
        torch.cuda.empty_cache.assert_not_called()


class RoMaDeviceTests(unittest.TestCase):
    def tearDown(self):
        _model.cache_clear()

    def test_non_cuda_models_use_float32_and_portable_correlation(self):
        factory = Mock()
        for device, dtype in (('cpu', 'float32'), ('mps', 'float32'), ('cuda', 'float16')):
            with self.subTest(device=device), patch.dict(
                sys.modules, torch=fake_torch(), romatch=SimpleNamespace(roma_outdoor=factory)
            ):
                _model(device)
                factory.assert_called_with(device=device, symmetric=True, use_custom_corr=False, amp_dtype=dtype)

    def test_unregistered_backbone_is_also_offloaded(self):
        backbone = Mock()
        model = Mock()
        model.encoder.dinov2_vitl14 = [backbone]
        _offload_model(model)
        model.to.assert_called_once_with('cpu')
        backbone.to.assert_called_once_with('cpu')
