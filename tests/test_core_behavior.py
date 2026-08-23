"""Tests for deterministic SPARC behavior that does not require model weights."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from sparc.core.config import (
    LoadConfig,
    ROIBackend,
    SegmentConfig,
    SegmentationBackend,
    SparcConfig,
)
from sparc.preprocessing.calibration import (
    apply_photometric_calibration,
    extract_incidence_angle,
)
from sparc.spectral.metrics import (
    compute_roi_spectra,
    correlation_distance,
    euclidean_distance,
    spectral_angle_distance,
    spectral_angle_similarity,
)


class SpectralMetricTests(unittest.TestCase):
    def test_standard_distance_cases(self):
        horizontal = np.array([1.0, 0.0])
        vertical = np.array([0.0, 1.0])

        self.assertAlmostEqual(spectral_angle_distance(horizontal, horizontal), 0.0)
        self.assertAlmostEqual(
            spectral_angle_distance(horizontal, vertical),
            np.pi / 2,
        )
        self.assertAlmostEqual(spectral_angle_similarity(horizontal, vertical), 0.0)
        self.assertAlmostEqual(euclidean_distance(horizontal, vertical), np.sqrt(2))

    def test_zero_and_constant_spectra_have_defined_distances(self):
        zeros = np.zeros(3)
        constant = np.ones(3)

        self.assertAlmostEqual(spectral_angle_distance(zeros, constant), np.pi / 2)
        self.assertEqual(correlation_distance(constant, constant), 1.0)

    def test_roi_spectra_use_inclusive_rectangle_coordinates(self):
        cube = np.arange(18, dtype=float).reshape(2, 3, 3)

        means, standard_deviations = compute_roi_spectra(cube, [(0, 0, 1, 1)])
        expected = cube[:, 0:2, 0:2]

        np.testing.assert_allclose(means[0], expected.mean(axis=(1, 2)))
        np.testing.assert_allclose(
            standard_deviations[0],
            expected.std(axis=(1, 2)),
        )


class CalibrationTests(unittest.TestCase):
    def test_zcam_incidence_angle_calibrates_iof(self):
        metadata = pd.DataFrame({'INCIDENCE_ANGLE': [60.0, 60.0]})
        cube = np.ones((2, 2, 2), dtype=float)

        calibrated = apply_photometric_calibration(cube, metadata, True)

        np.testing.assert_allclose(calibrated, 2.0)

    def test_pancam_solar_elevation_is_converted_to_incidence_angle(self):
        metadata = {'SOLAR_ELEVATION': {'value': -30.0}}

        self.assertAlmostEqual(extract_incidence_angle(metadata), 60.0)

    def test_missing_angle_returns_original_cube(self):
        cube = np.ones((1, 2, 2), dtype=float)

        with self.assertLogs('sparc.preprocessing.calibration', level='WARNING'):
            result = apply_photometric_calibration(cube, {}, True)

        self.assertIs(result, cube)


class ConfigurationTests(unittest.TestCase):
    def make_config(self):
        return SparcConfig(
            load=LoadConfig(iof_path='scene'),
            segment=SegmentConfig(sam_model_path='sam.pth'),
        )

    def test_mutable_defaults_are_not_shared(self):
        first = self.make_config()
        second = self.make_config()

        first.preprocess.shadow_kwargs['percentiles'] = (10, 90)

        self.assertEqual(second.preprocess.shadow_kwargs['percentiles'], (20, 100))

    def test_threaded_validation_selects_one_fewer_physical_core(self):
        config = self.make_config()
        config.roi.backend = ROIBackend.THREADED
        fake_psutil = SimpleNamespace(cpu_count=lambda logical: 8)

        with patch.dict(sys.modules, {'psutil': fake_psutil}):
            config.validate()

        self.assertEqual(config.performance.n_threads, 7)

    def test_gpu_validation_falls_back_when_torch_is_unavailable(self):
        config = self.make_config()
        config.segment.backend = SegmentationBackend.GPU

        with patch.dict(sys.modules, {'torch': None}):
            config.validate()

        self.assertEqual(config.segment.backend, SegmentationBackend.CPU)


if __name__ == '__main__':
    unittest.main()
