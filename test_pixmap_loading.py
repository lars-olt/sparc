import warnings
from types import SimpleNamespace
from unittest.mock import patch

import asdf
import numpy as np

from sparc.data.loading import (
    _load_zcam_pixmaps,
    _pcam_rgb,
    apply_pixel_masks,
    dcs_rgb,
)


class FakeBandset:
    def __init__(self, paths):
        self.metadata = {"PATH": SimpleNamespace(unique=lambda: paths)}
        self.pixmaps = {}
        self.associated = None

    def associate_metamaps(self, metamaps, code):
        self.associated = (metamaps, code)

    def load_metamaps(self, code):
        self.pixmaps = {"L1": np.zeros((2, 2), dtype=np.uint8)}


def test_load_zcam_pixmaps_loads_available_maps():
    bandset = FakeBandset(["left.img", "right.img"])
    result = ({"left.img": "left-map.img"}, [])
    with patch("asdf.scan.find_obs_metamaps", return_value=result):
        pixmaps = _load_zcam_pixmaps(bandset)

    assert pixmaps.keys() == {"L1"}
    assert bandset.associated == (
        {"left.img": "left-map.img", "right.img": ""},
        "pix_map",
    )


def test_load_zcam_pixmaps_associates_and_loads_complete_maps():
    bandset = FakeBandset(["left.img", "right.img"])
    metamaps = {
        "left.img": "left-map.img",
        "right.img": "right-map.img",
    }
    with patch("asdf.scan.find_obs_metamaps", return_value=(metamaps, [])):
        pixmaps = _load_zcam_pixmaps(bandset)

    assert pixmaps.keys() == {"L1"}
    assert bandset.associated == (metamaps, "pix_map")


def test_pixel_masks_remove_saturated_and_hot_pixels_by_band():
    bands = {
        "L1": np.ones((2, 2), dtype=np.float32),
        "L2": np.ones((2, 2), dtype=np.float32),
        "R1": np.ones((2, 2), dtype=np.float32),
    }
    pixmaps = {
        "L1": np.array([[0, 4], [0, 0]], dtype=np.uint8),
        "L2": np.array([[0, 0], [5, 0]], dtype=np.uint8),
        "R1": np.array([[0, 0], [0, 1]], dtype=np.uint8),
    }

    masked = apply_pixel_masks(bands, pixmaps)

    assert np.isnan(masked["L1"][0, 1])
    assert np.isfinite(masked["L1"][1, 0])
    assert np.isnan(masked["L2"][1, 0])
    assert np.isfinite(masked["L2"][0, 1])
    assert np.isnan(masked["R1"][1, 1])
    assert np.isfinite(masked["R1"][0, 1])


def test_pixel_masks_use_clear_filter_map_for_bayer_channels():
    bands = {
        "L0R": np.ones((2, 2), dtype=np.float32),
        "L0G": np.ones((2, 2), dtype=np.float32),
        "L0B": np.ones((2, 2), dtype=np.float32),
    }
    pixmaps = {
        "L0": np.array([[0, 5], [0, 0]], dtype=np.uint8),
    }

    masked = apply_pixel_masks(bands, pixmaps)

    for band in bands:
        assert np.isnan(masked[band][0, 1])
        assert np.isfinite(masked[band][1, 1])


def test_pcam_rgb_handles_masked_pixels_without_warnings():
    band = np.ma.array(
        np.arange(100, dtype=np.float32).reshape(10, 10),
        mask=np.eye(10, dtype=bool),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        rgb = _pcam_rgb(band, band + 1, band + 2)

    assert rgb.shape == (10, 10, 3)


def test_dcs_rgb_handles_masked_pixels_without_warnings():
    band = np.ma.array(
        np.arange(100, dtype=np.float32).reshape(10, 10),
        mask=np.eye(10, dtype=bool),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        rgb = dcs_rgb(band, band + 1, band + 2)

    assert rgb.shape == (10, 10, 3)
