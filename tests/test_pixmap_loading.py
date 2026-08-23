import warnings
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from sparc.data.loading import (
    _bandset_from_group,
    _load_zcam_base_bands,
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


class FakeZcamBandset:
    def __init__(self, metadata):
        self.metadata = metadata

    def format_metadata(self):
        pass


class FakeMaskedBandset:
    def __init__(self):
        self.band = np.ma.array(
            [[1.0, 0.0], [2.0, 3.0]],
            mask=[[False, True], [False, False]],
            dtype=np.float32,
        )

    def load(self, bands):
        pass

    def get_band(self, band):
        return self.band

    def bulk_debayer(self, bands):
        self.band = np.asarray(self.band)


def fake_module(name, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


# When two filters collide, the complete downlink should always win.
def test_bandset_prefers_complete_product_over_partial_duplicate():
    group = pd.DataFrame(
        {
            "FILTER": ["L5", "L5"],
            "COMPLETION": ["PARTIAL", "COMPLETE_CHECKSUM_PASS"],
            "PATH": ["partial.img", "complete.img"],
        }
    )

    scan = fake_module(
        "asdf.scan",
        rate_completion=lambda rows: rows["COMPLETION"].eq(
            "COMPLETE_CHECKSUM_PASS"
        ),
        rate_cal_offset=lambda _rows: None,
    )
    zcam_bandset = fake_module(
        "asdf.zcam_bandset",
        ZcamBandSet=FakeZcamBandset,
    )
    with patch.dict(sys.modules, {
        "asdf.scan": scan,
        "asdf.zcam_bandset": zcam_bandset,
    }):
        bandset = _bandset_from_group(group)

    assert bandset.metadata["PATH"].tolist() == ["complete.img"]


# Debayering should not turn a known bad source pixel back into real data.
def test_zcam_load_restores_source_mask_as_nan_after_debayering():
    bandset = FakeMaskedBandset()

    with patch("sparc.data.loading.ZCAM_CROP", (0, 0, 0, 0)):
        bands = _load_zcam_base_bands(bandset, ["L5"])

    assert np.isnan(bands["L5"][0, 1])
    assert bands["L5"][1, 1] == 3.0


# A partial set of pixel maps should still be associated with the right files.
def test_load_zcam_pixmaps_loads_available_maps():
    bandset = FakeBandset(["left.img", "right.img"])
    result = ({"left.img": "left-map.img"}, [])
    scan = fake_module(
        "asdf.scan",
        find_obs_metamaps=lambda _paths, code: result,
    )
    with patch.dict(sys.modules, {"asdf.scan": scan}):
        pixmaps = _load_zcam_pixmaps(bandset)

    assert pixmaps.keys() == {"L1"}
    assert bandset.associated == (
        {"left.img": "left-map.img", "right.img": ""},
        "pix_map",
    )


# A complete set of maps should be passed through without changing the paths.
def test_load_zcam_pixmaps_associates_and_loads_complete_maps():
    bandset = FakeBandset(["left.img", "right.img"])
    metamaps = {
        "left.img": "left-map.img",
        "right.img": "right-map.img",
    }
    scan = fake_module(
        "asdf.scan",
        find_obs_metamaps=lambda _paths, code: (metamaps, []),
    )
    with patch.dict(sys.modules, {"asdf.scan": scan}):
        pixmaps = _load_zcam_pixmaps(bandset)

    assert pixmaps.keys() == {"L1"}
    assert bandset.associated == (metamaps, "pix_map")


# Each filter should only use the bad-pixel flags from its own map.
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


# The three Bayer channels all come from the same clear-filter pixel map.
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


# Masked Pancam data should still produce a clean RGB preview.
def test_pcam_rgb_handles_masked_pixels_without_warnings():
    band = np.ma.array(
        np.arange(100, dtype=np.float32).reshape(10, 10),
        mask=np.eye(10, dtype=bool),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        rgb = _pcam_rgb(band, band + 1, band + 2)

    assert rgb.shape == (10, 10, 3)


# The ZCAM preview path should be just as quiet with masked data.
def test_dcs_rgb_handles_masked_pixels_without_warnings():
    band = np.ma.array(
        np.arange(100, dtype=np.float32).reshape(10, 10),
        mask=np.eye(10, dtype=bool),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        rgb = dcs_rgb(band, band + 1, band + 2)

    assert rgb.shape == (10, 10, 3)
