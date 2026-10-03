"""Optional stereo alignment dispatch shared by API, CLI, and GUI consumers."""

import importlib
import sys

from ..core.config import AlignmentConfig


def align_loaded_scene(scene, alignment=None):
    config = alignment or AlignmentConfig()
    config.validate()
    if config.method == 'homography':
        return scene
    if getattr(sys, 'frozen', False):
        raise RuntimeError('Experimental RoMa is unavailable in packaged builds.')
    return importlib.import_module('sparc.experimental.roma').align_scene(scene, config)
