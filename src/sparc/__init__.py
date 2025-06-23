"""SPARC: Spectral Pattern Analysis for ROI Classification."""

from .core.sparc import Sparc
from .core.constants import *
from .utils.threading import force_fix_kmeans_warnings

__version__ = "1.0.0"
__author__ = "Lars Olt"
__email__ = "larsolt1@gmail.com"

__all__ = ['Sparc', 'force_fix_kmeans_warnings']