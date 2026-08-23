"""SPARC spectral clustering and metrics, exposed lazily."""

from .._lazy import public_dir, resolve_attribute


_LAZY_IMPORTS = {
    "cluster_with_bayesian_gmm": (".analysis", "cluster_with_bayesian_gmm"),
    "fit_bayesian_gmm": (".analysis", "fit_bayesian_gmm"),
    "create_sam_features": (".analysis", "create_sam_features"),
    "find_optimal_k": (".analysis", "find_optimal_k"),
    "normalize_spectra": (".analysis", "normalize_spectra"),
    "spectral_angle_distance": (".metrics", "spectral_angle_distance"),
    "spectral_angle_similarity": (".metrics", "spectral_angle_similarity"),
    "euclidean_distance": (".metrics", "euclidean_distance"),
    "correlation_distance": (".metrics", "correlation_distance"),
    "compute_roi_spectra": (".metrics", "compute_roi_spectra"),
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name):
    return resolve_attribute(globals(), __name__, name, _LAZY_IMPORTS)


def __dir__():
    return public_dir(globals(), _LAZY_IMPORTS)
