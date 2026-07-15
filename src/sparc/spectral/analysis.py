"""Spectral analysis and clustering functionality."""

import numpy as np
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import BayesianGaussianMixture
from kneed import KneeLocator


def cluster_with_bayesian_gmm(data: np.ndarray,
                              max_components: int = 20) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Find optimal clustering using Bayesian Gaussian Mixture Model.

    Tests different preprocessing methods and selects best based on log-likelihood.
    """
    preprocessing_methods = {
        'original':    data,
        'standardized': StandardScaler().fit_transform(data),
        'sam_features': create_sam_features(data, max_components)
    }

    results = {}
    for method_name, processed_data in preprocessing_methods.items():
        result = fit_bayesian_gmm(processed_data, max_components, method_name)
        results[method_name] = result

    best_method = max(results.keys(), key=lambda k: results[k]['log_likelihood'])
    return results[best_method], results


def fit_bayesian_gmm(data: np.ndarray,
                     max_components: int,
                     method_name: str) -> Dict[str, Any]:
    """Fit a Bayesian GMM and return cluster assignments and diagnostics."""
    bgmm = BayesianGaussianMixture(
        n_components=max_components,
        covariance_type='full',
        weight_concentration_prior=1.0 / max_components,
        reg_covar=1e-4,   # raised from 1e-6 for robustness with small sample counts
        random_state=42,
        max_iter=200,
    )

    try:
        labels = bgmm.fit_predict(data)
    except ValueError:
        # Fall back to diagonal covariance for small sample counts.
        bgmm = BayesianGaussianMixture(
            n_components=max_components,
            covariance_type='diag',
            weight_concentration_prior=1.0 / max_components,
            reg_covar=1e-4,
            random_state=42,
            max_iter=200,
        )
        labels = bgmm.fit_predict(data)

    active_components = np.sum(bgmm.weights_ > 0.01)
    log_likelihood    = bgmm.score(data) * len(data)

    return {
        'labels':         labels,
        'n_components':   active_components,
        'weights':        bgmm.weights_,
        'log_likelihood': log_likelihood,
        'model':          method_name,
        'model_object':   bgmm,
        'processed_data': data,
    }


def create_sam_features(spectra: np.ndarray, n_components: int) -> np.ndarray:
    """
    Create features based on spectral angle similarity.
    
    Args:
        spectra: Input spectra
        n_components: Number of reference spectra to use
        
    Returns:
        SAM-based feature array
    """
    from .metrics import spectral_angle_similarity
    from ..utils.threading import SafeKMeans
    
    kmeans = SafeKMeans(n_clusters=min(10, n_components), random_state=42)
    kmeans.fit(spectra)
    references = kmeans.cluster_centers_
    
    features = np.zeros((len(spectra), len(references)))
    
    for i, spectrum in enumerate(spectra):
        for j, reference in enumerate(references):
            features[i, j] = spectral_angle_similarity(spectrum, reference)
    
    return features


def find_optimal_k(data: np.ndarray, k_range: range = range(5, 10)) -> int:
    """
    Find optimal number of clusters using elbow method.
    
    Args:
        data: Data to cluster
        k_range: Range of k values to test
        
    Returns:
        Optimal number of clusters
    """
    from ..utils.threading import SafeKMeans
    
    inertias = []
    for k in k_range:
        kmeans = SafeKMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    
    knee_locator = KneeLocator(
        k_range, inertias,
        curve="convex",
        direction="decreasing"
    )
    
    if knee_locator.knee:
        return knee_locator.knee
    
    return k_range[np.argmin(np.gradient(inertias))]


def normalize_spectra(spectra: np.ndarray) -> np.ndarray:
    """
    Normalize spectra by removing minimum and scaling to [0, 1].
    
    Args:
        spectra: Input spectra array
        
    Returns:
        Normalized spectra
    """
    normalized = np.zeros_like(spectra)
    
    for i, spectrum in enumerate(spectra):
        spec_norm = spectrum - spectrum.min()
        spec_max = spec_norm.max()
        
        if spec_max > 0:
            spec_norm /= spec_max
        
        normalized[i] = spec_norm
    
    return normalized
