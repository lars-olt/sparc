"""Spectral analysis and clustering functionality."""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import BayesianGaussianMixture
from scipy.fft import fft, fftfreq
from kneed import KneeLocator

from .metrics import sam_based_features
from ..utils.threading import SafeKMeans, suppress_kmeans_warnings

# Suppress KMeans warnings
suppress_kmeans_warnings()


def detect_unique_spectra(
    spectra: np.ndarray, contamination: float = 0.1, freq_threshold: float = 0.7
) -> np.ndarray:
    """
    Detect unique/outlier spectra using frequency domain analysis.

    This method identifies spectra with unusual high-frequency content,
    which often indicates unique spectral signatures.

    Args:
        spectra: Array of spectra to analyze
        contamination: Expected fraction of outliers
        freq_threshold: Threshold for high vs low frequency separation

    Returns:
        Boolean mask indicating outlier spectra
    """
    n_spectra, _ = spectra.shape
    hf_noise_ratios = np.zeros(n_spectra)

    for i, spectrum in enumerate(spectra):
        # Normalize
        spectrum_centered = spectrum - np.mean(spectrum)

        # Compute FFT and power spectrum
        fft_spectrum = fft(spectrum_centered)
        freqs = fftfreq(len(spectrum_centered))
        power_spectrum = np.abs(fft_spectrum) ** 2

        # Only consider positive frequencies
        pos_freq_mask = freqs > 0
        pos_freqs = freqs[pos_freq_mask]
        pos_power = power_spectrum[pos_freq_mask]

        if len(pos_power) == 0:
            continue

        # Separate high and low frequency power
        hf_threshold_freq = np.percentile(pos_freqs, freq_threshold * 100)
        hf_mask = pos_freqs >= hf_threshold_freq
        lf_mask = pos_freqs < hf_threshold_freq

        hf_power = np.sum(pos_power[hf_mask]) if np.any(hf_mask) else 0
        lf_power = np.sum(pos_power[lf_mask]) if np.any(lf_mask) else 1

        hf_noise_ratios[i] = hf_power / (lf_power + 1e-10)

    # Use percentile-based threshold for outlier detection
    threshold = np.percentile(hf_noise_ratios, (1 - contamination) * 100)
    outlier_labels = hf_noise_ratios > threshold

    return outlier_labels


def auto_bayesian_gmm(
    data: np.ndarray, max_components: int = 20
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Automatically find optimal Bayesian Gaussian Mixture Model.

    Tests different preprocessing methods and selects the best based on log-likelihood.

    Args:
        data: Input spectral data
        max_components: Maximum number of components to test

    Returns:
        Tuple of (best_result, all_results)
    """
    # Test different preprocessing approaches
    methods = {
        "original": data,
        "standardized": StandardScaler().fit_transform(data),
        "sam_features": sam_based_features(data, max_components),
    }

    results = {}

    for method_name, processed_data in methods.items():
        # Bayesian GMM with automatic component selection
        bgmm = BayesianGaussianMixture(
            n_components=max_components,
            covariance_type="full",
            weight_concentration_prior=1.0
            / max_components,  # Encourages fewer components
            random_state=42,
            max_iter=200,
        )

        labels = bgmm.fit_predict(processed_data)

        # Count active components (with significant weight)
        active_components = np.sum(bgmm.weights_ > 0.01)

        # Use log-likelihood for model selection
        log_likelihood = bgmm.score(processed_data) * len(processed_data)

        results[method_name] = {
            "labels": labels,
            "n_components": active_components,
            "weights": bgmm.weights_,
            "log_likelihood": log_likelihood,
            "model": method_name,
        }

        print(
            f"{method_name}: {active_components} components, Log-likelihood: {log_likelihood:.2f}"
        )

    # Select best method based on highest log-likelihood
    best_method = max(results.keys(), key=lambda k: results[k]["log_likelihood"])

    print(
        f"\nBest method: {best_method} with {results[best_method]['n_components']} components"
    )

    return results[best_method], results


def auto_k_elbow(X: np.ndarray, k_range: range = range(5, 10)) -> int:
    """
    Automatically determine optimal number of clusters using elbow method.

    Args:
        X: Data to cluster
        k_range: Range of k values to test

    Returns:
        Optimal number of clusters
    """
    inertias = []
    for k in k_range:
        kmeans = SafeKMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    kl = KneeLocator(k_range, inertias, curve="convex", direction="decreasing")
    return kl.knee if kl.knee else k_range[np.argmin(np.gradient(inertias))]


def cluster_spectra(spectra: np.ndarray) -> np.ndarray:
    """
    Cluster spectra using automatic k-means with elbow method.

    Args:
        spectra: Array of normalized spectra

    Returns:
        Cluster labels for each spectrum
    """
    n_clusters = auto_k_elbow(spectra)
    k_means = SafeKMeans(n_clusters=n_clusters, random_state=42)
    classifications = k_means.fit_predict(spectra)

    return classifications


def normalize_spectra(spectra: np.ndarray) -> np.ndarray:
    """
    Normalize spectra by removing minimum and scaling by range.

    Args:
        spectra: Input spectra array

    Returns:
        Normalized spectra
    """
    normalized = []

    for spectrum in spectra:
        spec_norm = spectrum.copy()

        # Remove minimum
        spec_min = spec_norm.min()
        spec_norm -= spec_min

        # Scale by maximum
        spec_max = spec_norm.max()
        if spec_max > 0:
            spec_norm /= spec_max

        normalized.append(spec_norm)

    return np.array(normalized)
