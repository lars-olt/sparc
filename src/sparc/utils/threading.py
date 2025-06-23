"""Threading utilities to fix KMeans memory leak warnings."""

import os
import warnings
import platform


def configure_threading():
    """Configure threading environment to avoid KMeans memory leak warnings."""

    # Set environment variables before any sklearn imports
    if platform.system() == "Windows":
        # Set OpenMP threading to 1 to avoid MKL memory leak
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"


def suppress_kmeans_warnings():
    """Suppress specific KMeans memory leak warnings."""

    warnings.filterwarnings(
        "ignore",
        message="KMeans is known to have a memory leak on Windows with MKL.*",
        category=UserWarning,
    )


def force_fix_kmeans_warnings():
    """
    Force fix for KMeans warnings - call this before running SPARC if warnings persist.
    """

    # Set threading environment variables
    configure_threading()

    # Suppress warnings
    suppress_kmeans_warnings()

    # Additional warning suppression for persistent cases
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    print(
        "Applied KMeans warning suppression. Threading configured for Windows compatibility."
    )


# Configure threading as early as possible
configure_threading()


class SafeKMeans:
    """Wrapper for KMeans with safe threading configuration."""

    def __init__(self, n_clusters, random_state=42, **kwargs):
        from sklearn.cluster import KMeans

        # Remove n_jobs from kwargs if present (KMeans doesn't support it)
        kwargs.pop("n_jobs", None)

        # Suppress warnings locally if they still appear
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="KMeans is known to have a memory leak.*"
            )

            self.kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init="auto",  # Use new default to avoid warnings
                **kwargs,
            )

    def fit(self, X):
        """Fit the KMeans model."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="KMeans is known to have a memory leak.*"
            )
            return self.kmeans.fit(X)

    def predict(self, X):
        """Predict cluster labels."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="KMeans is known to have a memory leak.*"
            )
            return self.kmeans.predict(X)

    def fit_predict(self, X):
        """Fit and predict in one step."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="KMeans is known to have a memory leak.*"
            )
            return self.kmeans.fit_predict(X)

    @property
    def cluster_centers_(self):
        """Get cluster centers."""
        return self.kmeans.cluster_centers_

    @property
    def inertia_(self):
        """Get inertia."""
        return self.kmeans.inertia_

    def __getattr__(self, name):
        """Delegate other attributes to the underlying KMeans object."""
        return getattr(self.kmeans, name)
