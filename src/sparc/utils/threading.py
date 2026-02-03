"""Threading utilities and safe KMeans wrapper."""

import warnings
import os
import psutil
import platform
from functools import wraps
from typing import Any, Callable, Iterable, List, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# --- Configuration & Environment ---

def configure_worker_env(n_threads: int = 1) -> None:
    """
    Set environment variables to restrict BLAS/MKL threads in workers.
    
    This is crucial when running many KMeans instances in parallel to avoid
    oversubscription of CPU cores.
    """
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)

def configure_global_settings():
    """
    Apply global settings for the SPARC pipeline.
    
    Configures threading environment variables for Windows/MKL.
    Suppresses known KMeans memory leak warnings.
    """
    if platform.system() == "Windows":
        configure_worker_env(1)

    warnings.filterwarnings(
        "ignore",
        message="KMeans is known to have a memory leak.*",
        category=UserWarning
    )
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# --- Decorators & Runners ---

def suppress_kmeans_warnings(func: Callable) -> Callable:
    """Decorator to suppress Windows MKL memory leak warnings."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", 
                message="KMeans is known to have a memory leak.*"
            )
            return func(*args, **kwargs)
    return wrapper

def run_parallel(
    func: Callable,
    items: Iterable[Any],
    n_jobs: Optional[int] = None,
    backend: str = "thread",
) -> List[Any]:
    """
    Execute a function in parallel over a list of items.
    
    Args:
        func: Function to apply to each item
        items: List of items to process
        n_jobs: Number of workers (None = auto-detect)
        backend: 'thread' for ThreadPoolExecutor or 'process' for ProcessPoolExecutor
        
    Returns:
        List of results
    """
    # Auto-detect CPUs if not provided
    if n_jobs is None:
        n_jobs = max(1, psutil.cpu_count(logical=False) or 1)
        
    # Ensure worker threads don't spawn sub-threads
    configure_worker_env(1)
    
    Executor = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
    
    with Executor(max_workers=n_jobs) as executor:
        results = list(executor.map(func, items))
        
    return results


# --- Safe Classes ---

class SafeKMeans:
    """
    Wrapper for sklearn KMeans with proper warning suppression.
    
    This wrapper handles the Windows MKL memory leak warning and ensures
    consistent behavior across platforms.
    """
    
    @suppress_kmeans_warnings
    def __init__(self, n_clusters: int, random_state: int = 42, **kwargs):
        """
        Initialize SafeKMeans wrapper.
        
        Args:
            n_clusters: Number of clusters
            random_state: Random seed for reproducibility
            **kwargs: Additional KMeans parameters
        """
        from sklearn.cluster import KMeans
        
        # Remove n_jobs if present (deprecated in newer sklearn)
        kwargs.pop('n_jobs', None)
        
        self._kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init='auto',
            **kwargs
        )
    
    @suppress_kmeans_warnings
    def fit(self, X: Any) -> 'SafeKMeans':
        """Fit the model."""
        self._kmeans.fit(X)
        return self
    
    @suppress_kmeans_warnings
    def predict(self, X: Any) -> Any:
        """Predict cluster labels."""
        return self._kmeans.predict(X)
    
    @suppress_kmeans_warnings
    def fit_predict(self, X: Any) -> Any:
        """Fit and predict in one step."""
        return self._kmeans.fit_predict(X)
    
    @property
    def cluster_centers_(self):
        """Get cluster centers."""
        return self._kmeans.cluster_centers_
    
    @property
    def inertia_(self):
        """Get inertia (sum of squared distances to centers)."""
        return self._kmeans.inertia_
    
    @property
    def labels_(self):
        """Get cluster labels."""
        return self._kmeans.labels_
    
    def __getattr__(self, name: str) -> Any:
        """Delegate other attributes to underlying KMeans."""
        return getattr(self._kmeans, name)