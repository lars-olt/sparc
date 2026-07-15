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
    """Limit native numerical-library threads within each worker."""
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)

def configure_global_settings():
    """Configure Windows worker limits and known scikit-learn warnings."""
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
    """Map a function over items with a thread or process executor."""
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
    """KMeans wrapper that suppresses the known Windows MKL warning."""
    
    @suppress_kmeans_warnings
    def __init__(self, n_clusters: int, random_state: int = 42, **kwargs):
        """Initialize the wrapped KMeans estimator."""
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
