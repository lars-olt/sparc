"""Memory cleanup helpers for optional accelerator workloads."""

import gc


def release_accelerator_memory() -> None:
    """Release unused CUDA/MPS cache blocks without hiding an inference error."""
    gc.collect()

    try:
        import torch
    except ImportError:
        return

    try:
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except (AttributeError, RuntimeError):
        pass
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except (AttributeError, RuntimeError):
        pass
