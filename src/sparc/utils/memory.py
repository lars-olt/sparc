"""Memory cleanup helpers for optional accelerator workloads."""

import gc


def release_cuda_memory() -> None:
    """Release unreachable objects and unused PyTorch CUDA cache blocks."""
    gc.collect()

    try:
        import torch
    except ImportError:
        return

    try:
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except (AttributeError, RuntimeError):
        # Cleanup must never hide the pipeline result or its original error.
        return
