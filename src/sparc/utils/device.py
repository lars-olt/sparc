"""Device selection shared by RoMa, SAM, and application entry points."""

import os
import sys

DEVICES = ('auto', 'cpu', 'cuda', 'mps')


def prepare_accelerators():
    """Allow unsupported MPS operations on CPU; call before importing PyTorch."""
    if sys.platform == 'darwin':
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')


def resolve_device(requested='auto'):
    if requested not in DEVICES:
        raise ValueError(f'Unknown device: {requested}')
    prepare_accelerators()
    import torch

    mps = getattr(getattr(torch, 'backends', None), 'mps', None)
    available = {
        'cpu': True,
        'cuda': torch.cuda.is_available(),
        'mps': mps is not None and mps.is_available(),
    }
    if requested == 'auto':
        requested = next(name for name in ('cuda', 'mps', 'cpu') if available[name])
    if not available[requested]:
        raise RuntimeError(
            f'{requested.upper()} was requested but is unavailable. '
            'Check your PyTorch installation or use device=cpu.'
        )
    return torch.device(requested)
