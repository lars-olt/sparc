"""
.sel file writer for merspect ROI export.

The format is a sequence of zlib-compressed IDL variable blocks. Instead of
reconstructing the IDL struct descriptors from scratch, we patch a known-good
blank template: swap out the LSELTEMP/RSELTEMP pixel masks, and leave
everything else from the original.

Blank templates live in sparc/utils/ as blank_mcz.sel and blank_pcam.sel.
"""

from __future__ import annotations

import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


_MAGIC             = b"SR"
_VERSION           = 6
_FILE_HEADER_SIZE  = 20
_BLOCK_HEADER_SIZE = 16

# Indices of the two mask blocks in the block list.
_LSEL_IDX = 2
_RSEL_IDX = 3

_INSTRUMENT_ALIASES = {
    "MCZ":             "mcz",
    "ZCAM":            "mcz",
    "MERSPECT_ZCAM":   "mcz",
    "PCAM":            "pcam",
    "PANCAM":          "pcam",
    "MERSPECT_PANCAM": "pcam",
}

_TEMPLATE_NAMES = {
    "mcz":  "blank_mcz.sel",
    "pcam": "blank_pcam.sel",
}

# merspect label conventions differ between instruments.
# MCZ: background=0, first ROI=4.
# Pancam: background=15, first ROI=0.
_MASK_DEFAULTS = {
    "mcz":  {"background": 0,  "first_id": 4},
    "pcam": {"background": 15, "first_id": 0},
}


@dataclass(frozen=True)
class _Block:
    compressed:   bytes
    decompressed: bytes


def export_sel(
    output_path: str,
    final_rois: np.ndarray,
    final_left_rois: np.ndarray,
    image_shape: Tuple[int, int],
    left_filenames: Sequence[str],
    right_filenames: Sequence[str],
    template_path: Optional[str] = None,
    instrument: Optional[str] = None,
    first_region_id: Optional[int] = None,
    background_value: Optional[int] = None,
) -> None:
    """Write a merspect-compatible .sel file by patching a blank template.

    Args:
        output_path:      Destination path.
        final_rois:       Right-camera ROIs (N, 4) as (x, y, w, h), full-sensor coords.
        final_left_rois:  Left-camera ROIs  (N, 4) as (x, y, w, h), full-sensor coords.
        image_shape:      Full sensor frame (height, width).
        left_filenames:   Reserved for future REGION_INFO patching (unused).
        right_filenames:  Reserved for future REGION_INFO patching (unused).
        template_path:    Path to a blank .sel template. If omitted, the packaged
                          template for 'instrument' is used.
        instrument:       "ZCAM"/"MCZ" or "PCAM"/"PANCAM".
        first_region_id:  Override the first ROI label (instrument default if None).
        background_value: Override the background label (instrument default if None).
    """
    inst_key         = _normalize_instrument(instrument)
    defaults         = _MASK_DEFAULTS[inst_key]
    first_region_id  = defaults["first_id"]   if first_region_id  is None else first_region_id
    background_value = defaults["background"] if background_value is None else background_value

    template   = _read_template(_resolve_template(template_path, inst_key))
    H, W       = _validated_shape(image_shape)
    left_rois  = _coerce_rois(final_left_rois)
    right_rois = _coerce_rois(final_rois)

    # Empty Pancam export: keep the compact scalar placeholder blocks from the
    # blank template rather than replacing them with full-size byte masks.
    if inst_key == "pcam" and len(left_rois) == 0 and len(right_rois) == 0:
        Path(output_path).write_bytes(_assemble(template))
        return

    blocks = list(template)
    blocks[_LSEL_IDX] = _Block(_make_mask("LSELTEMP", left_rois,  H, W, first_region_id, background_value), b"")
    blocks[_RSEL_IDX] = _Block(_make_mask("RSELTEMP", right_rois, H, W, first_region_id, background_value), b"")
    Path(output_path).write_bytes(_assemble(blocks))


def filenames_from_load_result(load_result: dict, n_rois: int) -> Tuple[List[str], List[str]]:
    """Pull left/right product filename stems out of a SPARC load_result."""
    scene_id = load_result.get("id", "UNKNOWN")
    bandset  = load_result.get("bandset")
    left     = _stem_from_bandset(bandset, "L", scene_id)
    right    = _stem_from_bandset(bandset, "R", scene_id)
    return [left] * n_rois, [right] * n_rois


def get_default_template_path(instrument: Optional[str] = None) -> Path:
    """Return the path to the packaged blank .sel template for this instrument."""
    inst_key = _normalize_instrument(instrument)
    name     = _TEMPLATE_NAMES[inst_key]

    # Try importlib.resources first (works when installed as a package).
    resource = _resource_path(inst_key)
    if resource is not None:
        return resource

    # Fall back to paths relative to this file.
    here       = Path(__file__).resolve().parent
    candidates = [
        here / name,
        here.parent / "resources" / name,
        here.parent / name,
        Path.cwd() / name,
        *(([here / "blank.sel", here / "no_rois.sel"]) if inst_key == "mcz" else []),
    ]
    for p in candidates:
        if p.is_file():
            return p

    raise FileNotFoundError(
        f"No blank .sel template found for {inst_key!r}. "
        f"Expected {name!r} — pass template_path= to override."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stem_from_bandset(bandset, prefix: str, fallback: str) -> str:
    try:
        meta    = bandset.metadata
        matches = meta.loc[meta["BAND"].str.startswith(prefix), "PATH"]
        if not matches.empty:
            return Path(matches.iloc[0]).stem
    except Exception:
        pass
    return fallback


def _normalize_instrument(instrument: Optional[str]) -> str:
    if instrument is None:
        return "mcz"
    key = str(instrument).strip().upper()
    if key not in _INSTRUMENT_ALIASES:
        raise ValueError(
            f"Unknown instrument {instrument!r}. "
            f"Supported: {', '.join(sorted(_INSTRUMENT_ALIASES))}"
        )
    return _INSTRUMENT_ALIASES[key]


def _resolve_template(template_path: Optional[str], inst_key: str) -> Path:
    if template_path:
        p = Path(template_path)
        if not p.is_file():
            raise FileNotFoundError(f"template_path does not exist: {p}")
        return p

    env = os.environ.get("SPARC_SEL_TEMPLATE")
    if env:
        p = Path(env)
        if not p.is_file():
            raise FileNotFoundError(f"SPARC_SEL_TEMPLATE points to missing file: {p}")
        return p

    return get_default_template_path(instrument=inst_key)


def _resource_path(inst_key: str) -> Optional[Path]:
    """Locate a packaged template via importlib.resources."""
    try:
        from importlib.resources import files
    except ImportError:
        return None

    package = (__package__ or "").split(".")[0]
    if not package:
        return None

    name = _TEMPLATE_NAMES[inst_key]
    for pkg in (f"{package}.resources", package):
        try:
            ref = files(pkg).joinpath(name)
            if ref.is_file():
                return Path(str(ref))
        except Exception:
            continue
    return None


def _read_template(path: Path) -> List[_Block]:
    """Parse a .sel file into its payload blocks."""
    raw = path.read_bytes()
    if raw[:2] != _MAGIC:
        raise ValueError(f"{path} is not a .sel file (missing SR magic).")

    version = struct.unpack(">H", raw[2:4])[0]
    if version != _VERSION:
        raise ValueError(f"Unsupported .sel version {version}; expected {_VERSION}.")

    n_blocks    = struct.unpack(">I", raw[4:8])[0] + 1
    payload_end = struct.unpack(">I", raw[8:12])[0]  # absolute end of block 0
    offset      = _FILE_HEADER_SIZE
    blocks: List[_Block] = []

    for i in range(n_blocks):
        if payload_end > len(raw):
            raise ValueError(f"Block {i} end offset {payload_end} exceeds file size.")
        compressed = raw[offset:payload_end]
        try:
            decompressed = zlib.decompress(compressed)
        except zlib.error as e:
            raise ValueError(f"Block {i} zlib decompression failed.") from e

        blocks.append(_Block(compressed=compressed, decompressed=decompressed))

        hdr_start = payload_end
        if hdr_start + _BLOCK_HEADER_SIZE > len(raw):
            raise ValueError("File ended while reading block header.")

        _, next_end = struct.unpack(">II", raw[hdr_start:hdr_start + 8])
        offset      = hdr_start + _BLOCK_HEADER_SIZE
        payload_end = next_end

    return blocks


def _validated_shape(image_shape: Tuple[int, int]) -> Tuple[int, int]:
    if len(image_shape) != 2:
        raise ValueError("image_shape must be (height, width).")
    H, W = int(image_shape[0]), int(image_shape[1])
    if H <= 0 or W <= 0:
        raise ValueError("image_shape values must be positive.")
    return H, W


def _coerce_rois(rois: np.ndarray) -> np.ndarray:
    rois = np.asarray(rois)
    if rois.size == 0:
        return np.empty((0, 4), dtype=np.int32)
    if rois.ndim != 2 or rois.shape[1] != 4:
        raise ValueError("ROIs must be shape (N, 4).")
    return rois.astype(np.int32, copy=False)


def _build_mask(
    rois: np.ndarray,
    H: int,
    W: int,
    first_id: int,
    background: int,
) -> np.ndarray:
    """Paint ROIs into a uint8 label mask in IDL array order (y=0 at bottom)."""
    mask = np.full((H, W), background, dtype=np.uint8)
    for i, roi in enumerate(rois):
        region_id = first_id + i
        if region_id > 255:
            raise ValueError("Too many ROIs — region ID exceeds uint8 range.")
        x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
        if w <= 0 or h <= 0:
            continue
        # Flip vertically to match merspect origin (y=0 at top of image)
        x0, x1 = max(0, x), min(W, x + w)
        y0, y1 = max(0, H - y - h), min(H, H - y)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = region_id
    return mask


def _make_mask(name: str, rois: np.ndarray, H: int, W: int, first_id: int, background: int) -> bytes:
    """Build a compressed LSELTEMP/RSELTEMP payload."""
    mask = _build_mask(rois, H, W, first_id, background)
    n    = H * W
    hdr  = struct.pack(">I", len(name)) + name.encode("ascii")
    hdr += struct.pack(">I", 1)   # IDL type BYTE
    hdr += struct.pack(">I", 20)
    hdr += struct.pack(">I", 8)
    hdr += struct.pack(">I", 1)   # n_dims
    hdr += struct.pack(">I", n)
    hdr += struct.pack(">I", n)
    hdr += struct.pack(">I", 2)   # n_dim_fields
    hdr += b"\x00" * 8
    hdr += struct.pack(">I", 8)
    hdr += struct.pack(">I", W)
    hdr += struct.pack(">I", H)
    hdr += struct.pack(">I", 1) * 6
    hdr += struct.pack(">I", 7)   # static IDL struct constant
    hdr += struct.pack(">I", n)
    return zlib.compress(hdr + mask.tobytes(), level=6)


def _assemble(blocks: Sequence[_Block]) -> bytes:
    """Serialize blocks back into a complete .sel binary."""
    if not blocks:
        raise ValueError(".sel file must have at least one block.")

    out  = bytearray(_MAGIC)
    out += struct.pack(">H", _VERSION)
    out += struct.pack(">I", len(blocks) - 1)
    out += struct.pack(">I", _FILE_HEADER_SIZE + len(blocks[0].compressed))
    out += b"\x00" * 8

    cursor = _FILE_HEADER_SIZE
    for i, block in enumerate(blocks):
        out    += block.compressed
        cursor += len(block.compressed)

        is_last  = i == len(blocks) - 1
        flags    = 0x06 if is_last else (0x0E if i == 0 else 0x02)
        next_end = 0 if is_last else cursor + _BLOCK_HEADER_SIZE + len(blocks[i + 1].compressed)
        out     += struct.pack(">II", flags, next_end)
        out     += b"\x00" * 8
        cursor  += _BLOCK_HEADER_SIZE

    return bytes(out)