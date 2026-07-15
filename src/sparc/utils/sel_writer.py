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

# Map Pancam frame sizes to templates containing the corresponding MERSpect POS blocks.
# Unknown sizes use blank_pcam.sel.
_PCAM_SIZED_TEMPLATES = {
    (300, 300): "blank_pcam_300x300.sel",
}

# MERSpect label conventions differ between instruments.
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
    label_ids: Optional[List[int]] = None,
) -> None:
    """Write a merspect-compatible .sel file by patching a blank template.

    ROI masks are painted in local scene coordinates. Everything that tells
    MERSpect where the scene frame sits on the full sensor - the POS blocks -
    comes from the template, so a scene-size-matched template must be supplied
    for subframe scenes (see _resolve_template, which picks one by image_shape).

    Args:
        output_path:      Destination path.
        final_rois:       Right-camera ROIs (N, 4) as (x, y, w, h), scene coords.
        final_left_rois:  Left-camera ROIs  (N, 4) as (x, y, w, h), scene coords.
        image_shape:      Scene frame (height, width) - the size the mask is painted
                          at, and the size the template must match.
        left_filenames:   Reserved for future REGION_INFO patching (unused).
        right_filenames:  Reserved for future REGION_INFO patching (unused).
        template_path:    Path to a blank .sel template. If omitted, the packaged
                          template for 'instrument' and image_shape is used.
        instrument:       "ZCAM"/"MCZ" or "PCAM"/"PANCAM".
        first_region_id:  Override the first ROI label (instrument default if None).
                          Ignored when label_ids is provided.
        background_value: Override the background label (instrument default if None).
        label_ids:        Explicit per-ROI MERSpect label indices. When provided,
                          first_region_id is ignored and each ROI is painted with
                          its corresponding label_ids value.
    """
    inst_key         = _normalize_instrument(instrument)
    defaults         = _MASK_DEFAULTS[inst_key]
    first_region_id  = defaults["first_id"]   if first_region_id  is None else first_region_id
    background_value = defaults["background"] if background_value is None else background_value

    H, W       = _validated_shape(image_shape)
    template   = _read_template(_resolve_template(template_path, inst_key, (H, W)))
    left_rois  = _coerce_rois(final_left_rois)
    right_rois = _coerce_rois(final_rois)

    # Empty Pancam export: keep the compact scalar placeholder blocks from the
    # blank template rather than replacing them with full-size byte masks.
    if inst_key == "pcam" and len(left_rois) == 0 and len(right_rois) == 0:
        Path(output_path).write_bytes(_assemble(template))
        return

    blocks = list(template)
    blocks[_LSEL_IDX] = _Block(_make_mask("LSELTEMP", left_rois,  H, W, first_region_id, background_value, label_ids), b"")
    blocks[_RSEL_IDX] = _Block(_make_mask("RSELTEMP", right_rois, H, W, first_region_id, background_value, label_ids), b"")
    Path(output_path).write_bytes(_assemble(blocks))


def read_sel(
    sel_path: str,
    instrument: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Read ROI bounding boxes and their MERSpect label indices from a .sel file.

    Returns:
        Tuple of (right_rois, left_rois, label_ids) where right_rois and left_rois
        are (N, 4) int32 arrays of (x, y, w, h) and label_ids are the MERSpect
        label index for each ROI.
    """
    inst_key   = _normalize_instrument(instrument)
    background = _MASK_DEFAULTS[inst_key]["background"]
    blocks     = _read_template(Path(sel_path))

    right_rois, label_ids = _rois_from_block(blocks[_RSEL_IDX].decompressed, background)
    left_rois,  _         = _rois_from_block(blocks[_LSEL_IDX].decompressed, background)

    n = max(len(right_rois), len(left_rois))
    def _pad(rois):
        if len(rois) == n:
            return rois
        pad = np.zeros((n - len(rois), 4), dtype=np.int32)
        return np.vstack([rois, pad]) if len(rois) else pad

    return _pad(right_rois), _pad(left_rois), label_ids


def filenames_from_load_result(load_result: dict, n_rois: int) -> Tuple[List[str], List[str]]:
    """Pull left/right product filename stems out of a SPARC load_result."""
    scene_id = load_result.get("id", "UNKNOWN")
    bandset  = load_result.get("bandset")
    left     = _stem_from_bandset(bandset, "L", scene_id)
    right    = _stem_from_bandset(bandset, "R", scene_id)
    return [left] * n_rois, [right] * n_rois


def _template_name(inst_key: str, image_shape: Optional[Tuple[int, int]]) -> str:
    """Blank template filename for an instrument and optional scene size.

    Pancam subframe sizes each have their own blank so the frame's sensor
    placement is correct; everything else falls back to the instrument default.
    """
    if inst_key == "pcam" and image_shape is not None:
        sized = _PCAM_SIZED_TEMPLATES.get(tuple(image_shape))
        if sized is not None:
            return sized
    return _TEMPLATE_NAMES[inst_key]


def get_default_template_path(instrument: Optional[str] = None,
                              image_shape: Optional[Tuple[int, int]] = None) -> Path:
    """Return the path to the packaged blank .sel template for this instrument and size."""
    inst_key = _normalize_instrument(instrument)
    name     = _template_name(inst_key, image_shape)

    resource = _resource_path(name)
    if resource is not None:
        return resource

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

    # Fall back to the instrument template when no size-specific template exists.
    if name != _TEMPLATE_NAMES[inst_key]:
        return get_default_template_path(instrument=inst_key)

    raise FileNotFoundError(
        f"No blank .sel template found for {inst_key!r}. "
        f"Expected {name!r} - pass template_path= to override."
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


def _resolve_template(template_path: Optional[str], inst_key: str,
                      image_shape: Optional[Tuple[int, int]] = None) -> Path:
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

    return get_default_template_path(instrument=inst_key, image_shape=image_shape)


def _resource_path(name: str) -> Optional[Path]:
    """Locate a packaged template by filename via importlib.resources."""
    try:
        from importlib.resources import files
    except ImportError:
        return None

    package = (__package__ or "").split(".")[0]
    if not package:
        return None

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
    payload_end = struct.unpack(">I", raw[8:12])[0]
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
    if np.issubdtype(rois.dtype, np.integer):
        return rois.astype(np.int32, copy=False)
    # truncation drags edges up to a pixel toward zero
    return np.rint(rois).astype(np.int32)


def _build_mask(
    rois: np.ndarray,
    H: int,
    W: int,
    first_id: int,
    background: int,
    label_ids: Optional[List[int]] = None,
) -> np.ndarray:
    """Paint ROIs into a uint8 label mask in IDL array order (y=0 at bottom)."""
    mask = np.full((H, W), background, dtype=np.uint8)
    for i, roi in enumerate(rois):
        region_id = label_ids[i] if label_ids is not None else first_id + i
        if region_id > 255:
            raise ValueError("Too many ROIs - region ID exceeds uint8 range.")
        x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
        if w <= 0 or h <= 0:
            continue
        # Flip vertically to match merspect origin (y=0 at top of image).
        x0, x1 = max(0, x), min(W, x + w)
        y0, y1 = max(0, H - y - h), min(H, H - y)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = region_id
    return mask


def _make_mask(
    name: str,
    rois: np.ndarray,
    H: int,
    W: int,
    first_id: int,
    background: int,
    label_ids: Optional[List[int]] = None,
) -> bytes:
    """Build a compressed LSELTEMP/RSELTEMP payload."""
    mask = _build_mask(rois, H, W, first_id, background, label_ids)
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


def _parse_mask_header(payload: bytes) -> Tuple[int, int, int]:
    """Extract (W, H, header_size) from a decompressed LSELTEMP/RSELTEMP payload."""
    if len(payload) < 8:
        raise ValueError("Mask payload too short.")

    name_len = struct.unpack(">I", payload[0:4])[0]
    _UINT    = 4
    _FIELD   = _UINT

    pre_dims  = _UINT + name_len + 7 * _FIELD  # name + type + 2 unknowns + n_dims + n + n + n_dim_fields
    post_dims = _FIELD + 2 * _UINT              # padding(8) + unknown

    base     = pre_dims + post_dims
    W        = struct.unpack(">I", payload[base:         base + _UINT])[0]
    H        = struct.unpack(">I", payload[base + _UINT: base + 2 * _UINT])[0]
    hdr_size = base + 2 * _UINT + 8 * _FIELD   # W + H + 6 static fields + 2 trailing constants

    return W, H, hdr_size


def _rois_from_block(payload: bytes, background: int) -> Tuple[np.ndarray, List[int]]:
    """Convert a decompressed mask payload into (N, 4) bounding boxes and their label values."""
    from scipy.ndimage import find_objects

    empty = np.empty((0, 4), dtype=np.int32), []

    if not payload:
        return empty

    try:
        W, H, hdr_size = _parse_mask_header(payload)
    except Exception:
        return empty

    mask   = np.frombuffer(payload[hdr_size:hdr_size + H * W], dtype=np.uint8).reshape(H, W)
    labels = sorted(v for v in np.unique(mask) if v != background)
    if not labels:
        return empty

    remapped = np.zeros_like(mask)
    for i, label in enumerate(labels, start=1):
        remapped[mask == label] = i

    slices = find_objects(remapped)
    rois        = []
    label_values = []
    for i, sl in enumerate(slices):
        if sl is None:
            continue
        y0_flipped = sl[0].start
        y1_flipped = sl[0].stop - 1
        x0 = sl[1].start
        y0 = H - y1_flipped - 1
        y1 = H - y0_flipped - 1
        rois.append((x0, y0, sl[1].stop - x0, y1 - y0 + 1))
        label_values.append(labels[i])

    if not rois:
        return empty
    return np.array(rois, dtype=np.int32), label_values


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
