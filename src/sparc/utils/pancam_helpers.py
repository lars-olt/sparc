"""Pancam data scanning and bandset construction."""

import os
import re
from functools import cache, partial
from pathlib import Path

import numpy as np
import pandas as pd

from marslab.bandset.pancam import PcamBandSet

np.seterr(divide="ignore", invalid="ignore")

_PCAM_FILENAME_RE = re.compile(
    r'^(?P<ROVER>\d)'
    r'P'
    r'(?P<SCLK>\d{9})'
    r'(?P<PRODUCT_TYPE>[A-Z]{3})'
    r'[A-Z0-9]{4}'
    r'(?P<SEQ_ID>[A-Z]\d{4})'
    r'(?P<FILTER>[LR]\d)'
    r'(?P<VERSION>[A-Z0-9])'
    r'.+$',
    re.IGNORECASE,
)

_INVALID_FILTERS = frozenset({'L0', 'L1', 'L8', 'R8'})


def parse_pcam_fn(filepath):
    """Parse a MER Pancam filename into its component metadata fields."""
    match = _PCAM_FILENAME_RE.match(Path(filepath).stem)
    if match is None:
        return None
    d = match.groupdict()
    p = Path(os.path.normpath(os.path.expanduser(str(filepath))))
    # Uppercase the extension - pdr resolves ^IMAGE pointers case-sensitively
    # and expects .IMG/.IMQ, not .img/.imq
    d['PATH'] = str(p.with_suffix(p.suffix.upper())).replace('\\', '/')
    d['SCLK'] = int(d['SCLK'])
    return d


def scan_pcam_files(root_dir, seq_id=None):
    """
    Scan a directory for Pancam IOF files and return a metadata DataFrame.

    Filters out non-IOF products and known-bad filter positions.
    Optionally narrows results to a specific sequence ID.
    """
    root  = Path(root_dir)
    files = [f for f in root.iterdir() if f.suffix.upper() in ('.IMG', '.IMQ')]

    products = [p for f in files if (p := parse_pcam_fn(f)) is not None]
    if not products:
        raise ValueError(f"No parsable Pancam files found in {root_dir}")

    df = (
        pd.DataFrame(products)
        .sort_values('SCLK')
        .reset_index(drop=True)
    )
    df = df.loc[df['PRODUCT_TYPE'].str.upper() == 'IOF'].copy()
    df = df.loc[~df['FILTER'].str.upper().isin(_INVALID_FILTERS)].reset_index(drop=True)

    if seq_id is not None:
        df = df.loc[df['SEQ_ID'].str.lower().str.contains(str(seq_id).lower())]

    if df.empty:
        raise ValueError(f"No IOF products matched in {root_dir}")

    df['FILTER'] = df['FILTER'].str.upper()
    df['BAND']   = df['FILTER']
    return df


def split_pcam_observations(products: pd.DataFrame) -> list:
    """
    Split a SCLK-sorted Pancam DataFrame into per-pointing sub-observations.

    Within a single SEQ_ID, a mosaic acquires the full filter set multiple times.
    Each time a filter name repeats, a new pointing started. Splitting on that
    boundary mirrors how ZCAM uses consecutive RSM pairs to separate pointings.

    Returns a list of DataFrames, one per pointing, each containing one full
    filter set sorted by SCLK.
    """
    groups     = []
    current    = []
    seen       = set()

    for _, row in products.sort_values('SCLK').iterrows():
        filt = row['FILTER']
        if filt in seen:
            if current:
                groups.append(pd.DataFrame(current).reset_index(drop=True))
            current = [row]
            seen    = {filt}
        else:
            current.append(row)
            seen.add(filt)

    if current:
        groups.append(pd.DataFrame(current).reset_index(drop=True))

    return groups


def get_pcam_bandset(image_path, roi_path=None, seq_id=None, observation_ix=0, load=True):
    """
    Build a PcamBandSet for the specified observation.

    observation_ix indexes across all per-pointing sub-observations found in
    image_path, properly splitting mosaic sequences that share a SEQ_ID.
    """
    products = scan_pcam_files(image_path, seq_id=seq_id)

    # split every SEQ_ID group into its per-pointing sub-observations
    observations = []
    for _, group in products.groupby('SEQ_ID'):
        observations.extend(split_pcam_observations(group))

    if observation_ix >= len(observations):
        raise ValueError(
            f"observation_ix={observation_ix} out of range - "
            f"found {len(observations)} pointing(s) in {image_path}"
        )

    observation = observations[observation_ix]
    bandset     = PcamBandSet(observation)

    if load:
        bandset.load("all")
        bandset.bulk_debayer("all")

    if roi_path is not None:
        bandset.rois = roi_path
        bandset.load_rois()
        bandset.count_rois()

    if hasattr(bandset, 'format_metadata'):
        bandset.format_metadata()

    return bandset