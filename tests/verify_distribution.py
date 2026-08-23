"""Validate that SPARC source and wheel distributions contain their resources."""

from pathlib import Path
import sys
import tarfile
from zipfile import ZipFile


REQUIRED_RESOURCES = (
    'blank_mcz.sel',
    'blank_pcam.sel',
    'blank_pcam_300x300.sel',
    'pcam_observation_clusters.csv.gz',
)


def _single_match(directory, pattern):
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise AssertionError(f'expected one {pattern}, found {matches}')
    return matches[0]


def verify(dist_directory):
    wheel = _single_match(dist_directory, '*.whl')
    source = _single_match(dist_directory, '*.tar.gz')

    with ZipFile(wheel) as archive:
        wheel_names = set(archive.namelist())
        metadata_name = next(
            name for name in wheel_names if name.endswith('.dist-info/METADATA')
        )
        metadata = archive.read(metadata_name).decode('utf-8')

    with tarfile.open(source, 'r:gz') as archive:
        source_names = set(archive.getnames())

    for resource in REQUIRED_RESOURCES:
        wheel_path = f'sparc/resources/{resource}'
        if wheel_path not in wheel_names:
            raise AssertionError(f'wheel is missing {wheel_path}')
        if not any(name.endswith(f'/src/{wheel_path}') for name in source_names):
            raise AssertionError(f'source distribution is missing {wheel_path}')

    if 'Provides-Extra: algorithm' not in metadata:
        raise AssertionError('wheel metadata is missing the algorithm extra')


def main():
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else 'dist')
    verify(directory)
    print('SPARC distribution audit passed')


if __name__ == '__main__':
    main()
