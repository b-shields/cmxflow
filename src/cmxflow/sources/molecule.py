"""Reader functions for molecular structure files."""

import gzip
from collections.abc import Iterator
from pathlib import Path

from rdkit import Chem


def read_sdf(path: Path) -> Iterator[Chem.Mol]:
    """Read molecules from an SDF file.

    Args:
        path: Path to the SDF file.

    Yields:
        RDKit Mol objects for each valid molecule in the file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    supplier = Chem.SDMolSupplier(str(path))
    for mol in supplier:
        if mol is not None:
            yield mol


def read_sdf_gz(path: Path) -> Iterator[Chem.Mol]:
    """Read molecules from a gzipped SDF file.

    Args:
        path: Path to the gzipped SDF file (.sdf.gz).

    Yields:
        RDKit Mol objects for each valid molecule in the file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    with gzip.open(path, "rb") as f:
        supplier = Chem.ForwardSDMolSupplier(f)
        for mol in supplier:
            if mol is not None:
                yield mol
