"""Reader functions for molecular structure files."""

import gzip
from collections.abc import Iterator
from pathlib import Path

from rdkit import Chem


def read_mol2(path: Path) -> Iterator[Chem.Mol]:
    """Read molecule from a Mol2 file.

    Args:
        path: Path to the Mol2 file.

    Yields:
        RDKit Mol object.
    """

    mol = Chem.MolFromMol2File(str(path))
    yield mol


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


def read_mol2_gz(path: Path) -> Iterator[Chem.Mol]:
    """Read molecule from a gzipped Mol2 file.

    Args:
        path: Path to the gzipped Mol2 file (.mol2.gz).

    Yields:
        RDKit Mol object.
    """
    with gzip.open(path, "rt") as f:
        mol2_block = f.read()
    mol = Chem.MolFromMol2Block(mol2_block)
    if mol is not None:
        yield mol
