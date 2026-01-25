"""Writer functions for molecular structure files."""

import gzip
from collections.abc import Iterator
from pathlib import Path

from rdkit import Chem


def write_sdf(mols: Iterator[Chem.Mol], path: Path) -> None:
    """Write molecules to an SDF file.

    Args:
        mols: Iterator of RDKit Mol objects to write.
        path: Path to the output SDF file.
    """
    writer = Chem.SDWriter(str(path))
    try:
        for mol in mols:
            writer.write(mol)
    finally:
        writer.close()


def write_sdf_gz(mols: Iterator[Chem.Mol], path: Path) -> None:
    """Write molecules to a gzipped SDF file.

    Args:
        mols: Iterator of RDKit Mol objects to write.
        path: Path to the output gzipped SDF file (.sdf.gz).
    """
    with gzip.open(path, "wt") as f:
        writer = Chem.SDWriter(f)
        try:
            for mol in mols:
                writer.write(mol)
        finally:
            writer.close()
