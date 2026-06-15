"""Reader functions for molecular structure files."""

import gzip
import logging
from collections.abc import Iterator
from pathlib import Path

from rdkit import Chem

logger = logging.getLogger(__name__)


def read_mol2(path: Path) -> Iterator[Chem.Mol]:
    """Read molecules from a multi-molecule Mol2 file.

    Args:
        path: Path to the Mol2 file.

    Yields:
        RDKit Mol objects for each valid molecule in the file.
    """
    block: list[str] = []
    idx = 0
    with open(path) as fh:
        for line in fh:
            if line.startswith("@<TRIPOS>MOLECULE") and block:
                mol = Chem.MolFromMol2Block("".join(block), removeHs=False)
                if mol is None:
                    logger.warning("Skipping unreadable record %d in %s", idx, path)
                else:
                    yield mol
                idx += 1
                block = []
            block.append(line)
    if block:
        mol = Chem.MolFromMol2Block("".join(block), removeHs=False)
        if mol is None:
            logger.warning("Skipping unreadable record %d in %s", idx, path)
        else:
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
    for idx, mol in enumerate(supplier):
        if mol is None:
            logger.warning("Skipping unreadable record %d in %s", idx, path)
            continue
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
        for idx, mol in enumerate(supplier):
            if mol is None:
                logger.warning("Skipping unreadable record %d in %s", idx, path)
                continue
            yield mol
