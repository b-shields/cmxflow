"""Combined molecule reader and source block."""

from collections.abc import Iterator
from pathlib import Path

from rdkit import Chem

from cmxflow.block import SourceBlock
from cmxflow.cmxmol import Mol, wrap_mol
from cmxflow.sources.molecule import read_sdf, read_sdf_gz
from cmxflow.sources.table import read_csv, read_parquet, read_smi


def read_molecules(path: Path, wrap: bool = True) -> Iterator[Chem.Mol | Mol]:
    """Read molecules from a file, dispatching based on extension.

    Supported formats:
        - .sdf: SDF files
        - .sdf.gz: Gzipped SDF files
        - .smi: SMILES files (space/tab separated)
        - .csv: CSV files with SMILES column
        - .parquet: Parquet files with SMILES column

    Args:
        path: Path to the molecule file.
        wrap: If True (default), wrap molecules in Mol for property
            preservation through pickling.

    Yields:
        RDKit Mol objects (or Mol wrappers if wrap=True) for each valid
        molecule in the file.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    suffix = "".join(path.suffixes).lower()

    if suffix == ".sdf.gz":
        mols = read_sdf_gz(path)
    elif suffix == ".sdf":
        mols = read_sdf(path)
    elif suffix == ".smi":
        mols = read_smi(path)
    elif suffix == ".csv":
        mols = read_csv(path)
    elif suffix == ".parquet":
        mols = read_parquet(path)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")

    if wrap:
        for mol in mols:
            yield wrap_mol(mol)
    else:
        yield from mols


class MoleculeSourceBlock(SourceBlock):
    """Source block for reading molecules from various file formats.

    Supports SDF, gzipped SDF, SMILES, CSV, and Parquet files.
    File format is automatically detected based on extension.

    Args:
        wrap: If True (default), wrap molecules in Mol for property
            preservation through pickling.
    """

    def __init__(self, wrap: bool = True) -> None:
        """Initialize the molecule source block.

        Args:
            wrap: If True (default), wrap molecules in Mol for property
                preservation through pickling.
        """
        self._wrap = wrap
        super().__init__(reader=lambda p: read_molecules(p, wrap=self._wrap))
        self.name = "MoleculeSource"
