"""Combined molecule reader and source block."""

from collections.abc import Iterator
from pathlib import Path

from rdkit import Chem

from cmxflow.block import SourceBlock
from cmxflow.sources.molecule import read_sdf, read_sdf_gz
from cmxflow.sources.table import read_csv, read_parquet, read_smi


def read_molecules(path: Path) -> Iterator[Chem.Mol]:
    """Read molecules from a file, dispatching based on extension.

    Supported formats:
        - .sdf: SDF files
        - .sdf.gz: Gzipped SDF files
        - .smi: SMILES files (space/tab separated)
        - .csv: CSV files with SMILES column
        - .parquet: Parquet files with SMILES column

    Args:
        path: Path to the molecule file.

    Yields:
        RDKit Mol objects for each valid molecule in the file.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    suffix = "".join(path.suffixes).lower()

    if suffix == ".sdf.gz":
        yield from read_sdf_gz(path)
    elif suffix == ".sdf":
        yield from read_sdf(path)
    elif suffix == ".smi":
        yield from read_smi(path)
    elif suffix == ".csv":
        yield from read_csv(path)
    elif suffix == ".parquet":
        yield from read_parquet(path)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")


class MoleculeSourceBlock(SourceBlock):
    """Source block for reading molecules from various file formats.

    Supports SDF, gzipped SDF, SMILES, CSV, and Parquet files.
    File format is automatically detected based on extension.
    """

    def __init__(self) -> None:
        """Initialize the molecule source block."""
        super().__init__(reader=read_molecules)
        self.name = "MoleculeSource"
