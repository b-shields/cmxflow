"""Combined molecule writer and sink block."""

from collections.abc import Iterator
from pathlib import Path

from rdkit import Chem

from cmxflow.block import SinkBlock
from cmxflow.sinks.molecule import write_sdf, write_sdf_gz
from cmxflow.sinks.table import write_csv, write_parquet, write_smi


def write_molecules(mols: Iterator[Chem.Mol], path: Path) -> None:
    """Write molecules to a file, dispatching based on extension.

    Args:
        mols: Iterator of RDKit Mol objects to write.
        path: Path to the output file.

    Raises:
        ValueError: If the file extension is not supported.
    """
    suffix = "".join(path.suffixes).lower()

    if suffix == ".sdf.gz":
        write_sdf_gz(mols, path)
    elif suffix == ".sdf":
        write_sdf(mols, path)
    elif suffix == ".smi":
        write_smi(mols, path)
    elif suffix == ".csv":
        write_csv(mols, path)
    elif suffix == ".parquet":
        write_parquet(mols, path)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")


class MoleculeSinkBlock(SinkBlock):
    """Sink block for writing molecules to various file formats.

    Supports .sdf, .sdf.gz, .mol2, .smi, .smi.gz, .csv, and .parquet
    files. File format is automatically detected based on extension.

    Example:
        ```python
        workflow.add(
            MoleculeSourceBlock(),
            MoleculeSinkBlock(),
        )
        ```
    """

    def __init__(self) -> None:
        """Initialize the molecule sink block."""
        super().__init__(writer=write_molecules)
        self.name = "MoleculeSink"
