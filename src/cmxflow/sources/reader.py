"""Combined molecule reader and source block."""

from collections.abc import Iterator
from pathlib import Path

from rdkit import Chem

from cmxflow.block import SourceBlock
from cmxflow.cmxmol import Mol, wrap_mol
from cmxflow.sources.molecule import read_mol2, read_sdf, read_sdf_gz
from cmxflow.sources.table import read_csv, read_parquet, read_smi, read_smi_gz


def _parse_suffix(path: Path) -> tuple[str, bool]:
    """Parse file suffix and detect gzip compression.

    Args:
        path: Path object to extract and parse suffixes from.

    Returns:
        Tuple of (base_suffix, is_gzipped) where base_suffix is the
        file extension without .gz, and is_gzipped indicates if the
        file is gzip compressed.
        Example: Path("file.sdf.gz") -> (".sdf", True)
    """
    suffixes = [s.lower() for s in path.suffixes]
    if suffixes and suffixes[-1] == ".gz":
        return "".join(suffixes[:-1]), True
    return "".join(suffixes), False


def read_molecules(path: Path, wrap: bool = True) -> Iterator[Chem.Mol | Mol]:
    """Read molecules from a file, dispatching based on extension.

    Supported formats:
        - .sdf: SDF files
        - .sdf.gz: Gzipped SDF files
        - .mol2: Mol2 files
        - .smi: SMILES files (space/tab separated)
        - .smi.gz: Gzipped SMILES files
        - .csv: CSV files with SMILES column
        - .csv.gz: Gzipped CSV files with SMILES column
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
    base_suffix, is_gzipped = _parse_suffix(path)

    if base_suffix == ".sdf":
        mols = read_sdf_gz(path) if is_gzipped else read_sdf(path)
    elif base_suffix == ".mol2":
        if is_gzipped:
            raise ValueError("Gzipped mol2 not supported.")
        mols = read_mol2(path)
    elif base_suffix == ".smi":
        mols = read_smi_gz(path) if is_gzipped else read_smi(path)
    elif base_suffix == ".csv":
        mols = read_csv(path)  # pandas handles gzip automatically
    elif base_suffix == ".parquet":
        if is_gzipped:
            raise ValueError(
                "Gzipped parquet not supported; parquet has internal compression"
            )
        mols = read_parquet(path)
    else:
        raise ValueError(f"Unsupported file extension: {base_suffix}")

    if wrap:
        for mol in mols:
            yield wrap_mol(mol)
    else:
        yield from mols


class MoleculeSourceBlock(SourceBlock):
    """Source block for reading molecules from various file formats.

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

    def __init__(self, wrap: bool = True) -> None:
        """Initialize the molecule source block.

        Args:
            wrap: If True (default), wrap molecules in Mol for property
                preservation through pickling.
        """
        self._wrap = wrap
        super().__init__(reader=lambda p: read_molecules(p, wrap=self._wrap))
        self.name = "MoleculeSource"

    def __getstate__(self) -> dict:
        """Get state for pickling, excluding the lambda."""
        state = self.__dict__.copy()
        state.pop("reader", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state from pickle, recreating the lambda."""
        self.__dict__.update(state)
        self.reader = lambda p: read_molecules(p, wrap=self._wrap)
