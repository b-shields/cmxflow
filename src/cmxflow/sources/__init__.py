"""Source blocks and readers for molecular data files."""

from cmxflow.sources.molecule import read_mol2, read_sdf, read_sdf_gz
from cmxflow.sources.reader import MoleculeSourceBlock, read_molecules
from cmxflow.sources.table import read_csv, read_parquet, read_smi, read_smi_gz

__all__ = [
    "MoleculeSourceBlock",
    "read_csv",
    "read_mol2",
    "read_molecules",
    "read_parquet",
    "read_sdf",
    "read_sdf_gz",
    "read_smi",
    "read_smi_gz",
]
