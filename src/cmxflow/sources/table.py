"""Reader functions for tabular files containing SMILES."""

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem


def read_smi(path: Path) -> Iterator[Chem.Mol]:
    """Read molecules from a SMILES file.

    Expects space or tab separated format with no header.
    SMILES should be in the first column. Additional columns
    are attached as molecule properties.

    Args:
        path: Path to the SMILES file.

    Yields:
        RDKit Mol objects for each valid SMILES in the file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            smiles = parts[0]
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                if len(parts) > 1:
                    mol.SetProp("_Name", parts[1])
                for i, value in enumerate(parts[2:], start=2):
                    mol.SetProp(f"Column_{i}", value)
                yield mol


def read_csv(path: Path, chunksize: int = 1000) -> Iterator[Chem.Mol]:
    """Read molecules from a CSV file in chunks.

    Expects a column named "SMILES" containing the SMILES strings.
    All other columns are attached as molecule properties.

    Args:
        path: Path to the CSV file.
        chunksize: Number of rows to read per chunk. Defaults to 1000.

    Yields:
        RDKit Mol objects for each valid SMILES in the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If the "SMILES" column is not found.
    """
    for chunk in pd.read_csv(path, chunksize=chunksize):
        if "SMILES" not in chunk.columns:
            raise KeyError("CSV file must contain a 'SMILES' column")

        for _, row in chunk.iterrows():
            smiles = row["SMILES"]
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                for col in chunk.columns:
                    if col != "SMILES":
                        value = row[col]
                        if pd.notna(value):
                            mol.SetProp(col, str(value))
                yield mol


def read_parquet(path: Path, batch_size: int = 1000) -> Iterator[Chem.Mol]:
    """Read molecules from a Parquet file in batches.

    Expects a column named "SMILES" containing the SMILES strings.
    All other columns are attached as molecule properties.

    Args:
        path: Path to the Parquet file.
        batch_size: Number of rows to read per batch. Defaults to 1000.

    Yields:
        RDKit Mol objects for each valid SMILES in the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If the "SMILES" column is not found.
    """
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        chunk = batch.to_pandas()
        if "SMILES" not in chunk.columns:
            raise KeyError("Parquet file must contain a 'SMILES' column")

        for _, row in chunk.iterrows():
            smiles = row["SMILES"]
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                for col in chunk.columns:
                    if col != "SMILES":
                        value = row[col]
                        if pd.notna(value):
                            mol.SetProp(col, str(value))
                yield mol
