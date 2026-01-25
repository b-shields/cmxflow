"""Writer functions for tabular files containing SMILES."""

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem


def write_smi(mols: Iterator[Chem.Mol], path: Path) -> None:
    """Write molecules to a SMILES file.

    Writes space-separated SMILES and molecule name (if available).

    Args:
        mols: Iterator of RDKit Mol objects to write.
        path: Path to the output SMILES file.
    """
    with open(path, "w") as f:
        for mol in mols:
            smiles = Chem.MolToSmiles(mol)
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
            if name:
                f.write(f"{smiles} {name}\n")
            else:
                f.write(f"{smiles}\n")


def write_csv(mols: Iterator[Chem.Mol], path: Path, chunksize: int = 1000) -> None:
    """Write molecules to a CSV file in chunks.

    Writes SMILES column plus all molecule properties as additional columns.

    Args:
        mols: Iterator of RDKit Mol objects to write.
        path: Path to the output CSV file.
        chunksize: Number of molecules to accumulate before writing. Defaults to 1000.
    """
    first_chunk = True
    chunk: list[dict[str, str]] = []

    for mol in mols:
        row = {"SMILES": Chem.MolToSmiles(mol)}
        row.update(mol.GetPropsAsDict())
        # Convert all values to strings for consistency
        row = {k: str(v) for k, v in row.items()}
        chunk.append(row)

        if len(chunk) >= chunksize:
            df = pd.DataFrame(chunk)
            df.to_csv(path, mode="a", header=first_chunk, index=False)
            first_chunk = False
            chunk = []

    # Write remaining molecules
    if chunk:
        df = pd.DataFrame(chunk)
        df.to_csv(path, mode="a", header=first_chunk, index=False)


def write_parquet(mols: Iterator[Chem.Mol], path: Path, batch_size: int = 1000) -> None:
    """Write molecules to a Parquet file in batches.

    Writes SMILES column plus all molecule properties as additional columns.

    Args:
        mols: Iterator of RDKit Mol objects to write.
        path: Path to the output Parquet file.
        batch_size: Number of molecules to accumulate before writing. Defaults to 1000.
    """
    writer: pq.ParquetWriter | None = None
    batch: list[dict[str, str]] = []

    for mol in mols:
        row = {"SMILES": Chem.MolToSmiles(mol)}
        row.update(mol.GetPropsAsDict())
        row = {k: str(v) for k, v in row.items()}
        batch.append(row)

        if len(batch) >= batch_size:
            table = pa.Table.from_pylist(batch)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table)
            batch = []

    # Write remaining molecules
    if batch:
        table = pa.Table.from_pylist(batch)
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema)
        writer.write_table(table)

    if writer is not None:
        writer.close()
