"""Writer functions for molecular structure files."""

import gzip
from collections.abc import Iterator
from pathlib import Path

from rdkit import Chem


def split_conformers(mols: Iterator[Chem.Mol]) -> Iterator[Chem.Mol]:
    """Split molecules with multiple conformers into separate molecules.

    Each conformer becomes a separate molecule with all properties copied
    from the original molecule.

    Args:
        mols: Iterator of RDKit Mol objects, potentially with multiple conformers.

    Yields:
        RDKit Mol objects, each with a single conformer.
    """
    for mol in mols:
        num_confs = mol.GetNumConformers()
        if num_confs <= 1:
            yield mol
        else:
            props = mol.GetPropsAsDict()
            for conf in mol.GetConformers():
                new_mol = Chem.Mol(mol)
                new_mol.RemoveAllConformers()
                new_mol.AddConformer(Chem.Conformer(conf), assignId=True)
                for key, value in props.items():
                    if isinstance(value, int):
                        new_mol.SetIntProp(key, value)
                    elif isinstance(value, float):
                        new_mol.SetDoubleProp(key, value)
                    elif isinstance(value, bool):
                        new_mol.SetBoolProp(key, value)
                    else:
                        new_mol.SetProp(key, str(value))
                yield new_mol


def write_sdf(
    mols: Iterator[Chem.Mol], path: Path, *, split_confs: bool = True
) -> None:
    """Write molecules to an SDF file.

    Args:
        mols: Iterator of RDKit Mol objects to write.
        path: Path to the output SDF file.
        split_confs: If True, split molecules with multiple conformers into
            separate molecules with properties copied.
    """
    if split_confs:
        mols = split_conformers(mols)
    writer = Chem.SDWriter(str(path))
    try:
        for mol in mols:
            writer.write(mol)
    finally:
        writer.close()


def write_sdf_gz(
    mols: Iterator[Chem.Mol], path: Path, *, split_confs: bool = True
) -> None:
    """Write molecules to a gzipped SDF file.

    Args:
        mols: Iterator of RDKit Mol objects to write.
        path: Path to the output gzipped SDF file (.sdf.gz).
        split_confs: If True, split molecules with multiple conformers into
            separate molecules with properties copied.
    """
    if split_confs:
        mols = split_conformers(mols)
    with gzip.open(path, "wt") as f:
        writer = Chem.SDWriter(f)
        try:
            for mol in mols:
                writer.write(mol)
        finally:
            writer.close()
