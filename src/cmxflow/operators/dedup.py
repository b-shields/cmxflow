"""Molecule de-duplication block."""

import logging
from typing import Any

from rdkit import Chem

from cmxflow.operators.base import MoleculeBlock

logger = logging.getLogger(__name__)


class MoleculeDeduplicateBlock(MoleculeBlock):
    """Remove duplicate molecules from a stream based on canonical SMILES.

    Keeps the first occurrence and discards subsequent duplicates.
    Uses RDKit canonical SMILES as the deduplication key.

    This block cannot be parallelized because it relies on shared
    mutable state (the set of seen SMILES).

    Attributes:
        _seen: Set of canonical SMILES strings already encountered.
    """

    def __init__(self, **kwargs: str) -> None:
        """Initialize the de-duplication block."""
        super().__init__(name="MoleculeDeduplicate")
        self._seen: set[str] = set()
        self.set_inputs(**kwargs)

    def _forward(self, mol: Chem.Mol) -> Chem.Mol | None:
        """Pass through first-seen molecules, drop duplicates.

        Args:
            mol: Input RDKit Mol object.

        Returns:
            The molecule if first seen, or None if duplicate.
        """
        smiles = Chem.MolToSmiles(mol)
        if smiles in self._seen:
            logger.info("Removing duplicate: %s", smiles)
            return None
        self._seen.add(smiles)
        return mol

    def check_output(self, arg: Any) -> bool:
        """Validate that output is a valid molecule.

        Args:
            arg: Output to validate.

        Returns:
            True if the output is a valid molecule, False otherwise.
        """
        return isinstance(arg, Chem.Mol)

    def reset_cache(self) -> None:
        """Clear the seen set for a new optimization iteration."""
        self._seen.clear()
