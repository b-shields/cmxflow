"""Base class for molecule operators."""

from abc import abstractmethod
from typing import Any

from rdkit import Chem

from cmxflow.block import Block
from cmxflow.cmxmol import Mol as CmxMol


class MoleculeBlock(Block):
    """Base class for blocks that operate on RDKit Mol objects.

    Provides input validation to ensure all inputs are valid RDKit Mol instances
    or Mol wrappers. Subclasses must implement the `forward` method to define
    the transformation.
    """

    def check_input(self, arg: Any) -> bool:
        """Validate that input is an RDKit Mol or Mol wrapper.

        Args:
            arg: Input item to validate.

        Returns:
            True if the input is a valid RDKit Mol or Mol wrapper,
            False otherwise.
        """
        return isinstance(arg, (Chem.Mol, CmxMol))

    @abstractmethod
    def forward(self, mol: Chem.Mol) -> Chem.Mol:
        """Transform a molecule.

        Args:
            mol: Input RDKit Mol object.

        Returns:
            Transformed RDKit Mol object.
        """
        ...
