"""Base class for molecule operators."""

from abc import abstractmethod
from typing import Any

from rdkit import Chem

from cmxflow.block import Block
from cmxflow.cmxmol import Mol as CmxMol


class MoleculeBlock(Block):
    """Base class for blocks that operate on RDKit Mol objects.

    Provides input validation and automatic property preservation through
    parallelization. Subclasses must implement `_forward` to define the
    transformation.

    Property preservation works by:
    1. Capturing input properties before calling _forward()
    2. Wrapping output in CmxMol (which survives pickling)
    3. Restoring input properties to output molecule
    """

    def check_input(self, arg: Any) -> bool:
        """Validate that input is an RDKit Mol or Mol wrapper.

        Args:
            arg: Input item to validate.

        Returns:
            True if the input is a valid RDKit Mol or Mol wrapper,
            False otherwise.
        """
        is_mol = isinstance(arg, (Chem.Mol, CmxMol))
        if isinstance(arg, CmxMol):
            arg.restore_properties()
        return is_mol

    @abstractmethod
    def _forward(self, mol: Chem.Mol) -> Chem.Mol | None:
        """Transform a molecule. Subclasses implement this.

        Args:
            mol: Input RDKit Mol object.

        Returns:
            Transformed RDKit Mol object, or None to skip.
        """
        ...

    def forward(self, mol: Chem.Mol) -> CmxMol | None:
        """Transform a molecule with automatic property preservation.

        Wraps _forward() to capture input properties and restore them
        to the output molecule. This ensures properties survive
        parallelization via pickling.

        Args:
            mol: Input RDKit Mol object (may be CmxMol).

        Returns:
            Transformed CmxMol with preserved properties, or None if
            _forward() returns None.
        """
        # Capture input properties for preservation through parallelization
        if isinstance(mol, CmxMol):
            input_props = mol._prop_cache.copy()
        else:
            input_props = mol.GetPropsAsDict(includePrivate=True)

        # Call subclass implementation
        result = self._forward(mol)
        if result is None:
            return None

        # Wrap in CmxMol for property preservation through parallelization
        output: CmxMol = result if isinstance(result, CmxMol) else CmxMol(result)

        # Restore input properties
        for key, value in input_props.items():
            if isinstance(value, float):
                output.SetDoubleProp(key, value)
            elif isinstance(value, int) and not isinstance(value, bool):
                output.SetIntProp(key, value)
            elif isinstance(value, bool):
                output.SetBoolProp(key, value)
            else:
                output.SetProp(key, str(value))

        return output
