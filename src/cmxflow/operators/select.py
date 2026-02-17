"""Blocks for selecting molecules by property values."""

import logging
from collections.abc import Iterator
from typing import Any

from rdkit import Chem

from cmxflow.cmxmol import Mol as CmxMol
from cmxflow.operators.base import MoleculeBlock

logger = logging.getLogger(__name__)


class PropertySelectBlock(MoleculeBlock):
    """Base class for blocks that select molecules by property ranking.

    Collects ALL molecules from the input iterator, sorts by a specified
    property, and returns the top or bottom N molecules.

    This is an N:M transform that must collect all inputs before producing
    any output, unlike the streaming 1:1 pattern of standard MoleculeBlock.

    Subclasses must define _ascending to control sort direction.

    Attributes:
        _ascending: If True, sort ascending (low to high). If False, descending.
    """

    _ascending: bool  # Subclasses must define

    def __init__(self, name: str, **kwargs: Any) -> None:
        """Initialize the property select block.

        Args:
            name: Block name.
            **kwargs: Additional arguments passed to set_inputs.
        """
        super().__init__(name=name, input_text=["property", "count"])
        self.set_inputs(**kwargs)

    def _forward(self, mol: Chem.Mol) -> Chem.Mol | None:
        """Not implemented - this block overrides __call__ directly.

        PropertySelectBlock is an N:M transform that must collect all inputs
        before producing output. The __call__ method handles the full logic.

        Args:
            mol: Unused.

        Raises:
            NotImplementedError: Always raised as this method should not be called.
        """
        raise NotImplementedError(f"{self.__class__.__name__} uses __call__ directly")

    def _get_count(self) -> int:
        """Parse count from input_text, defaulting to 0 (all).

        Returns:
            Count as integer, or 0 if empty/invalid.
        """
        count_str = self.input_text.get("count", "").strip()
        if not count_str:
            return 0
        try:
            return int(count_str)
        except ValueError:
            logger.warning(f"Invalid count '{count_str}', using 0 (all)")
            return 0

    def _get_property_value(self, mol: Chem.Mol, property_name: str) -> float:
        """Get a property value from a molecule as a float.

        Args:
            mol: RDKit Mol object.
            property_name: Name of the property to retrieve.

        Returns:
            Property value as float.

        Raises:
            KeyError: If the property is missing or cannot be converted to float.
        """
        if not mol.HasProp(property_name):
            raise KeyError(f"Molecule missing property: {property_name}")

        try:
            return float(mol.GetDoubleProp(property_name))
        except (KeyError, RuntimeError, ValueError):
            pass

        try:
            return float(mol.GetIntProp(property_name))
        except (KeyError, RuntimeError, ValueError):
            pass

        try:
            return float(mol.GetProp(property_name))
        except (KeyError, RuntimeError, ValueError):
            pass

        raise KeyError(
            f"Property '{property_name}' cannot be converted to numeric value"
        )

    def __call__(self, iter: Iterator[Any]) -> Iterator[CmxMol]:
        """Collect, sort, and yield selected molecules.

        Args:
            iter: Iterator of input molecules.

        Yields:
            Selected molecules sorted by property.

        Raises:
            KeyError: If a molecule is missing the specified property.
        """
        property_name = self.input_text.get("property", "").strip()
        if not property_name:
            logger.warning("No property specified, passing all molecules through")
            yield from iter
            return

        count = self._get_count()

        # Collect molecules with property values
        molecules_with_values: list[tuple[float, CmxMol]] = []

        for mol in iter:
            if not self.check_input(mol):
                continue

            value = self._get_property_value(mol, property_name)

            # Wrap in CmxMol if needed for property preservation
            if isinstance(mol, CmxMol):
                cmx_mol = mol
            else:
                cmx_mol = CmxMol(mol)

            molecules_with_values.append((value, cmx_mol))

        # Sort by property value
        molecules_with_values.sort(
            key=lambda x: x[0],
            reverse=not self._ascending,
        )

        # Select subset
        if count > 0:
            selected = molecules_with_values[:count]
        else:
            selected = molecules_with_values

        # Yield selected molecules
        for _, mol in selected:
            if self.check_output(mol):
                yield mol


class PropertyHeadBlock(PropertySelectBlock):
    """Block that returns molecules with the highest property values.

    Collects all input molecules, sorts by the specified property in
    descending order, and yields the top N molecules (highest values first).

    Required Inputs:
        property (text): Name of the molecule property to sort by.
        count (text): Number of molecules to return. 0 or empty returns all sorted.

    Example:
        workflow.add(PropertyHeadBlock())
        workflow.set_required_input({
            "1.text@property": "docking_score",
            "1.text@count": "10",
        })
    """

    _ascending = False  # Descending: highest values first

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the PropertyHeadBlock."""
        super().__init__(name="PropertyHead", **kwargs)


class PropertyTailBlock(PropertySelectBlock):
    """Block that returns molecules with the lowest property values.

    Collects all input molecules, sorts by the specified property in
    ascending order, and yields the bottom N molecules (lowest values first).

    Required Inputs:
        property (text): Name of the molecule property to sort by.
        count (text): Number of molecules to return. 0 or empty returns all sorted.

    Example:
        workflow.add(PropertyTailBlock())
        workflow.set_required_input({
            "1.text@property": "energy",
            "1.text@count": "5",
        })
    """

    _ascending = True  # Ascending: lowest values first

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the PropertyTailBlock."""
        super().__init__(name="PropertyTail", **kwargs)
