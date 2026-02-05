"""Molecule wrapper that preserves properties through pickling."""

from __future__ import annotations

from typing import Any

from rdkit import Chem


class Mol(Chem.Mol):
    """Wrapper around RDKit Mol that preserves properties through pickling.

    RDKit Mol objects can lose properties during certain operations. This wrapper
    maintains a property cache that survives pickling and can restore properties
    to the underlying Mol object.

    Attributes:
        mol: The wrapped RDKit Mol object.
        _prop_cache: Cached properties as a dictionary.
    """

    def __init__(self, mol: Chem.Mol) -> None:
        """Initialize with an RDKit Mol.

        Args:
            mol: RDKit Mol object to wrap.
        """
        super().__init__(mol)
        self._prop_cache: dict[str, Any] = super().GetPropsAsDict(includePrivate=True)

    def SetProp(self, key: str, value: str, **kwargs: bool) -> None:
        """Set a string property, caching for pickle preservation.

        Args:
            key: Property name.
            value: Property value as string.
            **kwargs: Additional arguments passed to RDKit SetProp.
        """
        self._prop_cache[key] = value
        super().SetProp(key, value, **kwargs)

    def SetDoubleProp(self, key: str, value: float, **kwargs: bool) -> None:
        """Set a float property, caching for pickle preservation.

        Args:
            key: Property name.
            value: Property value as float.
            **kwargs: Additional arguments passed to RDKit SetDoubleProp.
        """
        self._prop_cache[key] = value
        super().SetDoubleProp(key, value, **kwargs)

    def SetBoolProp(self, key: str, value: bool, **kwargs: bool) -> None:
        """Set a bool property, caching for pickle preservation.

        Args:
            key: Property name.
            value: Property value as bool.
            **kwargs: Additional arguments passed to RDKit SetBoolProp.
        """
        self._prop_cache[key] = value
        super().SetBoolProp(key, value, **kwargs)

    def SetIntProp(self, key: str, value: int, **kwargs: bool) -> None:
        """Set an int property, caching for pickle preservation.

        Args:
            key: Property name.
            value: Property value as int.
            **kwargs: Additional arguments passed to RDKit SetIntProp.
        """
        self._prop_cache[key] = value
        super().SetIntProp(key, value, **kwargs)

    def restore_properties(self) -> None:
        """Restore cached properties to the Mol."""
        for key, value in self._prop_cache.items():
            if isinstance(value, float):
                self.SetDoubleProp(key, value)
            elif isinstance(value, int) and not isinstance(value, bool):
                self.SetIntProp(key, value)
            elif isinstance(value, bool):
                self.SetBoolProp(key, value)
            else:
                self.SetProp(key, str(value))

    def GetPropsAsDict(self, **kwargs: bool) -> dict[str, Any]:
        """Get properties as dict, restoring cached properties first.

        Args:
            **kwargs: Additional arguments passed to RDKit GetPropsAsDict.

        Returns:
            Dictionary of all molecule properties.
        """
        self.restore_properties()
        result: dict[str, Any] = super().GetPropsAsDict(**kwargs)
        return result


def wrap_mol(mol: Chem.Mol) -> Mol:
    """Wrap an RDKit Mol in a Mol for property preservation.

    Args:
        mol: RDKit Mol object.

    Returns:
        Mol wrapper with cached properties.
    """
    return Mol(mol)


def unwrap_mol(cmx_mol: Mol | Chem.Mol) -> Chem.Mol:
    """Extract the RDKit Mol from a Mol wrapper.

    Args:
        cmx_mol: Mol wrapper or plain RDKit Mol.

    Returns:
        The underlying RDKit Mol object.
    """
    if isinstance(cmx_mol, Mol):
        return cmx_mol.mol
    return cmx_mol
