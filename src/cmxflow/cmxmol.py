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
            cache_props: If True, immediately cache current properties.
        """
        super().__init__(mol)
        self._prop_cache: dict[str, Any] = super().GetPropsAsDict(includePrivate=True)

    def SetProp(self, key: str, value: str, **kwargs):
        self._prop_cache[key] = value
        super().SetProp(key, value, **kwargs)

    def SetDoubleProp(self, key: str, value: float, **kwargs):
        self._prop_cache[key] = value
        super().SetDoubleProp(key, value, **kwargs)

    def SetBoolProp(self, key: str, value: bool, **kwargs):
        self._prop_cache[key] = value
        super().SetBoolProp(key, value, **kwargs)

    def SetIntProp(self, key: str, value: int, **kwargs):
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
