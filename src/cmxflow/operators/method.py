"""Block for applying arbitrary RDKit methods to molecules."""

import importlib
from typing import Any, Callable

from rdkit import Chem

from cmxflow import Mol
from cmxflow.operators.base import MoleculeBlock


class RDKitBlock(MoleculeBlock):
    """Block that applies an RDKit method to molecules.

    The method can be provided as:
    - A callable that accepts a Mol object
    - A string path to an RDKit method (e.g., "rdkit.Chem.Descriptors.MolWt")

    Return type handling:
    - If method returns Mol: use as the output molecule
    - If method returns str/int/float/bool: add as property with method name as key
    - If method returns None: molecule is filtered out

    Example:
        workflow.add(RDKitBlock(method="rdkit.Chem.Descriptors.MolWt"))
    """

    def __init__(
        self, method: Callable[[Chem.Mol], Any] | str, name: str | None = None, **kwargs
    ) -> None:
        """Initialize with an RDKit method.

        Args:
            method: RDKit method as callable or string path
                (e.g., "rdkit.Chem.Descriptors.MolWt").
            name: Optional property name for scalar results.
                Defaults to the method name extracted from callable or path.
        """
        self._method_ref = method

        if isinstance(method, str):
            self._method = self._import_method(method)
            self._method_name = method.rsplit(".", 1)[-1]
        else:
            self._method = method
            # Try to get a meaningful name from the callable
            func_name = getattr(method, "__name__", None)
            if func_name and func_name != "<lambda>":
                self._method_name = func_name
            else:
                # Fall back to qualname or require explicit name
                qual_name = getattr(method, "__qualname__", None)
                if qual_name and qual_name != "<lambda>":
                    self._method_name = qual_name.rsplit(".", 1)[-1]
                elif name is None:
                    raise ValueError(
                        "Cannot determine method name from callable. "
                        "Please provide an explicit 'name' parameter."
                    )
                else:
                    self._method_name = name

        self._property_name = name or self._method_name

        super().__init__(name=f"RDKit:{self._property_name}")

    def __getstate__(self) -> dict:
        """Get state for pickling, excluding the resolved method callable."""
        state = self.__dict__.copy()
        state.pop("_method", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state from pickle, re-resolving the method callable."""
        self.__dict__.update(state)
        if isinstance(self._method_ref, str):
            self._method = self._import_method(self._method_ref)
        else:
            self._method = self._method_ref

    def _import_method(self, method_path: str) -> Callable[[Chem.Mol], Any]:
        """Import method from a dot-separated string path.

        Args:
            method_path: Dot-separated path (e.g., "rdkit.Chem.Descriptors.MolWt").

        Returns:
            The imported callable.

        Raises:
            ValueError: If the path is invalid.
            ImportError: If the module cannot be imported.
            AttributeError: If the method does not exist in the module.
        """
        parts = method_path.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid method path '{method_path}'. "
                "Expected format: 'module.submodule.method'"
            )

        module_path, method_name = parts
        module = importlib.import_module(module_path)
        method: Callable[[Chem.Mol], Any] = getattr(module, method_name)
        return method

    def _forward(self, mol: Chem.Mol | Mol) -> Chem.Mol | Mol | None:
        """Apply the RDKit method to the molecule.

        Args:
            mol: Input RDKit Mol object.

        Returns:
            Modified molecule with property added, or new Mol if method returns
            Mol. Returns None if the method returns None (molecule filtered).
        """
        result = self._method(mol)

        if result is None:
            return None

        if isinstance(result, Chem.Mol):
            result = Mol(result)
            if isinstance(mol, Mol):
                result._prop_cache = {**mol._prop_cache, **result._prop_cache}
            return result
        elif isinstance(result, bool):
            mol.SetBoolProp(self._property_name, result)
        elif isinstance(result, float):
            mol.SetDoubleProp(self._property_name, result)
        elif isinstance(result, int):
            mol.SetIntProp(self._property_name, result)
        elif isinstance(result, str):
            mol.SetProp(self._property_name, result)
        else:
            mol.SetProp(self._property_name, str(result))

        return mol
