"""Block for molecule standardization and preprocessing."""

import logging

from rdkit import Chem
from rdkit.Chem.MolStandardize.rdMolStandardize import (
    LargestFragmentChooser,
    MetalDisconnector,
    Normalizer,
    TautomerEnumerator,
    Uncharger,
)

from cmxflow.operators.base import MoleculeBlock

logger = logging.getLogger(__name__)


class MoleculeStandardizeBlock(MoleculeBlock):
    """Standardize molecules for drug discovery preprocessing.

    Applies a standard pipeline: metal disconnection, normalization,
    salt/fragment removal, and charge neutralization. Optionally
    canonicalizes tautomers.

    Args:
        canonicalize_tautomers: Whether to canonicalize tautomers.
            Defaults to False.
        **kwargs: Additional keyword arguments passed to set_inputs.

    Example:
        >>> block = MoleculeStandardizeBlock()
        >>> mol = Chem.MolFromSmiles("CCO.[Na+].[Cl-]")
        >>> result = block._forward(mol)
        >>> Chem.MolToSmiles(result)
        'CCO'
    """

    def __init__(self, canonicalize_tautomers: bool = False, **kwargs: str) -> None:
        """Initialize the MoleculeStandardizeBlock.

        Args:
            canonicalize_tautomers: Whether to canonicalize tautomers.
            **kwargs: Additional keyword arguments passed to set_inputs.
        """
        super().__init__(name="MoleculeStandardize")
        self.canonicalize_tautomers = canonicalize_tautomers
        self._metal_disconnector = MetalDisconnector()
        self._normalizer = Normalizer()
        self._largest_fragment = LargestFragmentChooser()
        self._uncharger = Uncharger()
        self._tautomer_enumerator = (
            TautomerEnumerator() if canonicalize_tautomers else None
        )
        self.set_inputs(**kwargs)

    def __getstate__(self) -> dict:
        """Get state for pickling, excluding unpicklable RDKit objects."""
        state = self.__dict__.copy()
        for key in (
            "_metal_disconnector",
            "_normalizer",
            "_largest_fragment",
            "_uncharger",
            "_tautomer_enumerator",
        ):
            state.pop(key, None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state from pickle, recreating RDKit objects."""
        self.__dict__.update(state)
        self._metal_disconnector = MetalDisconnector()
        self._normalizer = Normalizer()
        self._largest_fragment = LargestFragmentChooser()
        self._uncharger = Uncharger()
        self._tautomer_enumerator = (
            TautomerEnumerator() if self.canonicalize_tautomers else None
        )

    def _forward(self, mol: Chem.Mol) -> Chem.Mol | None:
        """Standardize a molecule through the preprocessing pipeline.

        Args:
            mol: Input RDKit Mol object.

        Returns:
            Standardized molecule, or None if any step fails.
        """
        try:
            mol = self._metal_disconnector.Disconnect(mol)
            mol = self._normalizer.normalize(mol)
            mol = self._largest_fragment.choose(mol)
            mol = self._uncharger.uncharge(mol)
            if self._tautomer_enumerator is not None:
                mol = self._tautomer_enumerator.Canonicalize(mol)
            return mol
        except Exception:
            logger.warning("Failed to standardize molecule, skipping.", exc_info=True)
            return None
