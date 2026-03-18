"""Block for generating pH-dependent ionization states."""

import logging
from collections.abc import Iterator
from typing import Any

from rdkit import Chem

from cmxflow import Mol, wrap_mol
from cmxflow.operators.base import MoleculeBlock
from cmxflow.parameter import Continuous, Integer

logger = logging.getLogger(__name__)

# SMARTS for protonated tertiary amide nitrogen: N+ with exactly 1H bonded to C=O
_TERTIARY_AMIDE_SMARTS = "[NX4+;H1;$(NC=O)]"


class IonizeMoleculeBlock(MoleculeBlock):
    """Generate pH-dependent ionization states using dimorphite_dl.

    This is a 1:N transform: one input molecule can yield multiple protonation
    variants. Includes automatic correction for tertiary amide nitrogens that
    dimorphite_dl incorrectly protonates.

    Example:
        ```python
        workflow.add(
            MoleculeSourceBlock(),
            IonizeMoleculeBlock(),
            MoleculeSinkBlock()
        )
        ```

    Mutable Parameters:
        - precision: pH precision window around min/max (0.1–3.0).
        - max_variants: Maximum number of ionization variants per molecule (1–128).
    """

    def __init__(self, ph_min: float = 6.4, ph_max: float = 8.4, **kwargs: Any) -> None:
        """Initialize the IonizeMoleculeBlock.

        Args:
            ph_min: Minimum pH for protonation.
            ph_max: Maximum pH for protonation.
            **kwargs: Additional keyword arguments passed to set_inputs.
        """
        super().__init__(name="IonizeMolecule")
        self.ph_min = ph_min
        self.ph_max = ph_max

        self.mutable(
            Continuous("precision", default=1.0, low=0.1, high=3.0),
            Integer("max_variants", default=4, low=1, high=128),
        )

        self._amide_pattern = Chem.MolFromSmarts(_TERTIARY_AMIDE_SMARTS)
        self.set_inputs(**kwargs)

    def __getstate__(self) -> dict:
        """Get state for pickling, excluding unpicklable RDKit objects."""
        state = self.__dict__.copy()
        state.pop("_amide_pattern", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state from pickle, recreating RDKit objects."""
        self.__dict__.update(state)
        self._amide_pattern = Chem.MolFromSmarts(_TERTIARY_AMIDE_SMARTS)

    def _fix_tertiary_amides(self, mol: Chem.Mol) -> Chem.Mol:
        """Deprotonate incorrectly protonated tertiary amide nitrogens.

        Finds all N atoms matching the protonated tertiary amide pattern
        and sets their formal charge to 0 and explicit H count to 0.

        Args:
            mol: Molecule potentially containing protonated tertiary amides.

        Returns:
            Molecule with tertiary amide nitrogens deprotonated.
        """
        matches = mol.GetSubstructMatches(self._amide_pattern)
        if not matches:
            return mol
        rw = Chem.RWMol(mol)
        for (idx,) in matches:
            atom = rw.GetAtomWithIdx(idx)
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(0)
        try:
            Chem.SanitizeMol(rw)
        except Exception:
            return mol
        return rw.GetMol()

    def _forward(self, mol: Chem.Mol) -> Chem.Mol | None:
        """Not used - this block overrides forward() directly."""
        raise NotImplementedError("IonizeMoleculeBlock uses forward() directly")

    def forward(self, mol: Chem.Mol | Mol) -> Iterator[Mol]:  # type: ignore[override]
        """Generate ionization variants of a molecule.

        Args:
            mol: Input RDKit Mol object.

        Yields:
            Ionization variants with properties preserved.
        """
        try:
            import dimorphite_dl
        except ImportError:
            logger.error(
                "dimorphite_dl is not installed. "
                "Install it with: pip install dimorphite_dl"
            )
            raise

        # Get properties from input molecule
        if isinstance(mol, Mol):
            input_props = mol._prop_cache.copy()
        else:
            input_props = mol.GetPropsAsDict(includePrivate=True)

        input_smiles = Chem.MolToSmiles(mol)
        precision = self.params["precision"].get()
        max_variants = self.params["max_variants"].get()

        try:
            protonated = dimorphite_dl.protonate_smiles(
                input_smiles,
                ph_min=self.ph_min,
                ph_max=self.ph_max,
                precision=precision,
                max_variants=max_variants,
            )
        except Exception:
            logger.warning(
                "dimorphite_dl failed for %s, falling back to original.",
                input_smiles,
                exc_info=True,
            )
            protonated = [input_smiles]

        seen: set[str] = set()
        for smi in protonated:
            variant = Chem.MolFromSmiles(smi)
            if variant is None:
                logger.debug("Skipping invalid SMILES from dimorphite_dl: %s", smi)
                continue
            variant = self._fix_tertiary_amides(variant)
            canonical = Chem.MolToSmiles(variant)
            if canonical in seen:
                continue
            seen.add(canonical)

            variant = wrap_mol(variant)
            for key, value in input_props.items():
                if isinstance(value, float):
                    variant.SetDoubleProp(key, value)
                elif isinstance(value, int):
                    variant.SetIntProp(key, value)
                else:
                    variant.SetProp(key, str(value))
            yield variant

    def __call__(self, iter: Iterator[Any]) -> Iterator[Mol]:
        """Execute the block on an iterator of molecules.

        Overrides the base __call__ to handle 1:N transformation where
        each input molecule can produce multiple ionization variants.

        Args:
            iter: Iterator of input molecules to process.

        Yields:
            Ionization variants that pass input and output checks.
        """
        for arg in iter:
            if not self.check_input(arg):
                continue
            for out in self.forward(arg):
                if self.check_output(out):
                    yield out
