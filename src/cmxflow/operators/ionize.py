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

    When the input carries a 3D conformer the conformer is preserved: dimorphite
    only changes formal charges and hydrogen counts on an unchanged heavy-atom
    skeleton, so the protonation state is transferred back onto the original 3D
    heavy atoms (matched exactly by atom-map number, not substructure search) and
    any hydrogens added during protonation get coordinates from the heavy-atom
    geometry. Inputs without a 3D conformer take the plain SMILES path.

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

    def _protonated_variants(self, dimorphite_dl: Any, smiles: str) -> list[Chem.Mol]:
        """Run dimorphite_dl on a SMILES and return fixed 2D variant mols.

        Args:
            dimorphite_dl: The imported dimorphite_dl module.
            smiles: SMILES (optionally atom-map labelled) to protonate.

        Returns:
            RDKit mols for each protonation variant, with the tertiary-amide
            correction applied. Invalid SMILES are skipped; on dimorphite_dl
            failure the input is returned unchanged.
        """
        try:
            protonated = dimorphite_dl.protonate_smiles(
                smiles,
                ph_min=self.ph_min,
                ph_max=self.ph_max,
                precision=self.params["precision"].get(),
                max_variants=self.params["max_variants"].get(),
            )
        except Exception:
            logger.warning(
                "dimorphite_dl failed for %s, falling back to original.",
                smiles,
                exc_info=True,
            )
            protonated = [smiles]

        variants = []
        for smi in protonated:
            variant = Chem.MolFromSmiles(smi)
            if variant is None:
                logger.debug("Skipping invalid SMILES from dimorphite_dl: %s", smi)
                continue
            variants.append(self._fix_tertiary_amides(variant))
        return variants

    @staticmethod
    def _transfer_protonation(heavy: Chem.Mol, variant: Chem.Mol) -> Chem.Mol | None:
        """Apply a variant's protonation state onto a 3D heavy-atom template.

        ``heavy`` carries the original 3D conformer with every atom labelled by
        an atom-map number; ``variant`` is the (2D) protonated form whose atoms
        keep those same map numbers. Formal charge and hydrogen count are copied
        atom-for-atom by map number, then hydrogens are rebuilt with coordinates
        so any proton added during protonation gets a 3D position.

        Args:
            heavy: Map-labelled heavy-atom mol holding the original conformer.
            variant: Protonated variant mol with matching atom-map numbers.

        Returns:
            A 3D mol in the variant's protonation state, or None if it fails to
            sanitize.
        """
        by_map = {
            atom.GetAtomMapNum(): (atom.GetFormalCharge(), atom.GetTotalNumHs())
            for atom in variant.GetAtoms()
            if atom.GetAtomMapNum()
        }
        out = Chem.RWMol(heavy)
        for atom in out.GetAtoms():
            state = by_map.get(atom.GetAtomMapNum())
            if state is not None:
                charge, n_hs = state
                atom.SetFormalCharge(charge)
                atom.SetNumExplicitHs(n_hs)
                atom.SetNoImplicit(True)
            atom.SetAtomMapNum(0)
        mol = out.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            logger.warning("Failed to sanitize transferred protonation state.")
            return None
        # addCoords places newly added hydrogens from the heavy-atom geometry.
        return Chem.AddHs(mol, addCoords=True)

    def _attach_props(self, variant: Mol, input_props: dict) -> Mol:
        """Copy preserved input properties onto an output variant."""
        for key, value in input_props.items():
            if isinstance(value, float):
                variant.SetDoubleProp(key, value)
            elif isinstance(value, int):
                variant.SetIntProp(key, value)
            else:
                variant.SetProp(key, str(value))
        return variant

    def forward(self, mol: Chem.Mol | Mol) -> Iterator[Mol]:  # type: ignore[override]
        """Generate ionization variants of a molecule.

        Args:
            mol: Input RDKit Mol object.

        Yields:
            Ionization variants with properties preserved. A 3D conformer on the
            input is carried through to every variant.
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

        is_3d = mol.GetNumConformers() > 0 and mol.GetConformer().Is3D()
        seen: set[str] = set()

        if not is_3d:
            # SMILES path: no conformer to preserve.
            for variant in self._protonated_variants(
                dimorphite_dl, Chem.MolToSmiles(mol)
            ):
                canonical = Chem.MolToSmiles(variant)
                if canonical in seen:
                    continue
                seen.add(canonical)
                yield self._attach_props(wrap_mol(variant), input_props)
            return

        # 3D path: label heavy atoms so the protonation state can be mapped back
        # exactly onto the original conformer (dimorphite_dl preserves map nums).
        heavy = Chem.RemoveHs(mol)
        for atom in heavy.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)

        for variant in self._protonated_variants(
            dimorphite_dl, Chem.MolToSmiles(heavy)
        ):
            out = self._transfer_protonation(heavy, variant)
            if out is None:
                continue
            canonical = Chem.MolToSmiles(Chem.RemoveHs(out))
            if canonical in seen:
                continue
            seen.add(canonical)
            yield self._attach_props(wrap_mol(out), input_props)

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
