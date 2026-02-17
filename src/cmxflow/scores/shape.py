"""Shape overlay scoring block for molecular workflow optimization."""

import logging
from collections.abc import Iterator
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdShapeHelpers

from cmxflow.block import ScoreBlock
from cmxflow.cmxmol import Mol as CmxMol
from cmxflow.sources.reader import read_molecules

logger = logging.getLogger(__name__)


class ShapeOverlayScoreBlock(ScoreBlock):
    """ScoreBlock for shape overlay-based molecular scoring.

    Computes the average maximum shape Tanimoto similarity between input
    molecules and reference ligands. Both input and reference molecules
    must have pre-existing 3D conformers.

    The objective function computes:
        1. For each input molecule, find the maximum shape Tanimoto similarity
           across all conformer pairs (input conformer x reference conformer).
        2. Return the average of these maximum similarities across all molecules.

    Required Inputs:
        query (file): Path to reference ligand file with 3D conformers.

    Example:
        workflow.add(ShapeOverlayScoreBlock())
        workflow.set_required_input({
            "1.file@query": "reference_3d.sdf",
        })
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the shape overlay score block."""
        super().__init__(input_files=["query"])
        self.set_inputs(**kwargs)

        # Lazy-loaded reference data
        self._reference_mols: list[Chem.Mol] | None = None
        self._reference_names: list[str] | None = None

    def _has_3d_conformer(self, mol: Chem.Mol) -> bool:
        """Check if molecule has at least one 3D conformer.

        Args:
            mol: RDKit Mol object to check.

        Returns:
            True if molecule has at least one 3D conformer.
        """
        if mol.GetNumConformers() == 0:
            return False
        return bool(mol.GetConformer().Is3D())

    def _load_reference(self) -> None:
        """Load and validate reference molecules with 3D conformers.

        Raises:
            ValueError: If no valid reference molecules found or missing 3D conformers.
        """
        reference_path = Path(self.input_files["query"])

        self._reference_mols = []
        self._reference_names = []

        for i, mol in enumerate(read_molecules(reference_path, wrap=False)):
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"reference_{i}"

            if not self._has_3d_conformer(mol):
                raise ValueError(
                    f"Reference molecule '{name}' does not have 3D conformers. "
                    "All reference molecules must have pre-generated 3D conformers."
                )

            # Add hydrogens with computed 3D coordinates
            mol_with_h = Chem.AddHs(mol, addCoords=True)
            self._reference_mols.append(mol_with_h)
            self._reference_names.append(name)

        if not self._reference_mols:
            raise ValueError(
                f"No valid reference molecules found in file: {reference_path}"
            )

    def _compute_shape_tanimoto(
        self,
        mol: Chem.Mol,
        conf_id: int,
        ref: Chem.Mol,
        ref_conf_id: int,
    ) -> float:
        """Compute shape Tanimoto similarity between two conformers.

        Args:
            mol: Input molecule.
            conf_id: Conformer ID in input molecule.
            ref: Reference molecule.
            ref_conf_id: Conformer ID in reference molecule.

        Returns:
            Similarity score (0-1, higher is more similar).
        """
        try:
            # ShapeTanimotoDist returns distance (0=identical, 1=no overlap)
            distance: float = rdShapeHelpers.ShapeTanimotoDist(
                mol, ref, confId1=conf_id, confId2=ref_conf_id
            )
            return 1.0 - distance
        except Exception as e:
            logger.debug(f"Shape Tanimoto computation failed: {e}")
            return 0.0

    def _compute_max_similarity(self, mol: Chem.Mol) -> tuple[float, str]:
        """Find maximum shape Tanimoto similarity across all conformer pairs.

        Args:
            mol: Input molecule with 3D conformers.

        Returns:
            Tuple of (max_similarity, most_similar_reference_name).
        """
        assert self._reference_mols is not None
        assert self._reference_names is not None

        best_similarity = 0.0
        best_reference_idx = 0

        # Add hydrogens with computed 3D coordinates
        mol_with_h = Chem.AddHs(mol, addCoords=True)

        # Compare all conformer pairs
        for conf_id in range(mol_with_h.GetNumConformers()):
            for ref_idx, ref_mol in enumerate(self._reference_mols):
                for ref_conf_id in range(ref_mol.GetNumConformers()):
                    similarity = self._compute_shape_tanimoto(
                        mol_with_h, conf_id, ref_mol, ref_conf_id
                    )
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_reference_idx = ref_idx

        return best_similarity, self._reference_names[best_reference_idx]

    def objective(self, iter: Iterator[Chem.Mol | CmxMol]) -> float:
        """Compute average maximum shape Tanimoto similarity.

        Args:
            iter: Iterator of molecules with 3D conformers.

        Returns:
            Average maximum shape Tanimoto similarity score.
        """
        # Lazy load reference molecules
        if self._reference_mols is None:
            self._load_reference()

        scores: list[float] = []
        skipped = 0

        for mol in iter:
            if mol is None:
                continue

            if not self._has_3d_conformer(mol):
                logger.debug("Input molecule missing 3D conformers, skipping")
                skipped += 1
                continue

            max_similarity, _ = self._compute_max_similarity(mol)
            scores.append(max_similarity)

        if skipped > 0:
            logger.warning(f"Skipped {skipped} molecules without 3D conformers")

        if not scores:
            logger.warning("No valid molecules with 3D conformers, returning 0.0")
            return 0.0

        return sum(scores) / len(scores)

    def forward(self, mol: Chem.Mol | CmxMol) -> Chem.Mol | CmxMol | None:
        """Annotate molecule with shape overlay score.

        Args:
            mol: Input molecule.

        Returns:
            Molecule with shape_overlay_score and shape_overlay_reference
            properties, or None if molecule lacks 3D conformers.
        """
        if mol is None:
            return None

        if not self._has_3d_conformer(mol):
            logger.debug("Input molecule missing 3D conformers, skipping")
            return None

        # Lazy load reference molecules
        if self._reference_mols is None:
            self._load_reference()

        max_similarity, best_reference = self._compute_max_similarity(mol)

        mol.SetDoubleProp("shape_overlay_score", max_similarity)
        mol.SetProp("shape_overlay_reference", best_reference)

        return mol

    def reset_cache(self) -> None:
        """Clear cached reference molecules."""
        self._reference_mols = None
        self._reference_names = None
