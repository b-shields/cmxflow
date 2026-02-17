"""3D molecular similarity block."""

import logging
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdShapeHelpers

from cmxflow.operators.base import MoleculeBlock
from cmxflow.parameter import Categorical, Continuous
from cmxflow.sources.reader import read_molecules

logger = logging.getLogger(__name__)


class Molecule3DSimilarityBlock(MoleculeBlock):
    """Compute 3D molecular similarity against a set of query molecules.

    Both input and query molecules must have pre-existing 3D conformers.
    For each input molecule, computes maximum similarity across all conformer
    pairs and attaches the result as properties.

    Required Inputs:
        - query (file): Path to query molecule file with 3D conformers.

    Output Properties:
        - similarity_3d: Maximum 3D similarity score to any query conformer.
        - most_similar_query_3d: Name of the most similar query molecule.
        - similarity_3d_method: Similarity method used.
        - similarity_3d_conf_id: Conformer ID that gave the best similarity.

    Example:
        workflow.add(
            MoleculeSourceBlock(),
            EnumerateStereoBlock(),
            ConformerGenerationBlock(),
            Molecule3DSimilarityBlock(query="reference_3d.sdf"),
            MoleculeSinkBlock()
        )

    Mutable Parameters:
        - method: Similarity method (shape_tanimoto, shape_tversky, usr, usrcat).
        - tversky_alpha: Tversky alpha parameter (0.0–1.0).
        - tversky_beta: Tversky beta parameter (0.0–1.0).
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the 3D similarity block."""
        super().__init__(name="Molecule3DSimilarity", input_files=["query"])

        # Register mutable parameters
        self.mutable(
            Categorical(
                "method",
                default="shape_tanimoto",
                choices=["shape_tanimoto", "shape_tversky", "usr", "usrcat"],
            ),
            Continuous("tversky_alpha", default=1.0, low=0.0, high=1.0),
            Continuous("tversky_beta", default=1.0, low=0.0, high=1.0),
        )
        self.set_inputs(**kwargs)

        # Lazy-loaded query data
        self._query_mols: list[Chem.Mol] | None = None
        self._query_names: list[str] | None = None
        self._query_descriptors: list[list[float]] | None = None

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

    def _load_queries(self) -> None:
        """Load and validate query molecules with 3D conformers.

        For USR/USRCAT methods, also precomputes and caches descriptors.

        Raises:
            ValueError: If no valid query molecules found or missing 3D conformers.
        """
        query_path = Path(self.input_files["query"])
        method = self.params["method"].get()

        self._query_mols = []
        self._query_names = []
        self._query_descriptors = []

        for i, mol in enumerate(read_molecules(query_path)):
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"query_{i}"

            if not self._has_3d_conformer(mol):
                raise ValueError(
                    f"Query molecule '{name}' does not have 3D conformers. "
                    "All query molecules must have pre-generated 3D conformers."
                )

            # Add hydrogens with computed 3D coordinates
            mol_with_h = Chem.AddHs(mol, addCoords=True)
            self._query_mols.append(mol_with_h)
            self._query_names.append(name)

            # Precompute USR/USRCAT descriptors for efficiency
            if method == "usr":
                desc: list[float] = list(rdMolDescriptors.GetUSR(mol_with_h))
                self._query_descriptors.append(desc)
            elif method == "usrcat":
                desc = list(rdMolDescriptors.GetUSRCAT(mol_with_h))
                self._query_descriptors.append(desc)

        if not self._query_mols:
            raise ValueError(f"No valid query molecules found in file: {query_path}")

    def _compute_shape_similarity(
        self,
        mol: Chem.Mol,
        conf_id: int,
        ref: Chem.Mol,
        ref_conf_id: int,
    ) -> float:
        """Compute shape-based similarity between two conformers.

        Args:
            mol: Input molecule.
            conf_id: Conformer ID in input molecule.
            ref: Reference molecule.
            ref_conf_id: Conformer ID in reference molecule.

        Returns:
            Similarity score (0-1, higher is more similar).
        """
        method = self.params["method"].get()

        try:
            if method == "shape_tanimoto":
                # ShapeTanimotoDist returns distance (0=identical, 1=no overlap)
                distance: float = rdShapeHelpers.ShapeTanimotoDist(
                    mol, ref, confId1=conf_id, confId2=ref_conf_id
                )
                return 1.0 - distance

            elif method == "shape_tversky":
                alpha = self.params["tversky_alpha"].get()
                beta = self.params["tversky_beta"].get()
                # ShapeTverskyIndex returns similarity directly
                similarity: float = rdShapeHelpers.ShapeTverskyIndex(
                    mol, ref, alpha, beta, confId1=conf_id, confId2=ref_conf_id
                )
                return similarity

            else:
                raise ValueError(f"Unknown shape method: {method}")

        except Exception as e:
            logger.debug(f"Shape similarity computation failed: {e}")
            return 0.0

    def _compute_usr_similarity(
        self,
        mol: Chem.Mol,
        conf_id: int,
        ref_descriptor: list[float],
    ) -> float:
        """Compute USR-based similarity between molecule and reference descriptor.

        Args:
            mol: Input molecule.
            conf_id: Conformer ID to use.
            ref_descriptor: Precomputed USR/USRCAT descriptor for reference.

        Returns:
            Similarity score (0-1, higher is more similar).
        """
        method = self.params["method"].get()

        try:
            if method == "usr":
                mol_desc = rdMolDescriptors.GetUSR(mol, confId=conf_id)
            elif method == "usrcat":
                mol_desc = rdMolDescriptors.GetUSRCAT(mol, confId=conf_id)
            else:
                raise ValueError(f"Unknown USR method: {method}")

            # GetUSRScore returns similarity (0-1, higher is more similar)
            score: float = rdMolDescriptors.GetUSRScore(mol_desc, ref_descriptor)  # type: ignore
            return score

        except Exception as e:
            logger.debug(f"USR similarity computation failed: {e}")
            return 0.0

    def check_input(self, arg: Any) -> bool:
        """Validate that input is an RDKit Mol with 3D conformers.

        Args:
            arg: Input item to validate.

        Returns:
            True if valid, False otherwise.
        """
        if not super().check_input(arg):
            return False

        if not self._has_3d_conformer(arg):
            logger.debug("Input molecule does not have 3D conformers. Skipping.")
            return False

        return True

    def _forward(self, mol: Chem.Mol) -> Chem.Mol:
        """Compute 3D similarity between input molecule and query molecules.

        Args:
            mol: Input RDKit Mol object with 3D conformers.

        Returns:
            Input molecule with added properties:
                - similarity_3d: Maximum similarity score to any query.
                - most_similar_query_3d: Name of the most similar query.
                - similarity_3d_method: Method used for comparison.
                - similarity_3d_conf_id: Conformer ID that gave best similarity.
        """
        # Lazy load queries
        if self._query_mols is None:
            self._load_queries()
        assert self._query_mols is not None
        assert self._query_names is not None

        method = self.params["method"].get()
        use_usr = method in ("usr", "usrcat")

        best_similarity = 0.0
        best_query_idx = 0
        best_conf_id = 0

        # Compare all conformer pairs
        for conf_id in range(mol.GetNumConformers()):
            for query_idx, query_mol in enumerate(self._query_mols):
                if use_usr:
                    assert self._query_descriptors is not None
                    similarity = self._compute_usr_similarity(
                        mol, conf_id, self._query_descriptors[query_idx]
                    )
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_query_idx = query_idx
                        best_conf_id = conf_id
                else:
                    # Shape methods - compare against all query conformers
                    for ref_conf_id in range(query_mol.GetNumConformers()):
                        similarity = self._compute_shape_similarity(
                            mol, conf_id, query_mol, ref_conf_id
                        )
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_query_idx = query_idx
                            best_conf_id = conf_id

        # Attach properties to molecule
        mol.SetDoubleProp("similarity_3d", best_similarity)
        mol.SetProp("most_similar_query_3d", self._query_names[best_query_idx])
        mol.SetProp("similarity_3d_method", method)
        mol.SetIntProp("similarity_3d_conf_id", best_conf_id)

        return mol
