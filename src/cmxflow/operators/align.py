"""3D molecular alignment block."""

import logging
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS, rdMolAlign, rdShapeHelpers

from cmxflow.operators.base import MoleculeBlock
from cmxflow.parameter import Categorical, Integer
from cmxflow.sources.reader import read_molecules

logger = logging.getLogger(__name__)


class MoleculeAlignBlock(MoleculeBlock):
    """Block for 3D molecular alignment.

    Aligns input molecule conformers to reference molecules and returns the
    conformer with highest shape similarity. Reference molecules are lazy-loaded
    from the input file specified by the "query" key.

    Input molecules (both query references and molecules to align) must already
    have 3D conformers. Use a conformer generation block upstream if needed.

    Attributes:
        input_files: Dictionary containing "query" key for reference molecule file.
        params: Dictionary of mutable parameters for alignment configuration.
    """

    def __init__(self) -> None:
        """Initialize the molecular alignment block."""
        super().__init__(input_files=["query"])
        self.name = "MoleculeAlign"

        # Register mutable parameters
        self.mutable(
            Categorical(
                "alignment_method",
                default="crippen_o3a",
                choices=["crippen_o3a", "mmff_o3a", "mcs"],
            ),
            Integer("mcsTimeout", default=3, low=1, high=30),
            Integer("mcsMinAtoms", default=3, low=1, high=10),
        )

        # Lazy-loaded reference molecules with conformers
        self._reference_mols: list[Chem.Mol] | None = None
        self._reference_names: list[str] | None = None

    def _has_3d_conformer(self, mol: Chem.Mol) -> bool:
        """Check if molecule has at least one 3D conformer.

        Args:
            mol: RDKit Mol object to check.

        Returns:
            True if molecule has at least one 3D conformer, False otherwise.
        """
        if mol.GetNumConformers() == 0:
            return False
        return bool(mol.GetConformer().Is3D())

    def _load_reference_molecules(self) -> None:
        """Load and validate reference molecules with 3D conformers.

        Raises:
            ValueError: If no valid reference molecules with 3D conformers found.
        """
        query_path = Path(self.input_files["query"])

        self._reference_mols = []
        self._reference_names = []

        for i, mol in enumerate(read_molecules(query_path)):
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"ref_{i}"

            if not self._has_3d_conformer(mol):
                raise ValueError(
                    f"Reference molecule '{name}' does not have 3D conformers. "
                    "All reference molecules must have pre-generated 3D conformers."
                )

            mol_with_h = Chem.AddHs(mol)
            self._reference_mols.append(mol_with_h)
            self._reference_names.append(name)

        if not self._reference_mols:
            raise ValueError(
                f"No valid reference molecules found in query file: {query_path}"
            )

    def _align_crippen_o3a(
        self,
        mol: Chem.Mol,
        conf_id: int,
        ref: Chem.Mol,
        ref_conf_id: int,
    ) -> tuple[float, float]:
        """Align using Crippen force field based Open3D alignment.

        Args:
            mol: Molecule to align.
            conf_id: Conformer ID in mol.
            ref: Reference molecule.
            ref_conf_id: Conformer ID in reference.

        Returns:
            Tuple of (shape_similarity, rmsd).
        """
        try:
            o3a = rdMolAlign.GetCrippenO3A(mol, ref, prbCid=conf_id, refCid=ref_conf_id)
            o3a.Align()
            rmsd = o3a.Trans()[0]
            # Convert O3A score to similarity (higher is better)
            shape_sim = self._compute_shape_similarity(mol, conf_id, ref, ref_conf_id)
            return shape_sim, rmsd
        except Exception as e:
            logger.debug(f"Crippen O3A alignment failed: {e}")
            return 0.0, float("inf")

    def _align_mmff_o3a(
        self,
        mol: Chem.Mol,
        conf_id: int,
        ref: Chem.Mol,
        ref_conf_id: int,
    ) -> tuple[float, float]:
        """Align using MMFF94 force field based Open3D alignment.

        Args:
            mol: Molecule to align.
            conf_id: Conformer ID in mol.
            ref: Reference molecule.
            ref_conf_id: Conformer ID in reference.

        Returns:
            Tuple of (shape_similarity, rmsd).
        """
        try:
            o3a = rdMolAlign.GetO3A(mol, ref, prbCid=conf_id, refCid=ref_conf_id)
            o3a.Align()
            rmsd = o3a.Trans()[0]
            shape_sim = self._compute_shape_similarity(mol, conf_id, ref, ref_conf_id)
            return shape_sim, rmsd
        except Exception as e:
            logger.debug(f"MMFF O3A alignment failed: {e}")
            return 0.0, float("inf")

    def _align_mcs(
        self,
        mol: Chem.Mol,
        conf_id: int,
        ref: Chem.Mol,
        ref_conf_id: int,
    ) -> tuple[float, float]:
        """Align using maximum common substructure.

        Args:
            mol: Molecule to align.
            conf_id: Conformer ID in mol.
            ref: Reference molecule.
            ref_conf_id: Conformer ID in reference.

        Returns:
            Tuple of (shape_similarity, rmsd).
        """
        mcs_timeout = self.params["mcsTimeout"].get()
        mcs_min_atoms = self.params["mcsMinAtoms"].get()

        try:
            # Find MCS
            mcs_result = rdFMCS.FindMCS(
                [mol, ref],
                timeout=mcs_timeout,
                matchValences=False,
                ringMatchesRingOnly=True,
                completeRingsOnly=True,
            )

            if mcs_result.numAtoms < mcs_min_atoms:
                logger.debug(
                    f"MCS too small: {mcs_result.numAtoms} atoms < {mcs_min_atoms}"
                )
                return 0.0, float("inf")

            # Get atom mapping from MCS
            mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
            if mcs_mol is None:
                return 0.0, float("inf")

            mol_match = mol.GetSubstructMatch(mcs_mol)
            ref_match = ref.GetSubstructMatch(mcs_mol)

            if not mol_match or not ref_match:
                return 0.0, float("inf")

            # Create atom map for alignment
            atom_map = list(zip(mol_match, ref_match))

            # Align using the atom map
            rmsd = AllChem.AlignMol(  # type: ignore[attr-defined]
                mol, ref, prbCid=conf_id, refCid=ref_conf_id, atomMap=atom_map
            )

            shape_sim = self._compute_shape_similarity(mol, conf_id, ref, ref_conf_id)
            return shape_sim, rmsd
        except Exception as e:
            logger.debug(f"MCS alignment failed: {e}")
            return 0.0, float("inf")

    def _compute_shape_similarity(
        self,
        mol: Chem.Mol,
        conf_id: int,
        ref: Chem.Mol,
        ref_conf_id: int,
    ) -> float:
        """Compute shape Tanimoto similarity between aligned molecules.

        Args:
            mol: Aligned molecule.
            conf_id: Conformer ID in mol.
            ref: Reference molecule.
            ref_conf_id: Conformer ID in reference.

        Returns:
            Shape Tanimoto similarity (0-1, higher is better).
        """
        try:
            # ShapeTanimotoDistance returns distance, convert to similarity
            distance: float = rdShapeHelpers.ShapeTanimotoDistance(  # type: ignore[attr-defined]
                mol, ref, confId1=conf_id, confId2=ref_conf_id
            )
            return 1.0 - distance
        except Exception as e:
            logger.debug(f"Shape similarity computation failed: {e}")
            return 0.0

    def _find_best_alignment(
        self, mol: Chem.Mol
    ) -> tuple[int, int, int, float, float] | None:
        """Find the best alignment across all conformers and references.

        Args:
            mol: Input molecule with conformers.

        Returns:
            Tuple of (best_conf_id, best_ref_idx, best_ref_conf_id,
                     best_shape_sim, best_rmsd), or None if no valid alignment.
        """
        assert self._reference_mols is not None

        alignment_method = self.params["alignment_method"].get()

        # Select alignment function
        if alignment_method == "crippen_o3a":
            align_func = self._align_crippen_o3a
        elif alignment_method == "mmff_o3a":
            align_func = self._align_mmff_o3a
        else:  # mcs
            align_func = self._align_mcs

        best_conf_id = -1
        best_ref_idx = -1
        best_ref_conf_id = -1
        best_shape_sim = -1.0
        best_rmsd = float("inf")

        # Iterate over all combinations
        for conf_id in range(mol.GetNumConformers()):
            for ref_idx, ref in enumerate(self._reference_mols):
                for ref_conf_id in range(ref.GetNumConformers()):
                    # Create a copy for alignment to avoid modifying original
                    mol_copy = Chem.Mol(mol)
                    shape_sim, rmsd = align_func(mol_copy, conf_id, ref, ref_conf_id)

                    if shape_sim > best_shape_sim:
                        best_shape_sim = shape_sim
                        best_rmsd = rmsd
                        best_conf_id = conf_id
                        best_ref_idx = ref_idx
                        best_ref_conf_id = ref_conf_id

        if best_conf_id < 0:
            return None

        return best_conf_id, best_ref_idx, best_ref_conf_id, best_shape_sim, best_rmsd

    def check_output(self, arg: Any) -> bool:
        """Validate that output molecule has alignment properties.

        Args:
            arg: Output molecule to validate.

        Returns:
            True if the molecule has valid alignment, False otherwise.
        """
        if not isinstance(arg, Chem.Mol):
            return False

        if arg.GetNumConformers() == 0:
            logger.info("Alignment failed: output molecule has no conformers")
            return False

        if not arg.HasProp("alignment_shape_similarity"):
            logger.info("Alignment failed: missing alignment properties")
            return False

        return True

    def forward(self, mol: Chem.Mol) -> Chem.Mol | None:
        """Align molecule to reference molecules.

        Args:
            mol: Input RDKit Mol object with 3D conformers.

        Returns:
            Aligned molecule with single best conformer and alignment properties,
            or None if alignment failed or input molecule lacks 3D conformers.
        """
        # Lazy load reference molecules
        if self._reference_mols is None:
            self._load_reference_molecules()
        assert self._reference_mols is not None
        assert self._reference_names is not None

        alignment_method = self.params["alignment_method"].get()

        # Validate input molecule has 3D conformers
        if not self._has_3d_conformer(mol):
            logger.debug(
                "Input molecule does not have 3D conformers. " "Skipping alignment."
            )
            return None

        mol_with_conf = Chem.AddHs(Chem.Mol(mol))

        # Find best alignment
        result = self._find_best_alignment(mol_with_conf)
        if result is None:
            logger.debug("No valid alignment found")
            return None

        best_conf_id, best_ref_idx, best_ref_conf_id, best_shape_sim, best_rmsd = result

        # Perform final alignment on the molecule we'll return
        ref = self._reference_mols[best_ref_idx]

        # Select alignment function
        if alignment_method == "crippen_o3a":
            align_func = self._align_crippen_o3a
        elif alignment_method == "mmff_o3a":
            align_func = self._align_mmff_o3a
        else:  # mcs
            align_func = self._align_mcs

        # Final alignment
        align_func(mol_with_conf, best_conf_id, ref, best_ref_conf_id)

        # Keep only the best conformer
        best_conf = mol_with_conf.GetConformer(best_conf_id)
        output_mol = Chem.Mol(mol_with_conf)
        output_mol.RemoveAllConformers()
        output_mol.AddConformer(best_conf, assignId=True)

        # Attach alignment properties
        output_mol.SetDoubleProp("alignment_shape_similarity", best_shape_sim)
        output_mol.SetDoubleProp("alignment_rmsd", best_rmsd)
        output_mol.SetProp("alignment_reference", self._reference_names[best_ref_idx])
        output_mol.SetProp("alignment_method", alignment_method)
        output_mol.SetIntProp("alignment_ref_index", best_ref_idx)

        return output_mol
