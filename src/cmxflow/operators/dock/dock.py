"""Molecular docking block for cmxflow workflows.

This module provides a MoleculeBlock implementation for docking ligands
into protein binding sites using the empirical (Vinardo) scoring function and
rigid-body/torsional pose optimization.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from cmxflow.operators.base import MoleculeBlock
from cmxflow.operators.dock.ec import compute_gasteiger_charges
from cmxflow.operators.dock.pose import (
    OptimizationResult,
    PoseParams,
    _rigid_topk,
    optimize_pose_cached,
)
from cmxflow.operators.dock.score import (
    AtomTyping,
    EmpiricalParams,
    empirical_score_cached,
    get_atom_typing,
)
from cmxflow.parameter import (
    Categorical,
    Continuous,
    Integer,
)
from cmxflow.sources.reader import read_molecules

logger = logging.getLogger(__name__)


class MoleculeDockBlock(MoleculeBlock):
    """MoleculeBlock for docking ligands into protein binding sites.

    Performs pose optimization using the empirical (Vinardo) scoring function
    with configurable parameters. Supports both rigid-body and flexible
    (torsional) docking. An optional electrostatic complementarity (EC) term
    can be enabled via the ``w_ec`` parameter to reward charge complementarity.

    Supports multi-start optimization (``n_starts``) and scaffold constraints
    (``constraint_smarts``, ``constraint_weight``) for use with the
    MoleculeAlignBlock → MoleculeDockBlock workflow.

    Required Inputs:
        - receptor (file): Path to receptor PDB file.
        - site_reference (file, optional): Molecule file (.sdf, .mol2, etc.)
          whose heavy-atom centroid defines the binding site center. When
          provided, Sobol restart samples are anchored to the pocket rather
          than the input conformer position — required for blind docking from
          a fresh conformer. Omit for MCS/overlay refinement workflows where
          the input pose is already in the binding site.

    Output Properties:
        - docking_initial_pose_score: Score before optimization.
        - docking_score: Final optimized score (empirical + EC adjustment).
        - docking_empirical: Pure empirical score (without EC term).
        - docking_ec: Electrostatic complementarity value (0.0 when w_ec=0).
        - docking_converged: Whether optimization converged.
        When ``score_components=True`` (default), also writes:
        - docking_gauss1: Gaussian term contribution to docking_score.
        - docking_repulsion: Repulsion term contribution to docking_score.
        - docking_hydrophobic: Hydrophobic term contribution to docking_score.
        - docking_hbond: H-bond term contribution to docking_score.
        - docking_n_rot: Torsional entropy energetic term (w_rot * N_rot).
        - docking_scoring_function: Scoring weights used, for reproducibility.

    Example:
        ```python
        workflow.add(
            MoleculeSourceBlock(),
            EnumerateStereoBlock(),
            ConformerGenerationBlock(),
            MoleculeAlignBlock(query="reference.sdf"),
            MoleculeDockBlock(receptor="protein.pdb"),
            MoleculeSinkBlock()
        )
        ```

    Mutable Parameters:
        - w_gauss1: Vinardo Gaussian attractive term weight.
        - w_repulsion: Vinardo repulsion term weight.
        - w_hydrophobic: Vinardo hydrophobic term weight.
        - w_hbond: Vinardo hydrogen bond term weight.
        - w_rot: Torsional entropy divisor weight (0=pure Vinardo, 0.02=smina default).
        - w_ec: Weight for electrostatic complementarity term (0 = disabled).
        - n_starts: Number of L-BFGS-B restarts. 1 = local minimize from the
            input pose only. For blind docking (with site_reference), use
            1+2^k for ideal Sobol balance: 3, 5, 9, 17, 33, 65. Row 0 always
            minimizes from the aligned pose; rows 1+ sample the binding site box.
        - max_iterations: Maximum L-BFGS-B iterations per restart.
        - box_size: Translation search box half-width in Angstroms (default 5.0).
            Centred on site_reference centroid when provided, otherwise on the
            input conformer position.
        - rigid: If True, only rigid-body optimization (no torsions).
    """

    def __init__(
        self,
        score_components: bool = True,
        constraint_weight: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the molecular docking block.

        Args:
            score_components: If True (default), write per-term weighted score
                components as SDF properties on each docked molecule.
            constraint_weight: Penalty weight in kcal/mol/Å² applied to atoms
                matched by ``constraint_smarts``. 0 = disabled. Weight 100
                confines atoms to ~0.1 Å RMSD from their input pose.
            **kwargs: Passed to ``set_inputs``. Accepts ``receptor`` (file path),
                ``constraint_smarts`` (SMARTS string selecting constrained atoms),
                and any mutable parameter by name (``n_starts``, ``max_iterations``,
                ``box_size``, ``rigid``, score weights, ``w_ec``).

        Raises:
            ValueError: If constraint_smarts is invalid or contains explicit H
                (validated on first docking call).
        """
        super().__init__(
            name="MoleculeDock",
            input_files=["receptor", "site_reference"],
            input_text=["constraint_smarts"],
        )
        self._score_components = score_components
        self._constraint_weight = constraint_weight
        self._constraint_smarts_mol: Chem.Mol | None = None  # compiled lazily

        # Register mutable parameters
        self.mutable(
            # Vinardo score weights
            Continuous("w_gauss1", -0.045, -0.065, -0.025),
            Continuous("w_repulsion", 0.8, 0.8, 1.2),
            Continuous("w_hydrophobic", -0.035, -0.065, -0.015),
            Continuous("w_hbond", -0.6, -0.8, -0.4),
            Continuous("w_rot", 0.02, 0.0, 0.04),
            # Electrostatic complementarity
            Continuous("w_ec", 0.0, 0.0, 5.0),
            # Pose search
            Integer("n_starts", 17, 1, 33),
            Integer("max_iterations", 200, 0, 300),
            Continuous("box_size", 5.0, 0.5, 10.0),
            Categorical("rigid", False, [True, False]),
        )
        self.set_inputs(**kwargs)

        # Lazy-loaded protein scoring components
        self._protein_coords: np.ndarray | None = None
        self._protein_typing: AtomTyping | None = None
        self._protein_ec_coords: np.ndarray | None = None
        self._protein_ec_charges: np.ndarray | None = None

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

    def _load_site_reference(self) -> np.ndarray | None:
        """Compute binding site centroid from the site_reference reference file.

        Reads the first molecule with 3D coordinates from the ``site_reference``
        input file and returns its heavy-atom centroid. Supports .sdf, .mol2,
        and any other format handled by ``read_molecules``.

        Returns:
            (3,) centroid array, or None if no site_reference file is configured.

        Raises:
            FileNotFoundError: If the site_reference file does not exist.
            ValueError: If no molecule with 3D coordinates is found in the file.
        """
        site_path = self.input_files.get("site_reference")
        if site_path is None or str(site_path) == ".":
            return None

        site_path = Path(site_path)
        if not site_path.exists():
            raise FileNotFoundError(f"site_reference file does not exist: {site_path}")

        for mol in read_molecules(site_path, wrap=False):
            if mol is None or mol.GetNumConformers() == 0:
                continue
            heavy = Chem.RemoveAllHs(mol)
            coords = np.array(heavy.GetConformer().GetPositions())
            centroid = np.mean(coords, axis=0)
            logger.debug(
                "site_reference loaded from %s: centroid=(%.2f, %.2f, %.2f)",
                site_path.name,
                *centroid,
            )
            return centroid

        raise ValueError(
            f"No molecule with 3D coordinates found in site_reference file: {site_path}"
        )

    def _load_receptor(self) -> None:
        """Load and validate receptor from a PDB file.

        Raises:
            FileNotFoundError: If the receptor PDB file does not exist.
            ValueError: If no valid reference molecules with 3D conformers found.
        """
        receptor_path = Path(self.input_files["receptor"])
        if not receptor_path.exists():
            raise FileNotFoundError(f"Receptor path does not exist: {receptor_path}.")

        mol = Chem.MolFromPDBFile(str(receptor_path))

        # Prepare receptor with explicit H for EC scoring
        mol_with_h = Chem.AddHs(mol, addCoords=True)
        AllChem.ComputeGasteigerCharges(mol_with_h)
        self._protein_ec_coords = np.array(mol_with_h.GetConformer().GetPositions())
        self._protein_ec_charges = compute_gasteiger_charges(mol_with_h)

        # United atom for empirical scoring
        mol = Chem.RemoveAllHs(mol)

        if not self._has_3d_conformer(mol):
            raise ValueError(f"Receptor {receptor_path} does not have a 3D conformer.")

        protein_conf = mol.GetConformer()
        self._protein_coords = np.array(protein_conf.GetPositions())
        self._protein_typing = get_atom_typing(mol)

    def _prune_to_single_conformer(self, mol: Chem.Mol) -> Chem.Mol:
        """Reduce molecule to single conformer for docking.

        Args:
            mol: RDKit Mol object, possibly with multiple conformers.

        Returns:
            Molecule with at most one conformer.
        """
        num_confs = mol.GetNumConformers()
        if num_confs <= 1:
            return mol
        else:
            logger.info(
                f"Molecule has {num_confs} conformers. Docking only conformer 0."
            )
            conf = mol.GetConformer()
            new_mol = Chem.Mol(mol)
            new_mol.RemoveAllConformers()
            new_mol.AddConformer(Chem.Conformer(conf), assignId=True)
            return new_mol

    def _forward(self, mol: Chem.Mol) -> Chem.Mol | None:
        """Dock a ligand molecule into the receptor binding site.

        Args:
            mol: Ligand RDKit Mol with 3D conformer.

        Returns:
            Docked molecule with optimized pose and docking properties,
            or None if docking fails or input lacks 3D conformers.
        """
        if not self._has_3d_conformer(mol):
            logger.debug(
                "Input molecule does not have 3D conformers. Skipping docking."
            )
            return None

        mol = self._prune_to_single_conformer(mol)

        if self._protein_coords is None:
            self._load_receptor()
        assert isinstance(self._protein_coords, np.ndarray)
        assert isinstance(self._protein_typing, AtomTyping)

        # Compile constraint SMARTS lazily on first call
        smarts_str = self.input_text.get("constraint_smarts", "").strip()
        if smarts_str and self._constraint_smarts_mol is None:
            smarts_mol = Chem.MolFromSmarts(smarts_str)
            if smarts_mol is None:
                raise ValueError(f"Invalid constraint_smarts: {smarts_str!r}")
            for atom in smarts_mol.GetAtoms():
                if atom.GetAtomicNum() == 1:
                    raise ValueError(
                        f"constraint_smarts {smarts_str!r} contains an explicit "
                        "hydrogen. Use e.g. [#6H] not [#6]-[H]."
                    )
            self._constraint_smarts_mol = smarts_mol

        # Resolve constraint SMARTS → heavy-atom indices for this molecule
        constrained_atoms: tuple[int, ...] = ()
        if self._constraint_smarts_mol is not None and self._constraint_weight > 0.0:
            ligand_heavy_pre = Chem.RemoveAllHs(mol)
            matches = ligand_heavy_pre.GetSubstructMatches(self._constraint_smarts_mol)
            if matches:
                constrained_atoms = tuple(
                    sorted({idx for match in matches for idx in match})
                )
                logger.debug(
                    "Constraint matched %d time(s), constraining %d atoms.",
                    len(matches),
                    len(constrained_atoms),
                )

        score_params = EmpiricalParams(
            w_gauss1=self.get_param("w_gauss1"),
            w_repulsion=self.get_param("w_repulsion"),
            w_hydrophobic=self.get_param("w_hydrophobic"),
            w_hbond=self.get_param("w_hbond"),
            w_rot=self.get_param("w_rot"),
        )
        box_size = self.get_param("box_size")
        rigid_only = self.get_param("rigid")
        site_center = self._load_site_reference()
        w_ec = self.get_param("w_ec")

        # Number of top rigid poses to pass to flexible refinement.
        # Fixed at 3: covers the main failure mode (best rigid basin doesn't
        # survive torsion relaxation) without 3× the flexible cost.
        # Could be made mutable if tuning shows benefit.
        N_FLEX_REFINE = 3

        if rigid_only:
            # Rigid-only mode: single-phase global search, no torsion DOFs.
            pose_params = PoseParams(
                max_iterations=self.get_param("max_iterations"),
                translation_bounds=(-box_size, box_size),
                optimize_torsions=False,
                n_starts=self.get_param("n_starts"),
                constrained_atom_indices=constrained_atoms,
                constraint_weight=self._constraint_weight,
            )
            result = optimize_pose_cached(
                mol,
                protein_coords=self._protein_coords,
                protein_typing=self._protein_typing,
                params=pose_params,
                score_params=score_params,
                site_center=site_center,
                protein_ec_coords=self._protein_ec_coords,
                protein_ec_charges=self._protein_ec_charges,
                w_ec=w_ec,
            )
        else:
            # Phase 1: rigid global search — cheap, covers the full box.
            rigid_params = PoseParams(
                max_iterations=self.get_param("max_iterations"),
                translation_bounds=(-box_size, box_size),
                optimize_torsions=False,
                n_starts=self.get_param("n_starts"),
            )
            top_rigid = _rigid_topk(
                mol,
                protein_coords=self._protein_coords,
                protein_typing=self._protein_typing,
                params=rigid_params,
                score_params=score_params,
                site_center=site_center,
                n_top=N_FLEX_REFINE,
            )

            # Phase 2: flexible local refinement from each top rigid pose.
            # Small box — molecule is already in the right basin.
            flex_params = PoseParams(
                max_iterations=self.get_param("max_iterations"),
                translation_bounds=(-2.0, 2.0),
                optimize_torsions=True,
                n_starts=1,
                constrained_atom_indices=constrained_atoms,
                constraint_weight=self._constraint_weight,
            )

            _protein_coords = self._protein_coords
            _protein_typing = self._protein_typing
            assert isinstance(_protein_coords, np.ndarray)
            assert isinstance(_protein_typing, AtomTyping)

            def _flex(rigid_mol: Chem.Mol) -> OptimizationResult:
                return optimize_pose_cached(
                    rigid_mol,
                    protein_coords=_protein_coords,
                    protein_typing=_protein_typing,
                    params=flex_params,
                    score_params=score_params,
                    site_center=None,
                    protein_ec_coords=self._protein_ec_coords,
                    protein_ec_charges=self._protein_ec_charges,
                    w_ec=w_ec,
                )

            result = _flex(top_rigid[0][1])
            for _, rigid_mol in top_rigid[1:]:
                candidate = _flex(rigid_mol)
                if candidate.score < result.score:
                    result = candidate

            # Preserve the original input pose's score as the baseline.
            import dataclasses

            ligand_heavy_init = Chem.RemoveAllHs(mol)
            initial_score = empirical_score_cached(
                ligand_heavy_init,
                self._protein_coords,
                self._protein_typing,
                params=score_params,
            ).total
            result = dataclasses.replace(result, initial_score=initial_score)

        # Set properties
        result.mol.SetDoubleProp("docking_initial_pose_score", result.initial_score)
        result.mol.SetDoubleProp("docking_score", result.score)
        result.mol.SetDoubleProp("docking_empirical", result.score + w_ec * result.ec)
        result.mol.SetDoubleProp("docking_ec", result.ec)
        result.mol.SetBoolProp("docking_converged", result.converged)

        # Per-term score components (reporting path)
        if self._score_components:
            assert isinstance(self._protein_coords, np.ndarray)
            assert isinstance(self._protein_typing, AtomTyping)
            ligand_heavy = Chem.RemoveAllHs(result.mol)
            comps = empirical_score_cached(
                ligand_heavy,
                self._protein_coords,
                self._protein_typing,
                params=score_params,
            )
            result.mol.SetDoubleProp("docking_gauss1", comps.gauss1)
            result.mol.SetDoubleProp("docking_repulsion", comps.repulsion)
            result.mol.SetDoubleProp("docking_hydrophobic", comps.hydrophobic)
            result.mol.SetDoubleProp("docking_hbond", comps.hbond)
            result.mol.SetDoubleProp("docking_n_rot", comps.n_rot * comps.w_rot)
            result.mol.SetProp(
                "docking_scoring_function",
                f"(gauss1) {score_params.w_gauss1:.3f} "
                f"(repulsion) {score_params.w_repulsion:.3f} "
                f"(hydrophobic) {score_params.w_hydrophobic:.3f} "
                f"(hbond) {score_params.w_hbond:.3f} "
                f"(w_rot) {score_params.w_rot:.3f} "
                f"(w_ec) {w_ec:.3f}",
            )

        return result.mol

    def check_output(self, arg: Any) -> bool:
        """Validate that output molecule has docking properties.

        Args:
            arg: Output molecule to validate.

        Returns:
            True if the molecule has valid docking output, False otherwise.
        """
        if not isinstance(arg, Chem.Mol):
            logger.info(f"Docking failed: output is of type {type(arg)}")
            return False

        if arg.GetNumConformers() == 0:
            logger.info("Docking failed: output molecule has no conformers")
            return False

        required = (
            "docking_initial_pose_score",
            "docking_score",
            "docking_ec",
            "docking_converged",
        )
        for key in required:
            if not arg.HasProp(key):
                logger.info(f"Docking failed: missing {key} property")
                return False

        return True
