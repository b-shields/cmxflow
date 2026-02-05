"""Molecular docking block for cmxflow workflows.

This module provides a MoleculeBlock implementation for docking ligands
into protein binding sites using the Vinardo scoring function and
rigid-body/torsional pose optimization.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

from cmxflow.operators.base import MoleculeBlock
from cmxflow.operators.dock.pose import (
    PoseParams,
    optimize_pose_cached,
)
from cmxflow.operators.dock.score import (
    AtomTyping,
    VinardoParams,
    get_atom_typing,
    vinardo_score_cached,
)
from cmxflow.parameter import (
    Categorical,
    Continuous,
    Integer,
)

logger = logging.getLogger(__name__)


class MoleculeDockBlock(MoleculeBlock):
    """MoleculeBlock for docking ligands into protein binding sites.

    Performs pose optimization using Vinardo scoring with configurable
    parameters. Supports both rigid-body and flexible (torsional) docking.

    The block requires a receptor PDB file specified via input_files["receptor"].
    Input molecules must have pre-generated 3D conformers.

    Attributes:
        input_files: Dictionary containing "receptor" key for protein PDB file.

    Mutable Parameters:
        w_gauss1: Vinardo Gaussian attractive term weight.
        w_repulsion: Vinardo repulsion term weight.
        w_hydrophobic: Vinardo hydrophobic term weight.
        w_hbond: Vinardo hydrogen bond term weight.
        max_iterations: Maximum optimization iterations.
        box_size: Translation search box size in Angstroms.
        rigid: If True, only rigid-body optimization (no torsions).

    Output Properties:
        docking_initial_pose_score: Score before optimization.
        docking_score: Final optimized score.
        docking_converged: Whether optimization converged.

    Example:
        >>> block = MoleculeDockBlock()
        >>> block.input_files["receptor"] = "protein.pdb"
        >>> docked_mol = block.forward(ligand_mol)
        >>> print(docked_mol.GetDoubleProp("docking_score"))
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the molecular docking block."""
        super().__init__(name="MoleculeDock", input_files=["receptor"])

        # Register mutable parameters
        self.mutable(
            # Vinardo score weights
            Continuous("w_gauss1", -0.045, -0.065, -0.025),
            Continuous("w_repulsion", 0.8, 0.8, 1.2),
            Continuous("w_hydrophobic", -0.035, -0.065, -0.015),
            Continuous("w_hbond", -0.6, -0.8, -0.4),
            # Pose search
            Integer("max_iterations", 200, 100, 400),
            Continuous("box_size", 1.5, 0.5, 2.0),
            Categorical("rigid", False, [True, False]),
        )
        self.set_inputs(**kwargs)

        # Lazy-loaded protein scoring components
        self._protein_coords: np.ndarray | None = None
        self._protein_typing: AtomTyping | None = None

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
        mol = Chem.RemoveHs(mol)  # Unitd atom scoring functions

        if not self._has_3d_conformer(mol):
            raise ValueError(f"Receptor {receptor_path} does not have a 3D conformer.")

        protein_conf = mol.GetConformer()
        self._protein_coords = np.array(protein_conf.GetPositions())
        self._protein_typing = get_atom_typing(mol)

    def _prune_to_single_conformer(self, mol: Chem.Mol) -> Chem.Mol:
        """Reduce molecule to single conformer for docking.

        If the molecule has multiple conformers, only the first (index 0)
        is retained. This ensures consistent docking behavior.

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

        Performs pose optimization using Vinardo scoring. The input molecule
        must have 3D coordinates. If multiple conformers exist, only the
        first is used.

        Args:
            mol: Ligand RDKit Mol with 3D conformer.

        Returns:
            Docked molecule with optimized pose and docking properties,
            or None if docking fails or input lacks 3D conformers.
        """
        # Validate input molecule has 3D conformers
        if not self._has_3d_conformer(mol):
            logger.debug(
                "Input molecule does not have 3D conformers. Skipping docking."
            )
            return None

        # If there are more than one conformer only take the first
        mol = self._prune_to_single_conformer(mol)

        # Require receptor caches
        if self._protein_coords is None:
            self._load_receptor()
        assert isinstance(self._protein_coords, np.ndarray)
        assert isinstance(self._protein_typing, AtomTyping)

        # Set score and pose parameters
        score_params = VinardoParams(
            w_gauss1=self.get_param("w_gauss1"),
            w_repulsion=self.get_param("w_repulsion"),
            w_hydrophobic=self.get_param("w_hydrophobic"),
            w_hbond=self.get_param("w_hbond"),
        )
        box_size = self.get_param("box_size")
        pose_params = PoseParams(
            max_iterations=self.get_param("max_iterations"),
            translation_bounds=(-box_size, box_size),
            optimize_torsions=not self.get_param("rigid"),
        )

        # Dock
        result = optimize_pose_cached(
            mol,
            protein_coords=self._protein_coords,
            protein_typing=self._protein_typing,
            scoring_fn=vinardo_score_cached,
            scoring_fn_params=score_params,
            params=pose_params,
        )

        # Set properties
        result.mol.SetDoubleProp("docking_initial_pose_score", result.initial_score)
        result.mol.SetDoubleProp("docking_score", result.score)
        result.mol.SetBoolProp("docking_converged", result.converged)

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

        required = ("docking_initial_pose_score", "docking_score", "docking_converged")
        for key in required:
            if not arg.HasProp(key):
                logger.info(f"Docking failed: missing {key} property")
                return False

        return True
