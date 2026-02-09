"""Molecular docking operators.

This module provides scoring functions and pose optimization for
protein-ligand docking.

Scoring Functions:
    vinardo_score: Vinardo empirical scoring function
    vinardo_score_cached: Vinardo scoring function (cached protein data)
    VinardoParams: Parameters for Vinardo scoring
    get_scoring_function: Factory for scoring functions
    get_atom_typing: Get atom classification for a molecule
    AtomTyping: Atom classification data

Electrostatic Complementarity:
    electrostatic_complementarity: EC score for ligand-protein complex
    compute_esp_at_points: ESP computation at query points
    compute_gasteiger_charges: Gasteiger partial charges
    fibonacci_sphere: Uniform sphere point generation
    generate_sas_points: Solvent-accessible surface points

Pose Optimization:
    optimize_pose: General pose optimization
    optimize_pose_rigid: Rigid-body only optimization
    optimize_pose_flexible: Rigid + torsion optimization
    optimize_pose_cached: General pose optimization (cached input)
    PoseParams: Optimization parameters
    OptimizationResult: Optimization results

Example:
    >>> from rdkit import Chem
    >>> from rdkit.Chem import AllChem
    >>> from cmxflow.operators.dock import vinardo_score, optimize_pose_rigid
    >>>
    >>> # Create ligand with 3D coords
    >>> ligand = Chem.MolFromSmiles("CCO")
    >>> ligand = Chem.AddHs(ligand)
    >>> AllChem.EmbedMolecule(ligand)
    >>>
    >>> # Create protein (simplified example)
    >>> protein = Chem.MolFromSmiles("c1ccccc1O")
    >>> protein = Chem.AddHs(protein)
    >>> AllChem.EmbedMolecule(protein)
    >>>
    >>> # Score the pose
    >>> score = vinardo_score(ligand, protein)
    >>>
    >>> # Optimize pose
    >>> result = optimize_pose_rigid(ligand, protein)
    >>> print(f"Score improved: {result.initial_score} -> {result.score}")
"""

from cmxflow.operators.dock.dock import MoleculeDockBlock
from cmxflow.operators.dock.ec import (
    compute_esp_at_points,
    compute_gasteiger_charges,
    electrostatic_complementarity,
    fibonacci_sphere,
    generate_sas_points,
)
from cmxflow.operators.dock.pose import (
    OptimizationResult,
    PoseParams,
    apply_rigid_transform,
    get_rotatable_bonds,
    optimize_pose,
    optimize_pose_cached,
    optimize_pose_flexible,
    optimize_pose_rigid,
)
from cmxflow.operators.dock.score import (
    AtomTyping,
    VinardoParams,
    ec_score_cached,
    get_atom_typing,
    get_scoring_function,
    vinardo_score,
    vinardo_score_cached,
)

__all__ = [
    # Score
    "AtomTyping",
    "VinardoParams",
    "ec_score_cached",
    "get_atom_typing",
    "get_scoring_function",
    "vinardo_score",
    "vinardo_score_cached",
    # Electrostatic Complementarity
    "electrostatic_complementarity",
    "compute_esp_at_points",
    "compute_gasteiger_charges",
    "fibonacci_sphere",
    "generate_sas_points",
    # Pose
    "OptimizationResult",
    "PoseParams",
    "apply_rigid_transform",
    "get_rotatable_bonds",
    "optimize_pose",
    "optimize_pose_flexible",
    "optimize_pose_rigid",
    "optimize_pose_cached",
    # Dock
    "MoleculeDockBlock",
]
