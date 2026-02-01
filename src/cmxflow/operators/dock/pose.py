"""Pose optimization for molecular docking.

This module provides functions for optimizing ligand pose in a protein
binding site using rigid-body transformations and torsion angle optimization.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, TypeAlias

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from cmxflow.operators.dock.score import (
    AtomTyping,
    VinardoParams,
    vinardo_score,
    vinardo_score_cached,
)

logger = logging.getLogger(__name__)

# Type aliases
Coords: TypeAlias = NDArray[np.floating]


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class PoseParams:
    """Pose optimization parameters.

    Attributes:
        max_iterations: Maximum optimization iterations.
        tolerance: Convergence tolerance for gradient.
        translation_bounds: (min, max) bounds for translation in each axis.
        optimize_torsions: Whether to optimize rotatable bond torsions.
        max_torsion_change: Maximum torsion angle change in degrees.
    """

    max_iterations: int = 100
    tolerance: float = 1e-4
    translation_bounds: tuple[float, float] = (-10.0, 10.0)
    optimize_torsions: bool = True
    max_torsion_change: float = 180.0


@dataclass
class OptimizationResult:
    """Result of pose optimization.

    Attributes:
        mol: Optimized molecule with updated coordinates.
        score: Final score after optimization.
        initial_score: Score before optimization.
        translation: Applied translation vector (x, y, z).
        rotation: Applied rotation as scipy Rotation object.
        torsion_changes: Dict mapping (j, k) bond indices to angle changes.
        converged: Whether optimization converged.
        n_iterations: Number of iterations performed.
    """

    mol: Chem.Mol
    score: float
    initial_score: float
    translation: NDArray[np.floating]
    rotation: Rotation
    torsion_changes: dict[tuple[int, int], float] = field(default_factory=dict)
    converged: bool = False
    n_iterations: int = 0


# =============================================================================
# Coordinate Transformations
# =============================================================================


def get_molecule_centroid(mol: Chem.Mol, conf_id: int = 0) -> NDArray[np.floating]:
    """Compute molecule centroid (center of geometry).

    Args:
        mol: RDKit Mol with 3D conformer.
        conf_id: Conformer ID.

    Returns:
        Centroid as (x, y, z) array.
    """
    conf = mol.GetConformer(conf_id)
    coords = np.array(conf.GetPositions())
    centroid: NDArray[np.floating] = np.mean(coords, axis=0)
    return centroid


def apply_rigid_transform(
    mol: Chem.Mol,
    translation: NDArray[np.floating],
    rotation: Rotation,
    conf_id: int = 0,
    center: NDArray[np.floating] | None = None,
) -> Chem.Mol:
    """Apply rigid body transformation to molecule.

    Rotation is applied about the specified center (or molecule centroid),
    followed by translation.

    Args:
        mol: RDKit Mol with 3D conformer.
        translation: Translation vector (x, y, z).
        rotation: Rotation as scipy Rotation object.
        conf_id: Conformer ID to transform.
        center: Center of rotation. If None, uses centroid.

    Returns:
        New molecule with transformed coordinates.
    """
    mol_copy = Chem.Mol(mol)
    conf = mol_copy.GetConformer(conf_id)
    coords = np.array(conf.GetPositions())

    if center is None:
        center = np.mean(coords, axis=0)

    # Center, rotate, uncenter, then translate
    centered = coords - center
    rotated = rotation.apply(centered)
    transformed = rotated + center + translation

    # Update conformer coordinates
    for i, pos in enumerate(transformed):
        conf.SetAtomPosition(i, tuple(pos))

    return mol_copy


# =============================================================================
# Torsion Handling
# =============================================================================


def get_rotatable_bonds(mol: Chem.Mol) -> list[tuple[int, int, int, int]]:
    """Get rotatable bond dihedral atom indices.

    Returns 4-tuples of atom indices (i, j, k, l) defining each rotatable
    bond's dihedral angle where j-k is the rotatable bond.

    Args:
        mol: RDKit Mol object.

    Returns:
        List of (i, j, k, l) atom index tuples for each rotatable bond.
    """
    rotatable_smarts = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
    if rotatable_smarts is None:
        return []

    matches = mol.GetSubstructMatches(rotatable_smarts)

    dihedrals: list[tuple[int, int, int, int]] = []
    for j, k in matches:
        # Find atoms to define dihedral: need neighbors of j and k
        j_neighbors = [
            a.GetIdx() for a in mol.GetAtomWithIdx(j).GetNeighbors() if a.GetIdx() != k
        ]
        k_neighbors = [
            a.GetIdx() for a in mol.GetAtomWithIdx(k).GetNeighbors() if a.GetIdx() != j
        ]

        if j_neighbors and k_neighbors:
            idx_i = j_neighbors[0]
            idx_l = k_neighbors[0]
            dihedrals.append((idx_i, j, k, idx_l))

    return dihedrals


def get_dihedral_angle(
    mol: Chem.Mol,
    idx_i: int,
    idx_j: int,
    idx_k: int,
    idx_l: int,
    conf_id: int = 0,
) -> float:
    """Get dihedral angle in degrees.

    Args:
        mol: RDKit Mol with 3D conformer.
        idx_i, idx_j, idx_k, idx_l: Atom indices defining the dihedral.
        conf_id: Conformer ID.

    Returns:
        Dihedral angle in degrees.
    """
    conf = mol.GetConformer(conf_id)
    return float(rdMolTransforms.GetDihedralDeg(conf, idx_i, idx_j, idx_k, idx_l))


def apply_torsion_changes(
    mol: Chem.Mol,
    torsion_angles: dict[tuple[int, int, int, int], float],
    conf_id: int = 0,
) -> Chem.Mol:
    """Apply torsion angle changes to molecule.

    Args:
        mol: RDKit Mol with 3D conformer.
        torsion_angles: Dict mapping (i, j, k, l) to new angle in degrees.
        conf_id: Conformer ID to modify.

    Returns:
        New molecule with updated torsion angles.
    """
    mol_copy = Chem.Mol(mol)
    conf = mol_copy.GetConformer(conf_id)

    for (idx_i, idx_j, idx_k, idx_l), angle in torsion_angles.items():
        rdMolTransforms.SetDihedralDeg(conf, idx_i, idx_j, idx_k, idx_l, angle)

    return mol_copy


# =============================================================================
# Optimization Helpers
# =============================================================================


def _pack_pose_vector(
    translation: NDArray[np.floating],
    rotation_rotvec: NDArray[np.floating],
    torsions: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    """Pack pose parameters into optimization vector.

    Args:
        translation: Translation (x, y, z).
        rotation_rotvec: Rotation as rotation vector (3 values).
        torsions: Torsion angle changes in degrees.

    Returns:
        Flattened parameter vector.
    """
    if torsions is None or len(torsions) == 0:
        return np.concatenate([translation, rotation_rotvec])
    return np.concatenate([translation, rotation_rotvec, torsions])


def _unpack_pose_vector(
    x: NDArray[np.floating],
    n_torsions: int = 0,
) -> tuple[NDArray[np.floating], Rotation, NDArray[np.floating] | None]:
    """Unpack optimization vector to pose parameters.

    Args:
        x: Flattened parameter vector.
        n_torsions: Number of torsion angles.

    Returns:
        Tuple of (translation, rotation, torsion_deltas).
    """
    translation = x[:3]
    rotation = Rotation.from_rotvec(x[3:6])

    if n_torsions > 0:
        torsions = x[6:]
        return translation, rotation, torsions
    return translation, rotation, None


def _create_objective(
    ligand_mol: Chem.Mol,
    protein_mol: Chem.Mol,
    scoring_fn: Callable[[Chem.Mol, Chem.Mol, int, int], float],
    rotatable_dihedrals: list[tuple[int, int, int, int]],
    initial_torsions: NDArray[np.floating],
    ligand_conf_id: int,
    protein_conf_id: int,
    centroid: NDArray[np.floating],
) -> Callable[[NDArray[np.floating]], float]:
    """Create objective function for optimization.

    Args:
        ligand_mol: Initial ligand molecule.
        protein_mol: Protein molecule (fixed).
        scoring_fn: Scoring function.
        rotatable_dihedrals: List of rotatable dihedral definitions.
        initial_torsions: Initial torsion angles.
        ligand_conf_id: Ligand conformer ID.
        protein_conf_id: Protein conformer ID.
        centroid: Ligand centroid for rotation.

    Returns:
        Objective function that takes parameter vector and returns score.
    """
    n_torsions = len(rotatable_dihedrals)

    def objective(x: NDArray[np.floating]) -> float:
        translation, rotation, torsion_deltas = _unpack_pose_vector(x, n_torsions)

        # Apply rigid transform
        transformed = apply_rigid_transform(
            ligand_mol, translation, rotation, ligand_conf_id, centroid
        )

        # Apply torsion changes if requested
        if torsion_deltas is not None and len(rotatable_dihedrals) > 0:
            new_torsions = initial_torsions + torsion_deltas
            torsion_dict = {
                dihedral: angle
                for dihedral, angle in zip(rotatable_dihedrals, new_torsions)
            }
            transformed = apply_torsion_changes(
                transformed, torsion_dict, ligand_conf_id
            )

        return scoring_fn(transformed, protein_mol, ligand_conf_id, protein_conf_id)

    return objective


# =============================================================================
# Main Optimization Functions
# =============================================================================


def optimize_pose(
    ligand_mol: Chem.Mol,
    protein_mol: Chem.Mol,
    scoring_fn: Callable[[Chem.Mol, Chem.Mol, int, int], float] | None = None,
    params: PoseParams | None = None,
    ligand_conf_id: int = 0,
    protein_conf_id: int = 0,
) -> OptimizationResult:
    """Optimize ligand pose in protein binding site.

    Performs rigid-body optimization (translation + rotation) and
    optionally flexible optimization (torsion angles) using L-BFGS-B.

    Args:
        ligand_mol: Ligand RDKit Mol with 3D coordinates.
        protein_mol: Protein RDKit Mol with 3D coordinates.
        scoring_fn: Scoring function. If None, uses Vinardo.
        params: Optimization parameters. If None, uses defaults.
        ligand_conf_id: Ligand conformer ID to optimize.
        protein_conf_id: Protein conformer ID (fixed).

    Returns:
        OptimizationResult with optimized molecule and metadata.
    """
    if scoring_fn is None:
        scoring_fn = vinardo_score
    if params is None:
        params = PoseParams()

    # Get rotatable bonds if flexible docking
    rotatable_dihedrals: list[tuple[int, int, int, int]] = []
    initial_torsions: NDArray[np.floating] = np.array([])

    if params.optimize_torsions:
        rotatable_dihedrals = get_rotatable_bonds(ligand_mol)
        if rotatable_dihedrals:
            initial_torsions = np.array(
                [
                    get_dihedral_angle(ligand_mol, *dihedral, ligand_conf_id)
                    for dihedral in rotatable_dihedrals
                ]
            )

    n_torsions = len(rotatable_dihedrals)

    # Initial pose: no transformation
    x0 = _pack_pose_vector(
        translation=np.zeros(3),
        rotation_rotvec=np.zeros(3),
        torsions=np.zeros(n_torsions) if n_torsions > 0 else None,
    )

    # Compute initial score
    initial_score = scoring_fn(ligand_mol, protein_mol, ligand_conf_id, protein_conf_id)

    # Get ligand centroid for rotation
    centroid = get_molecule_centroid(ligand_mol, ligand_conf_id)

    # Set up bounds
    trans_bounds = [params.translation_bounds] * 3
    rot_bounds = [(-np.pi, np.pi)] * 3
    torsion_bounds = [
        (-params.max_torsion_change, params.max_torsion_change)
    ] * n_torsions
    bounds = trans_bounds + rot_bounds + torsion_bounds

    # Create objective
    objective = _create_objective(
        ligand_mol,
        protein_mol,
        scoring_fn,
        rotatable_dihedrals,
        initial_torsions,
        ligand_conf_id,
        protein_conf_id,
        centroid,
    )

    # Optimize
    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": params.max_iterations,
            "gtol": params.tolerance,
        },
    )

    # Unpack result
    translation, rotation, torsion_deltas = _unpack_pose_vector(result.x, n_torsions)

    # Apply final transformation
    optimized_mol = apply_rigid_transform(
        ligand_mol, translation, rotation, ligand_conf_id, centroid
    )

    torsion_changes: dict[tuple[int, int], float] = {}
    if torsion_deltas is not None and len(rotatable_dihedrals) > 0:
        new_torsions = initial_torsions + torsion_deltas
        torsion_dict = {
            dihedral: angle
            for dihedral, angle in zip(rotatable_dihedrals, new_torsions)
        }
        optimized_mol = apply_torsion_changes(
            optimized_mol, torsion_dict, ligand_conf_id
        )
        torsion_changes = {
            (dihedral[1], dihedral[2]): float(delta)
            for dihedral, delta in zip(rotatable_dihedrals, torsion_deltas)
        }

    return OptimizationResult(
        mol=optimized_mol,
        score=float(result.fun),
        initial_score=initial_score,
        translation=translation,
        rotation=rotation,
        torsion_changes=torsion_changes,
        converged=result.success,
        n_iterations=result.nit,
    )


def optimize_pose_rigid(
    ligand_mol: Chem.Mol,
    protein_mol: Chem.Mol,
    scoring_fn: Callable[[Chem.Mol, Chem.Mol, int, int], float] | None = None,
    max_iterations: int = 100,
    ligand_conf_id: int = 0,
    protein_conf_id: int = 0,
) -> OptimizationResult:
    """Optimize ligand pose with rigid-body transformation only.

    Convenience wrapper for optimize_pose with torsion optimization disabled.

    Args:
        ligand_mol: Ligand molecule.
        protein_mol: Protein molecule.
        scoring_fn: Scoring function.
        max_iterations: Maximum iterations.
        ligand_conf_id: Ligand conformer ID.
        protein_conf_id: Protein conformer ID.

    Returns:
        OptimizationResult with optimized rigid pose.
    """
    params = PoseParams(
        max_iterations=max_iterations,
        optimize_torsions=False,
    )
    return optimize_pose(
        ligand_mol,
        protein_mol,
        scoring_fn,
        params,
        ligand_conf_id,
        protein_conf_id,
    )


def optimize_pose_flexible(
    ligand_mol: Chem.Mol,
    protein_mol: Chem.Mol,
    scoring_fn: Callable[[Chem.Mol, Chem.Mol, int, int], float] | None = None,
    max_iterations: int = 200,
    max_torsion_change: float = 30.0,
    ligand_conf_id: int = 0,
    protein_conf_id: int = 0,
) -> OptimizationResult:
    """Optimize ligand pose with rigid-body and torsion flexibility.

    Convenience wrapper for optimize_pose with torsion optimization enabled.

    Args:
        ligand_mol: Ligand molecule.
        protein_mol: Protein molecule.
        scoring_fn: Scoring function.
        max_iterations: Maximum iterations.
        max_torsion_change: Maximum torsion change per bond in degrees.
        ligand_conf_id: Ligand conformer ID.
        protein_conf_id: Protein conformer ID.

    Returns:
        OptimizationResult with optimized flexible pose.
    """
    params = PoseParams(
        max_iterations=max_iterations,
        optimize_torsions=True,
        max_torsion_change=max_torsion_change,
    )
    return optimize_pose(
        ligand_mol,
        protein_mol,
        scoring_fn,
        params,
        ligand_conf_id,
        protein_conf_id,
    )


# =============================================================================
# Faster But Less User Friendly Optimization Functions
# =============================================================================


def _create_objective_cached(
    ligand_mol: Chem.Mol,
    protein_coords: np.ndarray,
    protein_typing: AtomTyping,
    scoring_fn: Callable[[Chem.Mol, np.ndarray, AtomTyping, int, VinardoParams], float],
    scoring_fn_params: VinardoParams,
    rotatable_dihedrals: list[tuple[int, int, int, int]],
    initial_torsions: NDArray[np.floating],
    ligand_conf_id: int,
    centroid: NDArray[np.floating],
) -> Callable[[NDArray[np.floating]], float]:
    """Create objective function for optimization with cached protein data.

    This is an internal helper that creates a closure for the optimizer.
    The returned function applies transformations to the ligand and scores
    it against the pre-computed protein data.

    Args:
        ligand_mol: Initial ligand molecule.
        protein_coords: Pre-computed protein atom 3D coordinates.
        protein_typing: Pre-computed protein atom typing.
        scoring_fn: Cached scoring function accepting pre-computed protein data.
        scoring_fn_params: Parameters for the scoring function.
        rotatable_dihedrals: List of (i, j, k, l) tuples for rotatable bonds.
        initial_torsions: Initial torsion angles in degrees.
        ligand_conf_id: Ligand conformer ID to transform.
        centroid: Ligand centroid for rotation center.

    Returns:
        Objective function that takes a parameter vector and returns a score.
    """
    n_torsions = len(rotatable_dihedrals)

    def objective(x: NDArray[np.floating]) -> float:
        translation, rotation, torsion_deltas = _unpack_pose_vector(x, n_torsions)

        # Apply rigid transform
        transformed = apply_rigid_transform(
            ligand_mol, translation, rotation, ligand_conf_id, centroid
        )

        # Apply torsion changes if requested
        if torsion_deltas is not None and len(rotatable_dihedrals) > 0:
            new_torsions = initial_torsions + torsion_deltas
            torsion_dict = {
                dihedral: angle
                for dihedral, angle in zip(rotatable_dihedrals, new_torsions)
            }
            transformed = apply_torsion_changes(
                transformed, torsion_dict, ligand_conf_id
            )

        return scoring_fn(
            transformed,
            protein_coords,
            protein_typing,
            ligand_conf_id,
            scoring_fn_params,
        )

    return objective


def optimize_pose_cached(
    ligand_mol: Chem.Mol,
    protein_coords: np.ndarray,
    protein_typing: AtomTyping,
    scoring_fn: (
        Callable[[Chem.Mol, np.ndarray, AtomTyping, int, VinardoParams | None], float]
        | None
    ) = None,
    scoring_fn_params: VinardoParams | None = None,
    params: PoseParams | None = None,
    ligand_conf_id: int = 0,
) -> OptimizationResult:
    """Optimize ligand pose with pre-computed protein data.

    This is a performance-optimized version of optimize_pose() that accepts
    pre-computed protein coordinates and atom typing. Use this when optimizing
    multiple ligands against the same protein to avoid redundant computation.

    Performs rigid-body optimization (translation + rotation) and optionally
    flexible optimization (torsion angles) using L-BFGS-B.

    Args:
        ligand_mol: Ligand RDKit Mol with 3D coordinates.
        protein_coords: Pre-computed protein atom 3D coordinates as numpy
            array with shape (n_atoms, 3).
        protein_typing: Pre-computed protein atom typing from get_atom_typing().
        scoring_fn: Cached scoring function. If None, uses vinardo_score_cached.
        scoring_fn_params: Parameters for the scoring function.
        params: Optimization parameters. If None, uses defaults.
        ligand_conf_id: Ligand conformer ID to optimize.

    Returns:
        OptimizationResult with optimized molecule and metadata.

    Example:
        >>> protein_coords = np.array(protein.GetConformer().GetPositions())
        >>> protein_typing = get_atom_typing(protein)
        >>> for ligand in ligands:
        ...     result = optimize_pose_cached(
        ...         ligand, protein_coords, protein_typing
        ...     )
    """
    if scoring_fn is None:
        scoring_fn = vinardo_score_cached
    if scoring_fn_params is None:
        scoring_fn_params = VinardoParams()
    if params is None:
        params = PoseParams()

    # Get rotatable bonds if flexible docking
    rotatable_dihedrals: list[tuple[int, int, int, int]] = []
    initial_torsions: NDArray[np.floating] = np.array([])

    if params.optimize_torsions:
        rotatable_dihedrals = get_rotatable_bonds(ligand_mol)
        if rotatable_dihedrals:
            initial_torsions = np.array(
                [
                    get_dihedral_angle(ligand_mol, *dihedral, ligand_conf_id)
                    for dihedral in rotatable_dihedrals
                ]
            )

    n_torsions = len(rotatable_dihedrals)

    # Initial pose: no transformation
    x0 = _pack_pose_vector(
        translation=np.zeros(3),
        rotation_rotvec=np.zeros(3),
        torsions=np.zeros(n_torsions) if n_torsions > 0 else None,
    )

    # Compute initial score
    initial_score = scoring_fn(
        ligand_mol, protein_coords, protein_typing, ligand_conf_id, scoring_fn_params
    )

    # Get ligand centroid for rotation
    centroid = get_molecule_centroid(ligand_mol, ligand_conf_id)

    # Set up bounds
    trans_bounds = [params.translation_bounds] * 3
    rot_bounds = [(-np.pi, np.pi)] * 3
    torsion_bounds = [
        (-params.max_torsion_change, params.max_torsion_change)
    ] * n_torsions
    bounds = trans_bounds + rot_bounds + torsion_bounds

    # Create objective
    objective = _create_objective_cached(
        ligand_mol,
        protein_coords,
        protein_typing,
        scoring_fn,
        scoring_fn_params,
        rotatable_dihedrals,
        initial_torsions,
        ligand_conf_id,
        centroid,
    )

    # Optimize
    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": params.max_iterations,
            "gtol": params.tolerance,
        },
    )

    # Unpack result
    translation, rotation, torsion_deltas = _unpack_pose_vector(result.x, n_torsions)

    # Apply final transformation
    optimized_mol = apply_rigid_transform(
        ligand_mol, translation, rotation, ligand_conf_id, centroid
    )

    torsion_changes: dict[tuple[int, int], float] = {}
    if torsion_deltas is not None and len(rotatable_dihedrals) > 0:
        new_torsions = initial_torsions + torsion_deltas
        torsion_dict = {
            dihedral: angle
            for dihedral, angle in zip(rotatable_dihedrals, new_torsions)
        }
        optimized_mol = apply_torsion_changes(
            optimized_mol, torsion_dict, ligand_conf_id
        )
        torsion_changes = {
            (dihedral[1], dihedral[2]): float(delta)
            for dihedral, delta in zip(rotatable_dihedrals, torsion_deltas)
        }

    return OptimizationResult(
        mol=optimized_mol,
        score=float(result.fun),
        initial_score=initial_score,
        translation=translation,
        rotation=rotation,
        torsion_changes=torsion_changes,
        converged=result.success,
        n_iterations=result.nit,
    )
