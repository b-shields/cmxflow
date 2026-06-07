"""Pose optimization for molecular docking.

This module provides functions for optimizing ligand pose in a protein
binding site using rigid-body transformations and torsion angle optimization.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdMolTransforms
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from cmxflow.operators.dock.score import (
    AtomTyping,
    EmpiricalParams,
    IntramolecularPairs,
    empirical_score_and_grad_cached,
    empirical_score_cached,
    get_atom_typing,
    intramolecular_score_and_grad,
)

logger = logging.getLogger(__name__)

# Type aliases
Coords: TypeAlias = NDArray[np.floating]

# Amide/thioamide N–C bond SMARTS for exclusion from rotatable bonds.
# Amide bonds have ~20 kcal/mol rotational barrier (partial double-bond character)
# and are treated as rigid in docking, consistent with smina/Vina.
_AMIDE_SMARTS: Chem.Mol = Chem.MolFromSmarts("[#7;X3]-[#6;X3]=[O,S]")  # type: ignore[assignment]


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
        n_starts: Number of L-BFGS-B restarts. Start 0 is always the aligned
            pose (x0=zeros). Starts 1..n_starts-1 are Sobol quasi-random
            samples. n_starts=1 preserves single-start behavior. For optimal
            Sobol coverage, prefer values where n_starts-1 is a power of 2
            (i.e. n_starts = 2, 3, 5, 9, 17, 33).
        use_analytical_grad: Use analytical gradients (True) or finite
            differences (False). Analytical is ~20× faster. Automatically
            falls back to finite differences when w_ec > 0 (EC has no
            analytical gradient).
        constrained_atom_indices: Heavy-atom indices to constrain near their
            initial positions. Resolved from SMARTS in MoleculeDockBlock.
        constraint_weight: Penalty weight in kcal/mol/Å². 0 = disabled.
            Weight 100 confines atoms to ~0.1 Å RMSD from initial positions.
        w_intra: Weight on the intramolecular ligand energy added to the search
            objective (same Vinardo terms/weights as the intermolecular score,
            over 1-4-and-beyond pairs that cross a rotatable bond). Keeps the
            conformer physical during torsion optimization and penalizes
            self-clash. 0 disables (exact pre-Phase-2 behavior). Vina uses 1.0.
            Intramolecular energy affects the search only — it is not included
            in the reported score.
    """

    max_iterations: int = 100
    tolerance: float = 1e-5
    translation_bounds: tuple[float, float] = (-10.0, 10.0)
    optimize_torsions: bool = True
    max_torsion_change: float = 180.0
    n_starts: int = 17
    use_analytical_grad: bool = True
    constrained_atom_indices: tuple[int, ...] = field(default_factory=tuple)
    constraint_weight: float = 0.0
    w_intra: float = 1.0


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
        converged: Whether optimization converged (best restart only).
        n_iterations: Number of L-BFGS-B iterations for the best restart.
        ec: Final electrostatic complementarity value (0.0 when w_ec=0).
        strain: Intramolecular energy added vs the input conformer (>=0).
            Reported for diagnostics; not included in ``score``.
    """

    mol: Chem.Mol
    score: float
    initial_score: float
    translation: NDArray[np.floating]
    rotation: Rotation
    torsion_changes: dict[tuple[int, int], float] = field(default_factory=dict)
    converged: bool = False
    n_iterations: int = 0
    ec: float = 0.0
    strain: float = 0.0


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

    centered = coords - center
    rotated = rotation.apply(centered)
    transformed = rotated + center + translation

    for i, pos in enumerate(transformed):
        conf.SetAtomPosition(i, tuple(pos))

    return mol_copy


# =============================================================================
# Torsion Handling
# =============================================================================


def get_rotatable_bonds(mol: Chem.Mol) -> list[tuple[int, int, int, int]]:
    """Get rotatable bond dihedral atom indices, excluding amide bonds.

    Returns 4-tuples of atom indices (i, j, k, l) defining each rotatable
    bond's dihedral angle where j-k is the rotatable bond. Amide/thioamide
    N–C bonds are excluded (high rotational barrier, treated as rigid).

    Args:
        mol: RDKit Mol object (heavy-atom only).

    Returns:
        List of (i, j, k, l) atom index tuples for each rotatable bond.
    """
    rotatable_smarts = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
    if rotatable_smarts is None:
        return []

    matches = mol.GetSubstructMatches(rotatable_smarts)

    # Amide/thioamide bonds: high barrier, treated as rigid
    amide_bonds: set[frozenset[int]] = set()
    if _AMIDE_SMARTS is not None:
        amide_bonds = {
            frozenset([m[0], m[1]]) for m in mol.GetSubstructMatches(_AMIDE_SMARTS)
        }

    dihedrals: list[tuple[int, int, int, int]] = []
    for j, k in matches:
        if frozenset([j, k]) in amide_bonds:
            continue
        j_neighbors = [
            a.GetIdx() for a in mol.GetAtomWithIdx(j).GetNeighbors() if a.GetIdx() != k
        ]
        k_neighbors = [
            a.GetIdx() for a in mol.GetAtomWithIdx(k).GetNeighbors() if a.GetIdx() != j
        ]
        if j_neighbors and k_neighbors:
            dihedrals.append((j_neighbors[0], j, k, k_neighbors[0]))

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
# Subtree Infrastructure
# =============================================================================


def _get_downstream_atoms(mol: Chem.Mol, j: int, k: int) -> tuple[int, ...]:
    """BFS from k blocking j — atoms that move when torsion (j→k) rotates.

    Args:
        mol: RDKit Mol (heavy-atom only).
        j: Pivot atom index (blocked).
        k: Downstream start atom index.

    Returns:
        Tuple of heavy-atom indices in the downstream subtree.
    """
    visited = {j}
    queue: deque[int] = deque([k])
    downstream: list[int] = []
    while queue:
        idx = queue.popleft()
        if idx in visited:
            continue
        visited.add(idx)
        downstream.append(idx)
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            if nb.GetIdx() not in visited:
                queue.append(nb.GetIdx())
    return tuple(downstream)


def _rigid_fragments(
    mol: Chem.Mol, rotatable_dihedrals: list[tuple[int, int, int, int]]
) -> NDArray[np.intp]:
    """Label each atom by its rigid fragment (components after cutting rot bonds).

    Atoms in different fragments can move relative to each other; atoms in the
    same fragment cannot.

    Args:
        mol: Heavy-atom RDKit Mol.
        rotatable_dihedrals: (i, j, k, l) tuples; the rotatable bond is j-k.

    Returns:
        Array of fragment labels, one per atom (union-find roots).
    """
    n = mol.GetNumAtoms()
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    rot_bonds = {frozenset((d[1], d[2])) for d in rotatable_dihedrals}
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if frozenset((a, b)) in rot_bonds:
            continue
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(n)], dtype=np.intp)


def _build_intramolecular_pairs(
    ligand_heavy: Chem.Mol,
    ligand_typing: AtomTyping,
    rotatable_dihedrals: list[tuple[int, int, int, int]],
) -> IntramolecularPairs:
    """Build the conformer-dependent intramolecular nonbonded pair list.

    Includes atom pairs that (1) are separated by >=3 bonds (1-4 and beyond,
    excluding 1-2 and 1-3) and (2) lie in different rigid fragments (so they
    move relative to each other under some torsion). Matches Vina's
    intramolecular pair selection.

    Args:
        ligand_heavy: Heavy-atom RDKit Mol.
        ligand_typing: Ligand atom typing (radii, hydrophobic, donor, acceptor).
        rotatable_dihedrals: (i, j, k, l) tuples; the rotatable bond is j-k.

    Returns:
        IntramolecularPairs for ``intramolecular_score_and_grad``.
    """
    n = ligand_heavy.GetNumAtoms()
    frag = _rigid_fragments(ligand_heavy, rotatable_dihedrals)
    topo = Chem.GetDistanceMatrix(ligand_heavy)  # bond counts (n, n)

    iu, ju = np.triu_indices(n, k=1)
    mask = (frag[iu] != frag[ju]) & (topo[iu, ju] >= 3)
    i_idx = iu[mask].astype(np.intp)
    j_idx = ju[mask].astype(np.intp)

    radii = ligand_typing.radii
    hydro = ligand_typing.is_hydrophobic
    donor = ligand_typing.is_hbond_donor
    acceptor = ligand_typing.is_hbond_acceptor
    return IntramolecularPairs(
        i_idx=i_idx,
        j_idx=j_idx,
        radii_sum=radii[i_idx] + radii[j_idx],
        hydro_pair=hydro[i_idx] & hydro[j_idx],
        hbond_pair=(donor[i_idx] & acceptor[j_idx]) | (acceptor[i_idx] & donor[j_idx]),
    )


# =============================================================================
# Gradient Projection
# =============================================================================


def _right_jac_T(r: NDArray) -> NDArray:
    """Transpose of right Jacobian of SO(3).

    Maps body-frame torque to rotvec gradient: grad_r = J_r^T @ tau_body.
    Reduces to identity at r=0.

    J_r(r) = (sinθ/θ)I + (1 − sinθ/θ)nnᵀ + ((cosθ−1)/θ)[n]×
    J_r^T  = (sinθ/θ)I + (1 − sinθ/θ)nnᵀ + ((1−cosθ)/θ)[n]×

    Args:
        r: Rotation vector (3,).

    Returns:
        3×3 matrix J_r^T.
    """
    theta = np.linalg.norm(r)
    if theta < 1e-8:
        return np.eye(3)
    n = r / theta
    n_skew = np.array([[0.0, -n[2], n[1]], [n[2], 0.0, -n[0]], [-n[1], n[0], 0.0]])
    s = np.sin(theta) / theta
    c = (1.0 - np.cos(theta)) / theta
    return s * np.eye(3) + (1.0 - s) * np.outer(n, n) + c * n_skew


def _compute_dof_gradient(
    x: NDArray,
    pos: NDArray,
    atom_grad: NDArray,
    p0_heavy: NDArray,
    centroid: NDArray,
    rotatable_dihedrals: list[tuple[int, int, int, int]],
    subtrees: list[tuple[int, ...]],
) -> NDArray:
    """Project per-atom gradient onto DOF gradient (T, r, θ).

    Args:
        x: Pose vector [T(3), r(3), θ(n_torsions)].
        pos: Current heavy-atom positions (n_heavy, 3).
        atom_grad: Per-atom gradient from scoring fn (n_heavy, 3).
        p0_heavy: Original heavy-atom positions before any transform (n_heavy, 3).
        centroid: Rotation centre (3,).
        rotatable_dihedrals: List of (i, j, k, l) dihedral tuples.
        subtrees: Precomputed downstream atom indices per torsion.

    Returns:
        DOF gradient of shape (6 + n_torsions,).
    """
    # Translation: sum of atom gradients
    grad_T = np.sum(atom_grad, axis=0)  # (3,)

    # Rotation: exact rotvec gradient via body-frame torque
    r = x[3:6]
    R_mat = Rotation.from_rotvec(r).as_matrix()
    v_body = p0_heavy - centroid  # (n_heavy, 3) pre-rotation lever arms
    g_body = atom_grad @ R_mat  # (n_heavy, 3) gradients in body frame
    tau_body = np.sum(np.cross(v_body, g_body), axis=0)  # (3,)
    grad_r = _right_jac_T(r) @ tau_body  # (3,)

    # Torsions: lever-arm projection for each bond
    n_torsions = len(rotatable_dihedrals)
    grad_theta = np.zeros(n_torsions)
    for k_tor, (_, j_atom, k_atom, _) in enumerate(rotatable_dihedrals):
        axis = pos[k_atom] - pos[j_atom]
        axis = axis / np.linalg.norm(axis)
        idx = list(subtrees[k_tor])
        r_i = pos[idx] - pos[k_atom]  # (n_downstream, 3)
        lever = np.cross(axis[None, :], r_i)  # (n_downstream, 3)
        # π/180 converts per-radian gradient to per-degree (torsions in degrees)
        grad_theta[k_tor] = np.einsum("ij,ij->", lever, atom_grad[idx]) * (
            np.pi / 180.0
        )

    return np.concatenate([grad_T, grad_r, grad_theta])


# =============================================================================
# Constraint Penalties
# =============================================================================


def _constraint_penalty_value(
    pos: NDArray,
    p0_heavy: NDArray,
    params: PoseParams,
) -> float:
    """Compute atom constraint penalty (no gradient).

    Args:
        pos: Current heavy-atom positions (n_heavy, 3).
        p0_heavy: Target positions (n_heavy, 3).
        params: Pose params with constrained_atom_indices and constraint_weight.

    Returns:
        Penalty value λ * Σ ||pos_i - target_i||².
    """
    if not params.constrained_atom_indices or params.constraint_weight == 0.0:
        return 0.0
    idx = list(params.constrained_atom_indices)
    delta = pos[idx] - p0_heavy[idx]
    return params.constraint_weight * float(np.sum(delta**2))


def _apply_constraint_penalty(
    score: float,
    atom_grad: NDArray,
    pos: NDArray,
    p0_heavy: NDArray,
    params: PoseParams,
) -> tuple[float, NDArray]:
    """Add atom constraint penalty to score and gradient.

    Penalty gradient is added to atom_grad before DOF projection so the
    kinematic chain rule automatically propagates it to all DOF types.

    Args:
        score: Current score.
        atom_grad: Per-atom gradient (n_heavy, 3) — not mutated in place.
        pos: Current heavy-atom positions (n_heavy, 3).
        p0_heavy: Target positions (n_heavy, 3).
        params: Pose params.

    Returns:
        Tuple of (score + penalty, atom_grad + penalty_grad).
    """
    if not params.constrained_atom_indices or params.constraint_weight == 0.0:
        return score, atom_grad
    atom_grad = atom_grad.copy()
    idx = list(params.constrained_atom_indices)
    delta = pos[idx] - p0_heavy[idx]
    score += params.constraint_weight * float(np.sum(delta**2))
    atom_grad[idx] += 2.0 * params.constraint_weight * delta
    return score, atom_grad


def constraint_weight_for_rmsd(target_rmsd_angstrom: float) -> float:
    """Approximate constraint weight to achieve a given RMSD tolerance.

    Args:
        target_rmsd_angstrom: Desired RMSD tolerance in Angstroms.

    Returns:
        Weight in kcal/mol/Å².
    """
    return 1.0 / (target_rmsd_angstrom**2)


# =============================================================================
# Sobol Restart Sampling
# =============================================================================


def optimize_sobol_restarts(
    ligand_mol: Chem.Mol,
    protein_coords: np.ndarray,
    protein_typing: AtomTyping,
    params: PoseParams,
    score_params: EmpiricalParams,
    site_center: np.ndarray | None = None,
    rigid: bool = False,
    ligand_conf_id: int = 0,
    max_tries: int = 1024,
    max_score_per_heavy_atom: float = 3.0,
    diversity_rmsd: float = 0.0,
    protein_tree: cKDTree | None = None,
) -> list[tuple[float, Chem.Mol]]:
    """Multi-start sobol pre-screening, returning starting poses for minimization.

    Row 0 is always the input aligned pose (unconditional). This wastes one slot
    when the aligned pose is clashing, but preserves MCS/overlay behaviour where
    the aligned start is intentionally the binding hypothesis.

    Candidates are sampled within ``box_size / 2`` so they land near the pocket;
    L-BFGS-B bounds remain at the full ``box_size``.

    Args:
        ligand_mol: Ligand with 3D coordinates.
        protein_coords: Pre-computed protein atom coordinates (n_atoms, 3).
        protein_typing: Pre-computed protein atom typing.
        params: Pose params. ``optimize_torsions`` is ignored; always rigid.
        score_params: Scoring function parameters.
        site_center: Optional binding site centroid — anchors Sobol restarts
            to the pocket (rows 1+). Row 0 always starts from the input pose.
        rigid: If True only optimize over translation and rotation.
        ligand_conf_id: Ligand conformer ID.
        max_tries: Maximum number of Sobol candidates to evaluate when
            searching for ``n_starts - 1`` acceptable starts.
        max_score_per_heavy_atom: Score threshold for accepting a Sobol candidate
            where ``max_score`` is computed by multiplying by ``n_heavy``.
        diversity_rmsd: Minimum heavy-atom RMSD (Å) between selected starting
            poses. Prevents L-BFGS-B restarts from redundantly exploring the
            same basin.

    Returns:
        List of (score, mol) tuples of max length ``n_starts`` (params).
    """
    # Get heavy atom centroid
    ligand_heavy = Chem.RemoveAllHs(ligand_mol)
    p0_heavy = np.array(ligand_heavy.GetConformer(ligand_conf_id).GetPositions())
    centroid = np.mean(p0_heavy, axis=0)
    ligand_typing = get_atom_typing(ligand_heavy)  # fixed topology; compute once

    # Set max score
    max_score = ligand_heavy.GetNumHeavyAtoms() * max_score_per_heavy_atom

    # Set translational box size
    box_size = max(abs(params.translation_bounds[0]), abs(params.translation_bounds[1]))
    site_offset: NDArray | None = None
    if site_center is not None:
        # Offset from center if specified
        site_offset = site_center - centroid

    # Set torsional bounds
    n_tor = 0
    if not rigid:
        rotatable_dihedrals = get_rotatable_bonds(ligand_heavy)
        initial_torsions = np.array(
            [
                get_dihedral_angle(ligand_heavy, *d, ligand_conf_id)
                for d in rotatable_dihedrals
            ]
        )
        n_tor = len(rotatable_dihedrals)

    # Sobol sampling
    n_random = params.n_starts - 1  # row 0 reserved for aligned pose
    sample_box = box_size / 2.0
    n_total = 6 + n_tor
    lo = np.array(
        [-sample_box] * 3 + [-np.pi] * 3 + [-params.max_torsion_change] * n_tor
    )
    hi = np.array([sample_box] * 3 + [np.pi] * 3 + [params.max_torsion_change] * n_tor)
    sampler = qmc.Sobol(d=n_total, scramble=True, seed=0)

    # Generate all max_tries points upfront so the Sobol sequence is contiguous.
    unit_all = sampler.random(max_tries)
    xs_sobol = qmc.scale(unit_all, lo, hi)
    if site_offset is not None:
        xs_sobol[:, :3] += site_offset

    # Merge with initial point (always score starting pose)
    x0 = np.zeros(shape=(1, n_total), dtype=np.float64)
    xs_all = np.vstack([x0, xs_sobol])

    # Rejection sampling loop: max_score and RMSD constraint
    passed: list[tuple[float, Chem.Mol, NDArray]] = []  # (score, x, positions)
    n_tried = 0
    for x in xs_all:
        n_tried += 1
        # Rigid translation + rotation
        T = x[:3]
        rot = Rotation.from_rotvec(x[3:6])
        mol_c = apply_rigid_transform(
            ligand_heavy, T, rot, ligand_conf_id, center=centroid
        )

        # Torsions
        if not rigid and n_tor > 0:
            new_torsions = initial_torsions + x[6:]
            mol_c = apply_torsion_changes(
                mol_c, dict(zip(rotatable_dihedrals, new_torsions)), ligand_conf_id
            )

        # New positions
        pos = np.array(mol_c.GetConformer(ligand_conf_id).GetPositions())

        # Check RMSD diversity contratin
        if len(passed) > 1 and not all(
            np.sqrt(np.mean(np.sum((pos - p[2]) ** 2, axis=1))) >= diversity_rmsd
            for p in passed
        ):
            continue

        # Check total empirical score constraint
        sc = empirical_score_cached(
            mol_c,
            protein_coords,
            protein_typing,
            ligand_conf_id,
            score_params,
            protein_tree=protein_tree,
            ligand_typing=ligand_typing,
        ).total

        if len(passed) == 0 or sc < max_score:
            passed.append((sc, mol_c, pos))

        # Early stopping
        if len(passed) >= n_random:
            break

    logger.info(
        "sobol_restarts: %d/%d candidates passed score < %.1f after %d tries.",
        len(passed),
        n_random,
        max_score,
        n_tried,
    )
    if len(passed) < n_random:
        logger.warning(
            "sobol_restarts: only %d of %d non-clashing starts found.",
            len(passed),
            n_random,
        )

    return [(p[0], p[1]) for p in passed]


# =============================================================================
# Main Optimization Functions (cached path)
# =============================================================================


def optimize_pose_cached(
    ligand_mol: Chem.Mol,
    protein_coords: np.ndarray,
    protein_typing: AtomTyping,
    params: PoseParams | None = None,
    score_params: EmpiricalParams | None = None,
    ligand_conf_id: int = 0,
    site_center: np.ndarray | None = None,
    protein_ec_coords: np.ndarray | None = None,
    protein_ec_charges: np.ndarray | None = None,
    w_ec: float = 0.0,
    protein_tree: cKDTree | None = None,
) -> OptimizationResult:
    """Optimize ligand pose with pre-computed protein data.

    Performs rigid-body optimization (translation + rotation) and optionally
    flexible optimization (torsion angles) using L-BFGS-B. Supports
    multi-start optimization via Sobol quasi-random sampling, analytical
    gradients, and atom-position constraints.

    When ``site_center`` is provided, Sobol restart samples (rows 1+) are
    anchored to the binding site rather than the input conformer. Row 0
    always minimizes locally from the input pose. This is the correct mode
    for blind docking from a fresh conformer. Without ``site_center``, all
    restarts are relative to the input conformer, suitable for MCS overlay
    refinement workflows.

    When ``w_ec > 0`` and EC data is provided, the optimizer jointly minimizes
    ``empirical_score - w_ec * ec_score`` using finite differences (EC lacks
    analytical gradients).

    Args:
        ligand_mol: Ligand RDKit Mol with 3D coordinates.
        protein_coords: Pre-computed protein atom 3D coordinates (n_atoms, 3).
        protein_typing: Pre-computed protein atom typing.
        params: Pose optimization parameters. If None, uses defaults.
        score_params: Scoring function parameters. If None, uses defaults.
        ligand_conf_id: Ligand conformer ID to optimize.
        site_center: Optional (3,) binding site centroid in Angstroms. When
            provided, Sobol restarts are distributed within ``±box_size`` of
            the site center. Typically the centroid of a co-crystallized
            ligand or a known active pose.
        protein_ec_coords: Pre-computed protein coordinates (with H) for EC.
        protein_ec_charges: Pre-computed protein Gasteiger charges for EC.
        w_ec: Weight for EC term. When 0, EC is not computed during optimization.

    Returns:
        OptimizationResult with optimized molecule and metadata.
    """
    if params is None:
        params = PoseParams()
    if score_params is None:
        score_params = EmpiricalParams()

    # --- Setup (once per call) ---
    ligand_heavy = Chem.RemoveAllHs(ligand_mol)
    p0_heavy = np.array(ligand_heavy.GetConformer(ligand_conf_id).GetPositions())
    centroid = np.mean(p0_heavy, axis=0)
    ligand_typing = get_atom_typing(ligand_heavy)  # fixed topology; compute once

    rotatable_dihedrals: list[tuple[int, int, int, int]] = []
    subtrees: list[tuple[int, ...]] = []
    initial_torsions: NDArray[np.floating] = np.array([])

    if params.optimize_torsions:
        rotatable_dihedrals = get_rotatable_bonds(ligand_heavy)
        subtrees = [
            _get_downstream_atoms(ligand_heavy, j, k)
            for (_, j, k, _) in rotatable_dihedrals
        ]
        if rotatable_dihedrals:
            initial_torsions = np.array(
                [
                    get_dihedral_angle(ligand_heavy, *d, ligand_conf_id)
                    for d in rotatable_dihedrals
                ]
            )

    n_torsions = len(rotatable_dihedrals)

    # Intramolecular setup: conf-dependent pair list + matching torsion divisor.
    # Built whenever torsions move (needed for both the search term and strain
    # reporting); only non-trivial when rotatable bonds exist.
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(ligand_heavy, strict=False)
    torsion_divisor = 1.0 + score_params.w_rot * n_rot
    intra_pairs: IntramolecularPairs | None = None
    initial_intra = 0.0
    if rotatable_dihedrals:
        intra_pairs = _build_intramolecular_pairs(
            ligand_heavy, ligand_typing, rotatable_dihedrals
        )
        initial_intra = intramolecular_score_and_grad(
            p0_heavy, intra_pairs, score_params, torsion_divisor
        )[0]

    # Index map: heavy_to_full[heavy_idx] = full_mol_idx (for output mol)
    heavy_to_full = [
        i
        for i in range(ligand_mol.GetNumAtoms())
        if ligand_mol.GetAtomWithIdx(i).GetAtomicNum() > 1
    ]
    rotatable_dihedrals_full: list[tuple[int, int, int, int]] = [
        (
            heavy_to_full[d[0]],
            heavy_to_full[d[1]],
            heavy_to_full[d[2]],
            heavy_to_full[d[3]],
        )
        for d in rotatable_dihedrals
    ]

    box_size = max(abs(params.translation_bounds[0]), abs(params.translation_bounds[1]))
    site_offset: NDArray | None = None
    if site_center is not None:
        site_offset = site_center - centroid
        # Bounds cover both x=0 (aligned pose) and the full site-centred box.
        trans_bounds = [
            (min(0.0, site_offset[k] - box_size), max(0.0, site_offset[k] + box_size))
            for k in range(3)
        ]
    else:
        trans_bounds = [(-box_size, box_size)] * 3

    bounds = (
        trans_bounds
        + [(-np.pi, np.pi)] * 3
        + [(-params.max_torsion_change, params.max_torsion_change)] * n_torsions
    )

    # Initial score (combined empirical + EC)
    initial_score = empirical_score_cached(
        ligand_heavy,
        protein_coords,
        protein_typing,
        ligand_conf_id,
        score_params,
        protein_tree=protein_tree,
        ligand_typing=ligand_typing,
    ).total
    if w_ec > 0 and protein_ec_coords is not None and protein_ec_charges is not None:
        from cmxflow.operators.dock.score import ec_score_cached

        initial_ec = ec_score_cached(
            ligand_mol, protein_ec_coords, protein_ec_charges, ligand_conf_id
        )
        initial_score -= w_ec * initial_ec

    # --- Helper: apply pose vector to ligand_heavy ---
    def _apply_pose_heavy(x: NDArray) -> Chem.Mol:
        T = x[:3]
        rot = Rotation.from_rotvec(x[3:6])
        mol = apply_rigid_transform(
            ligand_heavy, T, rot, ligand_conf_id, center=centroid
        )
        if rotatable_dihedrals and n_torsions > 0:
            new_torsions = initial_torsions + x[6:]
            mol = apply_torsion_changes(
                mol, dict(zip(rotatable_dihedrals, new_torsions)), ligand_conf_id
            )
        return mol

    # --- Select objective (analytical grad or finite differences) ---
    use_grad = params.use_analytical_grad and w_ec == 0.0

    if use_grad:

        def objective(x: NDArray) -> tuple[float, NDArray]:
            transformed = _apply_pose_heavy(x)
            pos = np.array(transformed.GetConformer(ligand_conf_id).GetPositions())
            score, atom_grad = empirical_score_and_grad_cached(
                transformed,
                protein_coords,
                protein_typing,
                ligand_conf_id,
                score_params,
                protein_tree=protein_tree,
                ligand_typing=ligand_typing,
            )
            # Intramolecular ligand energy (search objective only). Both atoms of
            # each pair move, so its gradient adds to atom_grad and propagates
            # through the DOF chain rule below.
            if intra_pairs is not None and params.w_intra > 0.0:
                intra_s, intra_g = intramolecular_score_and_grad(
                    pos, intra_pairs, score_params, torsion_divisor
                )
                score += params.w_intra * intra_s
                atom_grad = atom_grad + params.w_intra * intra_g
            score, atom_grad = _apply_constraint_penalty(
                score, atom_grad, pos, p0_heavy, params
            )
            dof_grad = _compute_dof_gradient(
                x, pos, atom_grad, p0_heavy, centroid, rotatable_dihedrals, subtrees
            )
            return score, dof_grad

    else:

        def objective(x: NDArray) -> float:  # type: ignore[misc]
            transformed_heavy = _apply_pose_heavy(x)
            pos = np.array(
                transformed_heavy.GetConformer(ligand_conf_id).GetPositions()
            )
            vinardo = empirical_score_cached(
                transformed_heavy,
                protein_coords,
                protein_typing,
                ligand_conf_id,
                score_params,
                protein_tree=protein_tree,
                ligand_typing=ligand_typing,
            ).total
            penalty = _constraint_penalty_value(pos, p0_heavy, params)
            if intra_pairs is not None and params.w_intra > 0.0:
                penalty += (
                    params.w_intra
                    * intramolecular_score_and_grad(
                        pos, intra_pairs, score_params, torsion_divisor
                    )[0]
                )
            if (
                w_ec > 0
                and protein_ec_coords is not None
                and protein_ec_charges is not None
            ):
                from cmxflow.operators.dock.score import ec_score_cached

                T = x[:3]
                rot = Rotation.from_rotvec(x[3:6])
                transformed_full = apply_rigid_transform(
                    ligand_mol, T, rot, ligand_conf_id, center=centroid
                )
                if rotatable_dihedrals_full and n_torsions > 0:
                    new_torsions = initial_torsions + x[6:]
                    transformed_full = apply_torsion_changes(
                        transformed_full,
                        dict(zip(rotatable_dihedrals_full, new_torsions)),
                        ligand_conf_id,
                    )
                ec = ec_score_cached(
                    transformed_full,
                    protein_ec_coords,
                    protein_ec_charges,
                    ligand_conf_id,
                )
                return vinardo - w_ec * ec + penalty
            return vinardo + penalty

    # --- Single L-BFGS-B minimize from x0=zeros (input pose) ---
    x0 = np.zeros(6 + n_torsions)
    best_res = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        jac=use_grad,
        bounds=bounds,
        options={"maxiter": params.max_iterations, "gtol": params.tolerance},
    )

    # --- Build output mol (full, with H) ---
    T = best_res.x[:3]
    rot = Rotation.from_rotvec(best_res.x[3:6])
    optimized_mol = apply_rigid_transform(
        ligand_mol, T, rot, ligand_conf_id, center=centroid
    )

    torsion_changes: dict[tuple[int, int], float] = {}
    if rotatable_dihedrals_full and n_torsions > 0:
        torsion_deltas = best_res.x[6:]
        new_torsions = initial_torsions + torsion_deltas
        optimized_mol = apply_torsion_changes(
            optimized_mol,
            dict(zip(rotatable_dihedrals_full, new_torsions)),
            ligand_conf_id,
        )
        torsion_changes = {
            (d[1], d[2]): float(delta)
            for d, delta in zip(rotatable_dihedrals, torsion_deltas)
        }

    # Reported score is intermolecular-only (+ EC), excluding the intramolecular
    # search term and constraint penalty — keeps docking_score comparable to
    # smina. Re-evaluate on the optimized pose.
    optimized_heavy = Chem.RemoveAllHs(optimized_mol)
    final_pos = np.array(optimized_heavy.GetConformer(ligand_conf_id).GetPositions())
    final_score = empirical_score_cached(
        optimized_heavy,
        protein_coords,
        protein_typing,
        ligand_conf_id,
        score_params,
        protein_tree=protein_tree,
        ligand_typing=ligand_typing,
    ).total

    # Strain: intramolecular energy added vs the input conformer (>=0; a penalty
    # for docking-induced distortion, reported but not added to score here).
    strain = 0.0
    if intra_pairs is not None:
        final_intra = intramolecular_score_and_grad(
            final_pos, intra_pairs, score_params, torsion_divisor
        )[0]
        strain = max(0.0, final_intra - initial_intra)

    # Final EC on optimized pose
    final_ec = 0.0
    if w_ec > 0 and protein_ec_coords is not None and protein_ec_charges is not None:
        from cmxflow.operators.dock.score import ec_score_cached

        final_ec = ec_score_cached(
            optimized_mol, protein_ec_coords, protein_ec_charges, ligand_conf_id
        )
        final_score -= w_ec * final_ec

    return OptimizationResult(
        mol=optimized_mol,
        score=final_score,
        initial_score=initial_score,
        translation=T,
        rotation=rot,
        torsion_changes=torsion_changes,
        converged=best_res.success,
        n_iterations=best_res.nit,
        ec=final_ec,
        strain=strain,
    )


# =============================================================================
# Convenience Wrappers (non-cached, use empirical_score directly)
# =============================================================================


def optimize_pose(
    ligand_mol: Chem.Mol,
    protein_mol: Chem.Mol,
    params: PoseParams | None = None,
    ligand_conf_id: int = 0,
    protein_conf_id: int = 0,
) -> OptimizationResult:
    """Optimize ligand pose in protein binding site.

    Convenience wrapper using full RDKit mol objects. For performance,
    prefer ``optimize_pose_cached`` when scoring multiple ligands against
    the same protein.

    Args:
        ligand_mol: Ligand RDKit Mol with 3D coordinates.
        protein_mol: Protein RDKit Mol with 3D coordinates.
        params: Optimization parameters. If None, uses defaults.
        ligand_conf_id: Ligand conformer ID to optimize.
        protein_conf_id: Protein conformer ID (fixed).

    Returns:
        OptimizationResult with optimized molecule and metadata.
    """
    if params is None:
        params = PoseParams()

    protein_heavy = Chem.RemoveAllHs(protein_mol)

    protein_conf = protein_heavy.GetConformer(protein_conf_id)
    protein_coords = np.array(protein_conf.GetPositions())
    from cmxflow.operators.dock.score import get_atom_typing

    protein_typing = get_atom_typing(protein_heavy)

    return optimize_pose_cached(
        ligand_mol,
        protein_coords,
        protein_typing,
        params=params,
        ligand_conf_id=ligand_conf_id,
    )


def optimize_pose_rigid(
    ligand_mol: Chem.Mol,
    protein_mol: Chem.Mol,
    max_iterations: int = 100,
    ligand_conf_id: int = 0,
    protein_conf_id: int = 0,
) -> OptimizationResult:
    """Optimize ligand pose with rigid-body transformation only.

    Args:
        ligand_mol: Ligand molecule.
        protein_mol: Protein molecule.
        max_iterations: Maximum iterations.
        ligand_conf_id: Ligand conformer ID.
        protein_conf_id: Protein conformer ID.

    Returns:
        OptimizationResult with optimized rigid pose.
    """
    params = PoseParams(max_iterations=max_iterations, optimize_torsions=False)
    return optimize_pose(
        ligand_mol, protein_mol, params, ligand_conf_id, protein_conf_id
    )


def optimize_pose_flexible(
    ligand_mol: Chem.Mol,
    protein_mol: Chem.Mol,
    max_iterations: int = 200,
    max_torsion_change: float = 30.0,
    ligand_conf_id: int = 0,
    protein_conf_id: int = 0,
) -> OptimizationResult:
    """Optimize ligand pose with rigid-body and torsion flexibility.

    Args:
        ligand_mol: Ligand molecule.
        protein_mol: Protein molecule.
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
        ligand_mol, protein_mol, params, ligand_conf_id, protein_conf_id
    )
