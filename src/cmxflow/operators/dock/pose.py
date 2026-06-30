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
from rdkit.Chem import rdDistGeom, rdMolDescriptors, rdMolTransforms
from scipy.optimize import basinhopping, minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from cmxflow.operators.dock.score import (
    AtomTyping,
    EmpiricalParams,
    IntramolecularPairs,
    NeighborList,
    empirical_score_and_grad_cached,
    empirical_score_and_grad_fast,
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
            differences (False). Analytical is ~20× faster; finite differences
            are retained only as a debug/comparison path.
        constrained_atom_indices: Heavy-atom indices to constrain near their
            initial positions. Resolved from SMARTS in MoleculeDockBlock.
        constraint_weight: Penalty weight in kcal/mol/Å². 0 = disabled.
            Weight 100 confines atoms to ~0.1 Å RMSD from initial positions.
        constraint_tol: Flat-bottom radius in Å. Constrained atoms move freely
            within ``constraint_tol`` of their target, and are penalized only by
            the displacement *beyond* it: ``weight · Σ max(0, ‖d‖ − tol)²``. 0.0
            (default) recovers the pure quadratic tether. A small tol (~0.5 Å)
            with a moderate weight lets a constrained core shift to relieve a
            substituent clash before the restraint resists.
        w_intra: Weight on the intramolecular ligand energy added to the search
            objective (same Vinardo terms/weights as the intermolecular score,
            over 1-4-and-beyond pairs that cross a rotatable bond). Keeps the
            conformer physical during torsion optimization and penalizes
            self-clash. 0 disables (exact pre-Phase-2 behavior). Vina uses 1.0.
            Intramolecular energy affects the search only — it is not included
            in the reported score.
        basin_hops: Number of basin-hopping (iterated local search) steps per
            start. 0 = single L-BFGS-B minimize from the start (pre-Phase-3
            behavior). >0 runs a Metropolis-coupled walk: mutate → minimize →
            accept/reject, keeping the best. This is the global-search engine.
        basin_temperature: Metropolis temperature for accepting uphill local
            minima during basin-hopping (kcal/mol scale).
        step_translation: Std dev (Å) of the per-hop translation perturbation.
        step_rotation: Std dev (rad) of the per-hop rotation-vector perturbation.
        step_torsion: Std dev (deg) of the per-hop torsion perturbation.
        seed: RNG seed for basin-hopping proposals and acceptance.
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
    constraint_tol: float = 0.0
    w_intra: float = 1.0
    basin_hops: int = 0
    basin_temperature: float = 1.0
    step_translation: float = 2.0
    step_rotation: float = 0.5
    step_torsion: float = 60.0
    seed: int = 0


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
    centroid: NDArray,
    rotatable_dihedrals: list[tuple[int, int, int, int]],
    subtrees: list[tuple[int, ...]],
) -> NDArray:
    """Project per-atom gradient onto DOF gradient (T, r, θ).

    Args:
        x: Pose vector [T(3), r(3), θ(n_torsions)].
        pos: Current heavy-atom positions (n_heavy, 3).
        atom_grad: Per-atom gradient from scoring fn (n_heavy, 3).
        centroid: Rotation centre (3,).
        rotatable_dihedrals: List of (i, j, k, l) dihedral tuples.
        subtrees: Precomputed downstream atom indices per torsion.

    Returns:
        DOF gradient of shape (6 + n_torsions,).
    """
    # Translation: sum of atom gradients
    grad_T = np.sum(atom_grad, axis=0)  # (3,)

    # Rotation: exact rotvec gradient via body-frame torque.
    # The global rotation is applied to the *post-torsion* body coordinates:
    #   pos = R @ q + centroid + T,  with q = R^T @ (pos - centroid - T).
    # The body-frame lever arms must therefore be q (current positions mapped
    # back through R and the pivot), NOT the pre-torsion p0_heavy - centroid.
    # The two coincide only for a rigid body; using p0 corrupts the rotation
    # gradient whenever torsions have displaced atoms.
    r = x[3:6]
    R_mat = Rotation.from_rotvec(r).as_matrix()
    d_spatial = pos - centroid - x[:3]  # (n_heavy, 3) lever arms about pivot
    v_body = d_spatial @ R_mat  # = R^T @ d_spatial, post-torsion body arms
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
        Flat-bottom penalty value λ * Σ max(0, ||pos_i - target_i|| - tol)².
        With tol=0 this is the pure quadratic tether λ * Σ ||pos_i - target_i||².
    """
    if not params.constrained_atom_indices or params.constraint_weight == 0.0:
        return 0.0
    idx = list(params.constrained_atom_indices)
    delta = pos[idx] - p0_heavy[idx]
    if params.constraint_tol <= 0.0:
        return params.constraint_weight * float(np.sum(delta**2))
    r = np.linalg.norm(delta, axis=1)
    excess = np.maximum(0.0, r - params.constraint_tol)
    return params.constraint_weight * float(np.sum(excess**2))


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
    if params.constraint_tol <= 0.0:
        # Pure quadratic tether (bit-identical to the pre-flat-bottom behavior).
        score += params.constraint_weight * float(np.sum(delta**2))
        atom_grad[idx] += 2.0 * params.constraint_weight * delta
        return score, atom_grad
    # Flat-bottom: free within tol, quadratic in the excess displacement beyond.
    r = np.linalg.norm(delta, axis=1)
    excess = np.maximum(0.0, r - params.constraint_tol)
    score += params.constraint_weight * float(np.sum(excess**2))
    # grad_i = 2·weight·excess_i·(delta_i / r_i), zero where r_i <= tol.
    active = excess > 0.0
    scale = np.zeros_like(r)
    scale[active] = 2.0 * params.constraint_weight * excess[active] / r[active]
    atom_grad[idx] += scale[:, None] * delta
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
# Restart Screening (rigid + flexible initialization)
# =============================================================================


def optimize_dg_restarts(
    ligand_mol: Chem.Mol,
    protein_coords: np.ndarray,
    protein_typing: AtomTyping,
    params: PoseParams,
    score_params: EmpiricalParams,
    site_center: np.ndarray | None = None,
    ligand_conf_id: int = 0,
    n_extra_confs: int = 0,
    n_center_rotations: int = 128,
    n_translation_samples: int = 128,
    center_fraction: float = 0.5,
    diversity_rmsd: float = 0.0,
    protein_tree: cKDTree | None = None,
) -> list[tuple[float, Chem.Mol]]:
    """Screen starting poses for L-BFGS-B refinement (rigid + flexible paths).

    Builds a candidate grid of ``conformer x rigid-body placement``, scores every
    candidate, and returns a diversity-spaced subset as minimization starts. A
    single method serves both docking modes via ``n_extra_confs``:

    * ``n_extra_confs == 0`` -- **rigid path**. The ensemble is just the input
      conformer, so this is pure rigid-body (translation + rotation) screening.
    * ``n_extra_confs > 0`` -- **flexible path**. The ensemble is the input
      conformer plus ``n_extra_confs`` ETKDGv3 conformers, adding torsion
      diversity on the physical low-energy manifold.

    The placements split into two groups so the search puts a deliberate prior on
    the binding-site center (where confidence is highest and the convergence
    funnel lives):

    * **Center group** (``translation = 0``): the identity placement plus
      ``n_center_rotations`` Sobol rotations, so every conformer is sampled at the
      anchor in many orientations. At the exact center the pocket is full, so all
      orientations clash -- but a near-native orientation clashes *less*, so
      ranking these by score biases toward native-like orientations.
    * **Spread group** (``translation != 0``): ``n_translation_samples`` Sobol
      samples over translation (uniform in ``+/- box/4``) and rotation (rotvec
      uniform in ``+/- pi``), covering nearby placements.

    Algorithm:

    1. **Ensemble.** ``[input_conformer]`` plus ``n_extra_confs`` distance-geometry
       conformers (empty extra set on embed failure). Every member keeps the input
       heavy-atom ordering so refinement / RMSD indices line up.
    2. **Anchor.** ``site_center`` when provided (blind docking onto a known
       pocket), else the input conformer's own centroid (overlay refinement).
    3. **Placements / grid.** The center + spread placements above, taken as a
       Cartesian product with the conformer ensemble. Each conformer is rotated
       about its own centroid and translated so that centroid lands at
       ``anchor + translation`` -- so any center placement puts that conformer's
       centroid exactly on ``anchor`` in the sampled orientation.
    4. **Selection.** Start 0 is always the input conformer at the identity
       placement (the input pose translated to the reference centroid). Then a
       reserved quota of ``round(center_fraction * n_starts)`` seeds is filled from
       the center group by ascending score under the ``diversity_rmsd`` spacing
       gate -- guaranteeing center coverage even though those poses clash. The
       remaining slots up to ``params.n_starts`` are filled from all leftover
       candidates (center + spread) by ascending score, same spacing gate.

    Args:
        ligand_mol: Ligand with 3D coordinates (the input pose / conformer 0).
        protein_coords: Pre-computed protein atom coordinates (n_atoms, 3).
        protein_typing: Pre-computed protein atom typing.
        params: Pose params. ``n_starts`` sets how many starts are returned,
            ``seed`` seeds conformer embedding, ``translation_bounds`` sets the
            sampling box (spread translations use ``box/4``).
        score_params: Scoring function parameters.
        site_center: Optional binding-site centroid. Anchors the whole grid to the
            pocket; ``None`` anchors it to the input conformer's own position.
        ligand_conf_id: Conformer ID of the input pose (becomes ensemble member 0).
        n_extra_confs: ETKDGv3 conformers to embed beyond the input conformer.
            ``0`` is the rigid path (input conformer only, no DG sampling).
        n_center_rotations: Sobol orientations sampled at the anchor (translation
            0), per conformer, in addition to the identity placement.
        n_translation_samples: Sobol (translation + rotation) placements sampled in
            the spread group, per conformer.
        center_fraction: Fraction of ``params.n_starts`` reserved for center-group
            seeds (kept even when they clash). The rest are filled from the whole
            grid by score.
        diversity_rmsd: Minimum heavy-atom RMSD (Angstroms) between selected
            starts. ``0`` (default) disables the spacing gate.
        protein_tree: Optional cached protein KD-tree for sparse scoring.

    Returns:
        List of ``(score, mol)`` of length <= ``params.n_starts``, index 0 being
        the input conformer placed at the anchor.
    """
    ligand_heavy = Chem.RemoveAllHs(ligand_mol)
    ligand_typing = get_atom_typing(ligand_heavy)  # fixed topology; compute once

    def _positions(mol: Chem.Mol) -> NDArray[np.floating]:
        return np.array(mol.GetConformer(0).GetPositions())

    def _score(mol: Chem.Mol) -> float:
        return empirical_score_cached(
            mol,
            protein_coords,
            protein_typing,
            0,
            score_params,
            protein_tree=protein_tree,
            ligand_typing=ligand_typing,
        ).total

    def _single_conformer(mol: Chem.Mol, conf_id: int) -> Chem.Mol:
        """Copy ``mol`` keeping only ``conf_id``, renumbered to id 0."""
        out = Chem.Mol(mol)
        out.RemoveAllConformers()
        conf = Chem.Conformer(mol.GetConformer(conf_id))
        conf.SetId(0)
        out.AddConformer(conf, assignId=False)
        return out

    # --- Step 1: conformer ensemble (input pose first, then DG extras) ---
    ensemble: list[Chem.Mol] = [_single_conformer(ligand_heavy, ligand_conf_id)]
    if n_extra_confs > 0:
        # Embed from ligand_heavy so heavy-atom order matches the input exactly.
        mol_h = Chem.AddHs(ligand_heavy)
        mol_h.RemoveAllConformers()
        dg_params = rdDistGeom.ETKDGv3()
        # ETKDG with randomSeed=0 degenerates to identical conformers, and
        # params.seed defaults to 0 -- offset to a nonzero, deterministic seed so a
        # single EmbedMultipleConfs call yields a diverse ensemble. No pruning: we
        # want exactly the requested conformers.
        dg_params.randomSeed = params.seed + 1
        dg_params.numThreads = 1  # deterministic + safe under block-level parallelism
        rdDistGeom.EmbedMultipleConfs(mol_h, numConfs=n_extra_confs, params=dg_params)
        dg_heavy = Chem.RemoveAllHs(mol_h)
        ensemble.extend(
            _single_conformer(dg_heavy, c.GetId()) for c in dg_heavy.GetConformers()
        )

    centroids = [np.mean(_positions(c), axis=0) for c in ensemble]

    # --- Step 2: anchor (binding-site centroid, or input position if no site) ---
    anchor = site_center if site_center is not None else centroids[0]

    # --- Step 3: placement sets (split translation / rotation sampling) ---
    box_size = max(abs(params.translation_bounds[0]), abs(params.translation_bounds[1]))
    spread_box = box_size / 4.0  # spread translations stay near the pocket
    lo_r, hi_r = [-np.pi] * 3, [np.pi] * 3

    # Center group rotations: identity first, then Sobol orientations at the anchor.
    center_rotvecs = [np.zeros(3)]
    if n_center_rotations > 0:
        rot_sampler = qmc.Sobol(d=3, scramble=True, seed=0)
        center_rotvecs += list(
            qmc.scale(rot_sampler.random(n_center_rotations), lo_r, hi_r)
        )

    # Spread group: paired translation + rotation Sobol draws (separate sequences
    # give better per-axis coverage than a single 6-D draw).
    spread_trans: list[NDArray] = []
    spread_rotvecs: list[NDArray] = []
    if n_translation_samples > 0:
        t_sampler = qmc.Sobol(d=3, scramble=True, seed=1)
        r_sampler = qmc.Sobol(d=3, scramble=True, seed=2)
        spread_trans = list(
            qmc.scale(
                t_sampler.random(n_translation_samples),
                [-spread_box] * 3,
                [spread_box] * 3,
            )
        )
        spread_rotvecs = list(
            qmc.scale(r_sampler.random(n_translation_samples), lo_r, hi_r)
        )

    # --- Step 4: score the conformer x placement grid (scoring only, no min) ---
    # center_grid[0] is conformer 0 at the identity placement = the input pose
    # translated to the anchor, which becomes the unconditional start 0.
    Cand = tuple[float, Chem.Mol, NDArray]
    center_grid: list[Cand] = []
    spread_grid: list[Cand] = []
    for conf, centroid in zip(ensemble, centroids):
        offset = anchor - centroid
        for rv in center_rotvecs:
            cand = apply_rigid_transform(
                conf, offset, Rotation.from_rotvec(rv), 0, centroid
            )
            center_grid.append((_score(cand), cand, _positions(cand)))
        for t, rv in zip(spread_trans, spread_rotvecs):
            cand = apply_rigid_transform(
                conf, t + offset, Rotation.from_rotvec(rv), 0, centroid
            )
            spread_grid.append((_score(cand), cand, _positions(cand)))

    # --- Step 5: select start 0 + reserved center seeds + score-ranked fill ---
    def _diverse(pos: NDArray, kept: list[Cand]) -> bool:
        return all(
            np.sqrt(np.mean(np.sum((pos - k[2]) ** 2, axis=1))) >= diversity_rmsd
            for k in kept
        )

    passed: list[Cand] = [center_grid[0]]
    n_keep = params.n_starts
    if n_keep <= 1:
        return [(s, m) for s, m, _ in passed]

    kept_ids: set[int] = {id(center_grid[0])}

    # Reserve a center-group quota so well-placed (but clashing) center seeds are
    # not outranked by low-clash peripheral poses.
    center_target = max(1, round(center_fraction * n_keep))
    for cand in sorted(center_grid[1:], key=lambda t: t[0]):
        if len(passed) >= center_target or len(passed) >= n_keep:
            break
        if _diverse(cand[2], passed):
            passed.append(cand)
            kept_ids.add(id(cand))

    # Fill the rest from all remaining candidates by ascending score.
    remaining = [c for c in center_grid + spread_grid if id(c) not in kept_ids]
    for cand in sorted(remaining, key=lambda t: t[0]):
        if len(passed) >= n_keep:
            break
        if _diverse(cand[2], passed):
            passed.append(cand)
            kept_ids.add(id(cand))

    logger.info(
        "dg_restarts: %d confs x (1 + %d center rot + %d spread) -> %d starts "
        "(%d center reserved).",
        len(ensemble),
        n_center_rotations,
        n_translation_samples,
        len(passed),
        center_target,
    )
    return [(s, m) for s, m, _ in passed]


# =============================================================================
# Basin-hopping (iterated local search)
# =============================================================================


class _SingleDOFStep:
    """Vina-style single-group proposal for iterated local search.

    Each call perturbs exactly ONE degree-of-freedom group -- translation
    (3 components together), the rotation vector (3 together), or one torsion
    -- chosen uniformly at random, then clips back into the L-BFGS-B bounds.
    Mutating a single group keeps the proposal near the current pose so the
    subsequent local minimization refines that basin rather than restarting,
    which is what lets the Metropolis chain accumulate partial progress across
    hops (the mechanism independent restarts lack).
    """

    def __init__(
        self,
        n_torsions: int,
        lo: NDArray[np.floating],
        hi: NDArray[np.floating],
        t_sigma: float,
        r_sigma: float,
        tor_sigma: float,
        rng: np.random.Generator,
    ) -> None:
        self.n_torsions = n_torsions
        self.lo = lo
        self.hi = hi
        self.t_sigma = t_sigma
        self.r_sigma = r_sigma
        self.tor_sigma = tor_sigma
        self.rng = rng

    def __call__(self, x: NDArray) -> NDArray:
        x = x.copy()
        # Groups: 0 = translation, 1 = rotation, 2.. = individual torsions.
        group = int(self.rng.integers(0, 2 + self.n_torsions))
        if group == 0:
            x[:3] += self.rng.normal(0.0, self.t_sigma, 3)
        elif group == 1:
            x[3:6] += self.rng.normal(0.0, self.r_sigma, 3)
        else:
            x[6 + (group - 2)] += self.rng.normal(0.0, self.tor_sigma)
        return np.clip(x, self.lo, self.hi)


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
    protein_tree: cKDTree | None = None,
) -> OptimizationResult:
    """Optimize ligand pose with pre-computed protein data.

    Performs rigid-body optimization (translation + rotation) and optionally
    flexible optimization (torsion angles) using L-BFGS-B.

    When ``site_center`` is provided, Sobol restart samples (rows 1+) are
    anchored to the binding site rather than the input conformer. Row 0
    always minimizes locally from the input pose. This is the correct mode
    for blind docking from a fresh conformer. Without ``site_center``, all
    restarts are relative to the input conformer, suitable for MCS overlay
    refinement workflows.

    The search objective is the empirical score alone, so the optimization is
    fully analytical. Reporting-only descriptors such as electrostatic
    complementarity are computed by the caller on the returned pose.

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
        protein_tree: Precomputed KDTree for sparse scoring.

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

    # Identify rotatable bonds and subtrees for paired movement
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

    # Search box definition
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

    # Initial score (empirical only)
    initial_score = empirical_score_cached(
        ligand_heavy,
        protein_coords,
        protein_typing,
        ligand_conf_id,
        score_params,
        protein_tree=protein_tree,
        ligand_typing=ligand_typing,
    ).total

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
    use_grad = params.use_analytical_grad

    # Reused Verlet neighbor list for the sparse hot path: built once here and
    # shared across every L-BFGS-B eval and basin hop of this start, so the
    # KD-tree query is paid only on rebuilds (ligand displacement > skin), not
    # per call. Only when a protein_tree is supplied -- the tree-less callers
    # (tests) keep the exact dense per-call path below.
    neighbor_list = (
        NeighborList(
            protein_coords, protein_typing, ligand_typing, protein_tree=protein_tree
        )
        if protein_tree is not None
        else None
    )

    if use_grad:

        def objective(x: NDArray) -> tuple[float, NDArray]:
            transformed = _apply_pose_heavy(x)
            pos = np.array(transformed.GetConformer(ligand_conf_id).GetPositions())
            if neighbor_list is not None:
                score, atom_grad = empirical_score_and_grad_fast(
                    pos, neighbor_list, score_params, torsion_divisor
                )
            else:
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
                x, pos, atom_grad, centroid, rotatable_dihedrals, subtrees
            )
            return score, dof_grad

    else:

        def objective(x: NDArray) -> float:  # type: ignore[misc]
            transformed_heavy = _apply_pose_heavy(x)
            pos = np.array(
                transformed_heavy.GetConformer(ligand_conf_id).GetPositions()
            )
            score = empirical_score_cached(
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
            return score + penalty

    # --- Local minimize from x0=zeros, optionally wrapped in basin-hopping ---
    x0 = np.zeros(6 + n_torsions)
    minimizer_kwargs = dict(
        method="L-BFGS-B",
        jac=use_grad,
        bounds=bounds,
        options={"maxiter": params.max_iterations, "gtol": params.tolerance},
    )
    if params.basin_hops > 0:
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        take_step = _SingleDOFStep(
            n_torsions,
            lo,
            hi,
            params.step_translation,
            params.step_rotation,
            params.step_torsion,
            np.random.default_rng(params.seed),
        )
        best_res = basinhopping(
            objective,
            x0,
            niter=params.basin_hops,
            T=params.basin_temperature,
            minimizer_kwargs=minimizer_kwargs,
            take_step=take_step,
            seed=params.seed,
        )
        converged = bool(best_res.lowest_optimization_result.success)
        n_iter = int(best_res.nit)
    else:
        best_res = minimize(objective, x0, **minimizer_kwargs)
        converged = bool(best_res.success)
        n_iter = int(best_res.nit)

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

    return OptimizationResult(
        mol=optimized_mol,
        score=final_score,
        initial_score=initial_score,
        translation=T,
        rotation=rot,
        torsion_changes=torsion_changes,
        converged=converged,
        n_iterations=n_iter,
        strain=strain,
    )
