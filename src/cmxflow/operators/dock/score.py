"""Empirical scoring function for molecular docking.

Implements the Vinardo empirical scoring function for protein-ligand binding
affinity estimation as described in Quiroga & Villarreal (2016), as implemented
in smina.

Reference:
    Quiroga & Villarreal (2016). Vinardo: A Scoring Function Based on
    Autodock Vina Improves Scoring, Docking, and Virtual Screening.
    PLOS ONE 11(5): e0155183. https://doi.org/10.1371/journal.pone.0155183
"""

import logging
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

# Type aliases
Coords: TypeAlias = NDArray[np.floating]
DistanceMatrix: TypeAlias = NDArray[np.floating]


# =============================================================================
# Empirical Parameters
# =============================================================================


@dataclass(frozen=True)
class EmpiricalParams:
    """Vinardo/empirical scoring function parameters.

    All default values are from Quiroga & Villarreal (2016) as implemented
    in smina.

    Attributes:
        w_gauss1: Weight for Gaussian attractive term.
        w_repulsion: Weight for repulsion term.
        w_hydrophobic: Weight for hydrophobic interactions.
        w_hbond: Weight for hydrogen bonding.
        w_rot: Torsional entropy divisor weight. The final score is divided by
            ``(1 + w_rot * N_rot)`` where N_rot is the number of rotatable bonds.
            Default 0.02 matches smina's Vinardo implementation; the original
            Vinardo paper uses 0.0 (no correction).
        gauss1_offset: Gaussian center offset (o1 in paper).
        gauss1_width: Gaussian width (s1 in paper).
        hydro_good: Inner cutoff for hydrophobic (p1 in paper).
        hydro_bad: Outer cutoff for hydrophobic (p2 in paper).
        hbond_good: Inner cutoff for H-bond (h1 in paper).
    """

    w_gauss1: float = -0.045
    w_repulsion: float = 0.800
    w_hydrophobic: float = -0.035
    w_hbond: float = -0.600
    gauss1_offset: float = 0.0
    gauss1_width: float = 0.8
    hydro_good: float = 0.0
    hydro_bad: float = 2.5
    hbond_good: float = -0.6
    w_rot: float = 0.02


@dataclass
class ScoreComponents:
    """Per-term raw sums and weights from a scoring evaluation.

    Raw values are the unweighted sums over all atom pairs. The per-term weighted
    properties (``gauss1``, ``repulsion``, ``hydrophobic``, ``hbond``) are
    ``weight * raw`` with NO torsion divisor applied -- they match smina's
    pre-weighting term log directly. The torsion divisor (smina's
    ``num_tors_div`` / ``conf_independent`` correction) is applied once to their
    sum in ``total``, which is the score returned by the scoring function. This
    is identical to dividing each term (``sum(w*raw)/d == sum(w*raw/d)``) but
    cheaper and clearer.

    Attributes:
        gauss1_raw: Unweighted sum of Gaussian attractive term.
        repulsion_raw: Unweighted sum of repulsion term.
        hydrophobic_raw: Unweighted sum of hydrophobic term.
        hbond_raw: Unweighted sum of H-bond term.
        w_gauss1: Weight applied to gauss1 (from scoring params).
        w_repulsion: Weight applied to repulsion.
        w_hydrophobic: Weight applied to hydrophobic.
        w_hbond: Weight applied to hbond.
        n_rot: Number of rotatable bonds in the ligand.
        w_rot: Torsional entropy divisor weight.
    """

    gauss1_raw: float
    repulsion_raw: float
    hydrophobic_raw: float
    hbond_raw: float
    w_gauss1: float
    w_repulsion: float
    w_hydrophobic: float
    w_hbond: float
    n_rot: int = 0
    w_rot: float = 0.0

    @property
    def _torsion_divisor(self) -> float:
        return 1.0 + self.w_rot * self.n_rot

    @property
    def gauss1(self) -> float:
        return self.w_gauss1 * self.gauss1_raw

    @property
    def repulsion(self) -> float:
        return self.w_repulsion * self.repulsion_raw

    @property
    def hydrophobic(self) -> float:
        return self.w_hydrophobic * self.hydrophobic_raw

    @property
    def hbond(self) -> float:
        return self.w_hbond * self.hbond_raw

    @property
    def total(self) -> float:
        return (
            self.gauss1 + self.repulsion + self.hydrophobic + self.hbond
        ) / self._torsion_divisor


# =============================================================================
# Atom Typing
# =============================================================================

# Vinardo atomic radii (Angstroms)
VINARDO_RADII: dict[int, float] = {
    6: 2.0,  # Carbon (aliphatic)
    7: 1.7,  # Nitrogen
    8: 1.6,  # Oxygen
    9: 1.5,  # Fluorine
    15: 2.1,  # Phosphorus
    16: 2.0,  # Sulfur
    17: 1.8,  # Chlorine
    35: 2.0,  # Bromine
    53: 2.2,  # Iodine
}
AROMATIC_CARBON_RADIUS = 1.9
DEFAULT_RADIUS = 1.7

# SMARTS patterns for atom classification
# Vinardo hydrophobic atom definition:
#   - Aromatic C: always hydrophobic (aromatic ring carbons)
#   - Aliphatic C: hydrophobic only when not adjacent to polar atoms (N, O, P, S)
#   - Halogens (F, Cl, Br, I): always hydrophobic
# This matches smina/AutoDock Vina XS atom typing (C_H = hydrophobic carbon).
HYDROPHOBIC_SMARTS = (
    "[$([#6;a]),$([#6;A;!$([#6]~[#7,#8,#15,#16])]),$([#9,#17,#35,#53])]"
)
HBOND_DONOR_SMARTS = (
    "[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O;!H0;+0]),$([n;H1;+0]),$([n;!H0;+1])"
    ",Li+1,Na+1,K+1,Cs+1,Mg+2,Ca+2,Mn+2,Zn+2]"
)
HBOND_ACCEPTOR_SMARTS = (
    "[$([O;H1;v2]-[!$(*=[O,N,P,S])]),$([O;H0;v2]),$([O;H2;v2]),$([O;-]),"
    "$([N;v3;!$(N-*=!@[O,N,P,S]);!$(N-c)]),$([nH0,o;+0])]"
)


@dataclass
class AtomTyping:
    """Atom classification for empirical scoring.

    Attributes:
        radii: Van der Waals radii for each atom.
        is_hydrophobic: Boolean mask for hydrophobic atoms.
        is_hbond_donor: Boolean mask for H-bond donors.
        is_hbond_acceptor: Boolean mask for H-bond acceptors.
        weights: Per-atom occupancy weight applied to every pairwise term this
            atom contributes (1.0 for ordinary atoms). Used to occupancy-weight
            crystallographic alternate-location (altLoc) conformers so a residue
            sampling two states contributes the ensemble-averaged interaction.
            Defaults to all-ones when not provided.
    """

    radii: NDArray[np.floating]
    is_hydrophobic: NDArray[np.bool_]
    is_hbond_donor: NDArray[np.bool_]
    is_hbond_acceptor: NDArray[np.bool_]
    # Defaults to all-ones in __post_init__; type-ignore the None sentinel so
    # downstream use sites see a plain (non-Optional) array.
    weights: NDArray[np.floating] = field(default=None)  # type: ignore[arg-type]

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = np.ones(len(self.radii), dtype=np.float64)


def get_atom_radii(mol: Chem.Mol) -> NDArray[np.floating]:
    """Get Vinardo atomic radii for all atoms in a molecule.

    Args:
        mol: RDKit Mol object.

    Returns:
        Array of radii (n_atoms,) in Angstroms.
    """
    n_atoms = mol.GetNumAtoms()
    radii = np.zeros(n_atoms, dtype=np.float64)

    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        atomic_num = atom.GetAtomicNum()

        if atomic_num == 6 and atom.GetIsAromatic():
            radii[i] = AROMATIC_CARBON_RADIUS
        else:
            radii[i] = VINARDO_RADII.get(atomic_num, DEFAULT_RADIUS)

    return radii


def get_smarts_matches(mol: Chem.Mol, smarts: str) -> NDArray[np.bool_]:
    """Get boolean mask for atoms matching a SMARTS pattern.

    Args:
        mol: RDKit Mol object.
        smarts: SMARTS pattern string.

    Returns:
        Boolean array (n_atoms,) where True indicates a match.
    """
    n_atoms = mol.GetNumAtoms()
    mask = np.zeros(n_atoms, dtype=bool)

    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        logger.warning(f"Invalid SMARTS pattern: {smarts}")
        return mask

    # maxMatches defaults to 1000 in RDKit, which silently truncates results
    # for large molecules (e.g. a protein with >1000 hydrophobic atoms).
    matches = mol.GetSubstructMatches(pattern, maxMatches=mol.GetNumAtoms())
    for match in matches:
        for atom_idx in match:
            mask[atom_idx] = True

    return mask


def get_atom_typing(mol: Chem.Mol) -> AtomTyping:
    """Classify atoms for empirical scoring.

    Assigns van der Waals radii and identifies hydrophobic atoms,
    H-bond donors, and H-bond acceptors using SMARTS patterns.

    Args:
        mol: RDKit Mol object.

    Returns:
        AtomTyping with radii and boolean masks.
    """
    radii = get_atom_radii(mol)
    is_hydrophobic = get_smarts_matches(mol, HYDROPHOBIC_SMARTS)
    is_hbond_donor = get_smarts_matches(mol, HBOND_DONOR_SMARTS)
    is_hbond_acceptor = get_smarts_matches(mol, HBOND_ACCEPTOR_SMARTS)

    return AtomTyping(
        radii=radii,
        is_hydrophobic=is_hydrophobic,
        is_hbond_donor=is_hbond_donor,
        is_hbond_acceptor=is_hbond_acceptor,
    )


# =============================================================================
# Distance Calculation
# =============================================================================


def compute_surface_distances(
    ligand_coords: Coords,
    protein_coords: Coords,
    ligand_radii: NDArray[np.floating],
    protein_radii: NDArray[np.floating],
) -> DistanceMatrix:
    """Compute surface-to-surface distances between atom pairs.

    Surface distance d = r_ij - R_i - R_j, where r_ij is the interatomic
    distance and R_i, R_j are atomic radii.

    Args:
        ligand_coords: Ligand atom coordinates (n_ligand, 3).
        protein_coords: Protein atom coordinates (n_protein, 3).
        ligand_radii: Ligand atomic radii (n_ligand,).
        protein_radii: Protein atomic radii (n_protein,).

    Returns:
        Surface distance matrix (n_ligand, n_protein).
    """
    r_ij: DistanceMatrix = cdist(ligand_coords, protein_coords, metric="euclidean")
    surface_dist: DistanceMatrix = (
        r_ij - ligand_radii[:, np.newaxis] - protein_radii[np.newaxis, :]
    )
    return surface_dist


# =============================================================================
# Interaction Terms
# =============================================================================


def gauss1_term(
    distances: DistanceMatrix,
    offset: float = 0.0,
    width: float = 0.8,
) -> NDArray[np.floating]:
    """Compute Gaussian attractive term.

    Gauss1(d) = exp(-((d - offset) / width)^2)

    Args:
        distances: Surface distance matrix.
        offset: Gaussian center (default 0.0 for Vinardo).
        width: Gaussian width (default 0.8 for Vinardo).

    Returns:
        Gaussian values for each atom pair.
    """
    result: NDArray[np.floating] = np.exp(-(((distances - offset) / width) ** 2))
    return result


def repulsion_term(distances: DistanceMatrix) -> NDArray[np.floating]:
    """Compute repulsion term for clashing atoms.

    Repulsion(d) = d^2 if d < 0, else 0

    Args:
        distances: Surface distance matrix.

    Returns:
        Repulsion penalty for each atom pair.
    """
    result = np.zeros_like(distances)
    mask = distances < 0
    result[mask] = distances[mask] ** 2
    return result


def hydrophobic_term(
    distances: DistanceMatrix,
    ligand_hydrophobic: NDArray[np.bool_],
    protein_hydrophobic: NDArray[np.bool_],
    good_cutoff: float = 0.0,
    bad_cutoff: float = 2.5,
) -> NDArray[np.floating]:
    """Compute hydrophobic interaction term.

    Hydrophobic(d) = 1 if d <= good_cutoff
                   = (bad - d) / (bad - good) if good < d < bad
                   = 0 if d >= bad_cutoff

    Only applies to hydrophobic-hydrophobic atom pairs.

    Args:
        distances: Surface distance matrix.
        ligand_hydrophobic: Boolean mask for ligand hydrophobic atoms.
        protein_hydrophobic: Boolean mask for protein hydrophobic atoms.
        good_cutoff: Inner distance cutoff (p1).
        bad_cutoff: Outer distance cutoff (p2).

    Returns:
        Hydrophobic interaction values.
    """
    pair_mask = ligand_hydrophobic[:, np.newaxis] & protein_hydrophobic[np.newaxis, :]
    result = np.zeros_like(distances)
    inner_mask = (distances <= good_cutoff) & pair_mask
    result[inner_mask] = 1.0
    trans_mask = (distances > good_cutoff) & (distances < bad_cutoff) & pair_mask
    range_size = bad_cutoff - good_cutoff
    result[trans_mask] = (bad_cutoff - distances[trans_mask]) / range_size

    return result


def hbond_term(
    distances: DistanceMatrix,
    ligand_donor: NDArray[np.bool_],
    ligand_acceptor: NDArray[np.bool_],
    protein_donor: NDArray[np.bool_],
    protein_acceptor: NDArray[np.bool_],
    good_cutoff: float = -0.6,
) -> NDArray[np.floating]:
    """Compute hydrogen bonding term.

    HBond(d) = 1 if d <= good_cutoff
             = -d / (-good_cutoff) if good_cutoff < d < 0
             = 0 if d >= 0

    Applies to donor-acceptor pairs (ligand donor - protein acceptor
    or ligand acceptor - protein donor).

    Args:
        distances: Surface distance matrix.
        ligand_donor: Boolean mask for ligand H-bond donors.
        ligand_acceptor: Boolean mask for ligand H-bond acceptors.
        protein_donor: Boolean mask for protein H-bond donors.
        protein_acceptor: Boolean mask for protein H-bond acceptors.
        good_cutoff: Inner cutoff (h1).

    Returns:
        H-bond interaction values.
    """
    pair_mask = (ligand_donor[:, np.newaxis] & protein_acceptor[np.newaxis, :]) | (
        ligand_acceptor[:, np.newaxis] & protein_donor[np.newaxis, :]
    )
    result = np.zeros_like(distances)
    inner_mask = (distances <= good_cutoff) & pair_mask
    result[inner_mask] = 1.0
    trans_mask = (distances > good_cutoff) & (distances < 0) & pair_mask
    result[trans_mask] = -distances[trans_mask] / (-good_cutoff)

    return result


# =============================================================================
# Neighbor cutoff (sparse scoring)
# =============================================================================

# Euclidean cutoff (Angstroms) beyond which no atom pair contributes. All terms
# are exactly zero past hydro_bad=2.5 Å *surface* distance; with the largest
# Vinardo radii summing to ~4.4 Å, a pair at 8 Å euclidean has surface distance
# >=3.6 Å, where even the Gaussian tail is ~1e-9. So 8 Å is numerically exact.
INTERACTION_CUTOFF: float = 8.0

# Verlet skin margin (Angstroms). A reused neighbor list is built at
# ``INTERACTION_CUTOFF + NEIGHBOR_SKIN`` and stays exact until a ligand atom
# drifts past the skin, so the per-minimize rebuild count trades against pairs
# scored per call. 2.0 Å is a safe default for local L-BFGS-B trajectories.
NEIGHBOR_SKIN: float = 2.0


def build_protein_tree(protein_coords: np.ndarray) -> cKDTree:
    """Build a KD-tree over protein atoms for neighbor queries.

    Build once per receptor and pass to the scoring functions to enable the
    sparse (cutoff) path, whose cost scales with the number of nearby atoms
    rather than the full protein.

    Args:
        protein_coords: Protein atom coordinates (n_atoms, 3).

    Returns:
        cKDTree over the protein coordinates.
    """
    return cKDTree(protein_coords)


def _neighbor_pairs(
    ligand_coords: np.ndarray,
    protein_tree: cKDTree,
    cutoff: float,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Ligand/protein atom index pairs within ``cutoff`` (euclidean).

    Args:
        ligand_coords: Ligand atom coordinates (n_ligand, 3).
        protein_tree: KD-tree over protein atoms.
        cutoff: Euclidean cutoff in Angstroms.

    Returns:
        Tuple (i_idx, j_idx) of ligand and protein atom indices for each pair.
    """
    neighbors = protein_tree.query_ball_point(ligand_coords, r=cutoff)
    counts = np.fromiter(
        (len(n) for n in neighbors), dtype=np.intp, count=len(neighbors)
    )
    if counts.sum() == 0:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    i_idx = np.repeat(np.arange(len(neighbors), dtype=np.intp), counts)
    j_idx = np.concatenate([np.asarray(n, dtype=np.intp) for n in neighbors if n])
    return i_idx, j_idx


def _pair_term_sums_and_grad(
    d: NDArray[np.floating],
    hydro_pair: NDArray[np.bool_],
    hbond_pair: NDArray[np.bool_],
    params: EmpiricalParams,
    torsion_divisor: float,
    weights: NDArray[np.floating] | float,
) -> tuple[tuple[float, float, float, float], NDArray[np.floating]]:
    """Per-pair Vinardo term sums and ``d(score)/d(surface_distance)``.

    Elementwise in ``d`` so it serves both the dense (2-D distance matrix) and
    sparse (1-D neighbor-pair) paths, keeping the term math in one place.

    Args:
        d: Surface distances, any shape.
        hydro_pair: Hydrophobic-pair mask, broadcastable to ``d``.
        hbond_pair: H-bond donor/acceptor-pair mask, broadcastable to ``d``.
        params: Scoring parameters.
        torsion_divisor: ``1 + w_rot * n_rot`` (applied to df_dd only).
        weights: Per-pair occupancy weight (protein-atom weight broadcast to
            ``d``); scales each term sum and df_dd. Use ``1.0`` for unweighted.

    Returns:
        ((gauss1_raw, repulsion_raw, hydrophobic_raw, hbond_raw), df_dd) where
        df_dd has the same shape as ``d``.
    """
    z = (d - params.gauss1_offset) / params.gauss1_width
    g1 = np.exp(-(z**2))

    rep_mask = d < 0
    rep = np.where(rep_mask, d**2, 0.0)

    range_hydro = params.hydro_bad - params.hydro_good
    hydro_inner = (d <= params.hydro_good) & hydro_pair
    hydro_trans = (d > params.hydro_good) & (d < params.hydro_bad) & hydro_pair
    hydro = np.where(
        hydro_inner,
        1.0,
        np.where(hydro_trans, (params.hydro_bad - d) / range_hydro, 0.0),
    )

    hb_inner = (d <= params.hbond_good) & hbond_pair
    hb_trans = (d > params.hbond_good) & (d < 0) & hbond_pair
    hb = np.where(hb_inner, 1.0, np.where(hb_trans, -d / (-params.hbond_good), 0.0))

    raws = (
        float(np.sum(g1 * weights)),
        float(np.sum(rep * weights)),
        float(np.sum(hydro * weights)),
        float(np.sum(hb * weights)),
    )

    df_dd = (
        (
            params.w_gauss1 * (-2.0 * z / params.gauss1_width) * g1
            + params.w_repulsion * np.where(rep_mask, 2.0 * d, 0.0)
            + params.w_hydrophobic * np.where(hydro_trans, -1.0 / range_hydro, 0.0)
            + params.w_hbond * np.where(hb_trans, -1.0 / (-params.hbond_good), 0.0)
        )
        * weights
        / torsion_divisor
    )

    return raws, df_dd


# Lazily-bound numba kernel. Imported on first use (not at module import) so the
# ~0.3 s numba import stays off ``score.py``'s widely-imported path / CLI startup.
_SCORE_GRAD_KERNEL = None


def _get_score_grad_kernel():
    """Return the cached numba ``score_grad_pairs`` kernel, importing on first use."""
    global _SCORE_GRAD_KERNEL
    if _SCORE_GRAD_KERNEL is None:
        from cmxflow.operators.dock._kernels import score_grad_pairs

        _SCORE_GRAD_KERNEL = score_grad_pairs
    return _SCORE_GRAD_KERNEL


_SCORE_POCKET_KERNEL = None
_SCORE_POCKET_BATCH_KERNEL = None


def _get_score_pocket_kernel():
    """Return the cached numba ``score_pocket`` kernel, importing on first use."""
    global _SCORE_POCKET_KERNEL
    if _SCORE_POCKET_KERNEL is None:
        from cmxflow.operators.dock._kernels import score_pocket

        _SCORE_POCKET_KERNEL = score_pocket
    return _SCORE_POCKET_KERNEL


def _get_score_pocket_batch_kernel():
    """Return the cached numba ``score_pocket_batch`` kernel, import on first use."""
    global _SCORE_POCKET_BATCH_KERNEL
    if _SCORE_POCKET_BATCH_KERNEL is None:
        from cmxflow.operators.dock._kernels import score_pocket_batch

        _SCORE_POCKET_BATCH_KERNEL = score_pocket_batch
    return _SCORE_POCKET_BATCH_KERNEL


def build_pocket_subset(
    protein_coords: np.ndarray,
    protein_typing: "AtomTyping",
    anchor: np.ndarray,
    radius: float,
    protein_tree: cKDTree | None = None,
) -> tuple[np.ndarray, "AtomTyping"]:
    """Protein atoms within ``radius`` of ``anchor`` (coords + sliced typing).

    Used to pre-screen docking restarts: when every candidate placement of a
    molecule lies within a bounded distance of the binding-site anchor, the
    atoms that could contact *any* placement all fall in one ball around the
    anchor. Scoring against this fixed subset is exact (atoms outside the ball
    are beyond ``INTERACTION_CUTOFF`` of every placement) and pays the KD-tree
    query once instead of once per candidate.

    Args:
        protein_coords: Full protein atom coordinates (n_atoms, 3).
        protein_typing: Full protein atom typing.
        anchor: Binding-site anchor point (3,).
        radius: Inclusion radius (Angstroms).
        protein_tree: Optional cached KD-tree over ``protein_coords``.

    Returns:
        ``(pocket_coords, pocket_typing)`` restricted to the in-radius atoms.
    """
    tree = (
        protein_tree if protein_tree is not None else build_protein_tree(protein_coords)
    )
    idx = np.asarray(tree.query_ball_point(anchor, r=radius), dtype=np.intp)
    pocket_coords = np.ascontiguousarray(protein_coords[idx])
    pocket_typing = AtomTyping(
        radii=protein_typing.radii[idx],
        is_hydrophobic=protein_typing.is_hydrophobic[idx],
        is_hbond_donor=protein_typing.is_hbond_donor[idx],
        is_hbond_acceptor=protein_typing.is_hbond_acceptor[idx],
        weights=protein_typing.weights[idx],
    )
    return pocket_coords, pocket_typing


def empirical_score_pocket(
    ligand_coords: np.ndarray,
    pocket_coords: np.ndarray,
    pocket_typing: "AtomTyping",
    ligand_typing: "AtomTyping",
    params: EmpiricalParams,
    torsion_divisor: float,
    cutoff: float = INTERACTION_CUTOFF,
) -> float:
    """Empirical score over a pre-built pocket subset -- score only, no gradient.

    Restart-screening hot path. Pairs every ligand atom with every pocket atom in
    the numba ``score_pocket`` kernel (cutoff-gated), so there is no per-call
    KD-tree query. Numerically equal to ``empirical_score_cached(...).total`` when
    the pocket subset contains every protein atom within ``cutoff`` of the ligand.

    Args:
        ligand_coords: Ligand heavy-atom coordinates (n_lig, 3).
        pocket_coords: Pocket protein coordinates (n_pocket, 3).
        pocket_typing: Pocket atom typing (from ``build_pocket_subset``).
        ligand_typing: Ligand atom typing.
        params: Scoring parameters.
        torsion_divisor: ``1 + w_rot * n_rot`` (fixed topology; computed once).
        cutoff: Euclidean interaction cutoff.

    Returns:
        Scalar empirical score.
    """
    kernel = _get_score_pocket_kernel()
    g1, rep, hydro, hb = kernel(
        np.ascontiguousarray(ligand_coords),
        pocket_coords,
        ligand_typing.radii,
        pocket_typing.radii,
        ligand_typing.is_hydrophobic,
        pocket_typing.is_hydrophobic,
        ligand_typing.is_hbond_donor,
        ligand_typing.is_hbond_acceptor,
        pocket_typing.is_hbond_donor,
        pocket_typing.is_hbond_acceptor,
        pocket_typing.weights,
        params.gauss1_offset,
        params.gauss1_width,
        params.hydro_good,
        params.hydro_bad,
        params.hbond_good,
        cutoff,
    )
    score = (
        params.w_gauss1 * g1
        + params.w_repulsion * rep
        + params.w_hydrophobic * hydro
        + params.w_hbond * hb
    ) / torsion_divisor
    return float(score)


def empirical_score_pocket_batch(
    coords_batch: np.ndarray,
    pocket_coords: np.ndarray,
    pocket_typing: "AtomTyping",
    ligand_typing: "AtomTyping",
    params: EmpiricalParams,
    torsion_divisor: float,
    cutoff: float = INTERACTION_CUTOFF,
) -> NDArray[np.floating]:
    """Empirical score for K candidate poses over a pre-built pocket subset.

    Batched restart-screening hot path: one numba dispatch scores the whole
    candidate grid (vs one call per candidate), and the caller generates the
    coordinates with vectorized numpy rather than per-candidate RDKit conformer
    copies. Per-pose result equals :func:`empirical_score_pocket`.

    Args:
        coords_batch: Candidate ligand coordinates (K, n_lig, 3).
        pocket_coords: Pocket protein coordinates (n_pocket, 3).
        pocket_typing: Pocket atom typing (from ``build_pocket_subset``).
        ligand_typing: Ligand atom typing.
        params: Scoring parameters.
        torsion_divisor: ``1 + w_rot * n_rot`` (fixed topology; computed once).
        cutoff: Euclidean interaction cutoff.

    Returns:
        ``(K,)`` array of empirical scores.
    """
    kernel = _get_score_pocket_batch_kernel()
    raws = kernel(
        np.ascontiguousarray(coords_batch, dtype=np.float64),
        pocket_coords,
        ligand_typing.radii,
        pocket_typing.radii,
        ligand_typing.is_hydrophobic,
        pocket_typing.is_hydrophobic,
        ligand_typing.is_hbond_donor,
        ligand_typing.is_hbond_acceptor,
        pocket_typing.is_hbond_donor,
        pocket_typing.is_hbond_acceptor,
        pocket_typing.weights,
        params.gauss1_offset,
        params.gauss1_width,
        params.hydro_good,
        params.hydro_bad,
        params.hbond_good,
        cutoff,
    )
    weights = np.array(
        [params.w_gauss1, params.w_repulsion, params.w_hydrophobic, params.w_hbond]
    )
    return (raws @ weights) / torsion_divisor


def _sparse_score_grad(
    ligand_coords: NDArray[np.floating],
    protein_coords: np.ndarray,
    i_idx: NDArray[np.intp],
    j_idx: NDArray[np.intp],
    ligand_typing: "AtomTyping",
    protein_typing: "AtomTyping",
    params: EmpiricalParams,
    inv_divisor: float,
    cutoff: float = INTERACTION_CUTOFF,
) -> tuple[float, float, float, float, NDArray]:
    """Gather per-pair arrays for the neighbor list and run the numba kernel.

    Shared by the sparse paths of ``empirical_score_cached`` (gradient ignored)
    and ``empirical_score_and_grad_cached``. The KD-tree query itself stays in
    the caller; this only fuses the gather + term sums + gradient scatter.

    Args:
        ligand_coords: Ligand heavy-atom coordinates (n_lig, 3).
        protein_coords: Protein coordinates (n_prot, 3).
        i_idx: Ligand atom index per neighbor pair.
        j_idx: Protein atom index per neighbor pair.
        ligand_typing: Ligand atom typing.
        protein_typing: Protein atom typing.
        params: Scoring parameters.
        inv_divisor: ``1 / (1 + w_rot * n_rot)`` (applied to the gradient only).
        cutoff: Euclidean core cutoff passed to the kernel's Verlet gate. Pairs
            already within ``cutoff`` (the per-call path) are unaffected; a
            skin-padded Verlet list is trimmed back to ``cutoff`` here.

    Returns:
        ``(gauss1_raw, repulsion_raw, hydrophobic_raw, hbond_raw, atom_grad)``.
    """
    kernel = _get_score_grad_kernel()
    # ``asarray`` (not ``astype``) avoids a copy when indices are already int64
    # (intp on 64-bit), which is the common case for both the per-call and
    # reused-Verlet paths.
    result = kernel(
        np.ascontiguousarray(ligand_coords),
        protein_coords,
        np.asarray(i_idx, dtype=np.int64),
        np.asarray(j_idx, dtype=np.int64),
        ligand_typing.radii,
        protein_typing.radii,
        ligand_typing.is_hydrophobic,
        protein_typing.is_hydrophobic,
        ligand_typing.is_hbond_donor,
        ligand_typing.is_hbond_acceptor,
        protein_typing.is_hbond_donor,
        protein_typing.is_hbond_acceptor,
        params.w_gauss1,
        params.w_repulsion,
        params.w_hydrophobic,
        params.w_hbond,
        params.gauss1_offset,
        params.gauss1_width,
        params.hydro_good,
        params.hydro_bad,
        params.hbond_good,
        cutoff,
        protein_typing.weights,
        inv_divisor,
    )
    return cast(tuple[float, float, float, float, NDArray], result)


# =============================================================================
# Intramolecular Ligand Energy
# =============================================================================


@dataclass(frozen=True)
class IntramolecularPairs:
    """Precomputed conformer-dependent intramolecular ligand atom pairs.

    Built once per ligand (topology is fixed). Should contain only pairs that
    change with the pose — i.e. 1-4-and-beyond pairs that cross a rotatable bond.
    Within-rigid-fragment pairs are constant during pose optimization and add
    nothing to the gradient or argmin, so they are excluded.

    Attributes:
        i_idx: First heavy-atom index of each pair.
        j_idx: Second heavy-atom index of each pair.
        radii_sum: Sum of the two atomic radii per pair (surface-distance offset).
        hydro_pair: Hydrophobic-pair mask per pair.
        hbond_pair: H-bond donor/acceptor-pair mask per pair.
    """

    i_idx: NDArray[np.intp]
    j_idx: NDArray[np.intp]
    radii_sum: NDArray[np.floating]
    hydro_pair: NDArray[np.bool_]
    hbond_pair: NDArray[np.bool_]


def intramolecular_score_and_grad(
    ligand_coords: NDArray[np.floating],
    pairs: IntramolecularPairs,
    params: EmpiricalParams,
    torsion_divisor: float = 1.0,
) -> tuple[float, NDArray]:
    """Intramolecular ligand energy and per-atom gradient (same Vinardo terms).

    Uses the same terms/weights as the intermolecular score. Both atoms of each
    pair are mobile, so the gradient scatters to atom i (+) and atom j (-).

    Args:
        ligand_coords: Heavy-atom coordinates (n_heavy, 3).
        pairs: Precomputed intramolecular pairs from the optimizer.
        params: Scoring parameters.
        torsion_divisor: ``1 + w_rot * n_rot``, matching intermolecular scaling.

    Returns:
        (score, atom_grad) where atom_grad has shape (n_heavy, 3).
    """
    n = ligand_coords.shape[0]
    if pairs.i_idx.size == 0:
        return 0.0, np.zeros((n, 3))

    diff = ligand_coords[pairs.i_idx] - ligand_coords[pairs.j_idx]
    eucl = np.linalg.norm(diff, axis=-1)
    d = eucl - pairs.radii_sum
    (g1_raw, rep_raw, hydro_raw, hb_raw), df_dd = _pair_term_sums_and_grad(
        d, pairs.hydro_pair, pairs.hbond_pair, params, torsion_divisor, 1.0
    )
    score = (
        params.w_gauss1 * g1_raw
        + params.w_repulsion * rep_raw
        + params.w_hydrophobic * hydro_raw
        + params.w_hbond * hb_raw
    ) / torsion_divisor

    safe_eucl = np.where(eucl > 1e-8, eucl, 1.0)
    unit = np.where(eucl[:, None] > 1e-8, diff / safe_eucl[:, None], 0.0)
    contrib = df_dd[:, None] * unit  # (n_pairs, 3)
    atom_grad = np.zeros((n, 3))
    for axis in range(3):
        atom_grad[:, axis] = np.bincount(
            pairs.i_idx, weights=contrib[:, axis], minlength=n
        ) - np.bincount(pairs.j_idx, weights=contrib[:, axis], minlength=n)
    return float(score), atom_grad


# =============================================================================
# Main Scoring Functions
# =============================================================================


def empirical_score(
    ligand_mol: Chem.Mol,
    protein_mol: Chem.Mol,
    ligand_conf_id: int = 0,
    protein_conf_id: int = 0,
    params: EmpiricalParams | None = None,
) -> float:
    """Compute docking score for a ligand-protein complex.

    Convenience function that accepts full RDKit mol objects. Strips H
    internally. For repeated scoring against the same protein, prefer
    ``empirical_score_cached`` with pre-computed protein data.

    Score = (w_gauss1 * sum(Gauss1) + w_rep * sum(Repulsion)
          + w_hydro * sum(Hydrophobic) + w_hbond * sum(HBond))
          / (1 + w_rot * N_rot)

    Lower (more negative) scores indicate better binding.

    Args:
        ligand_mol: Ligand RDKit Mol with 3D coordinates.
        protein_mol: Protein RDKit Mol with 3D coordinates.
        ligand_conf_id: Ligand conformer ID to use.
        protein_conf_id: Protein conformer ID to use.
        params: Scoring parameters. If None, uses defaults.

    Returns:
        Docking score (kcal/mol-like units).

    Raises:
        ValueError: If molecules lack 3D conformers.
    """
    if params is None:
        params = EmpiricalParams()

    ligand_heavy = Chem.RemoveAllHs(ligand_mol)
    protein_heavy = Chem.RemoveAllHs(protein_mol)

    if ligand_heavy.GetNumConformers() == 0:
        raise ValueError("Ligand molecule has no conformers")
    if protein_heavy.GetNumConformers() == 0:
        raise ValueError("Protein molecule has no conformers")

    protein_conf = protein_heavy.GetConformer(protein_conf_id)
    protein_coords = np.array(protein_conf.GetPositions())
    protein_typing = get_atom_typing(protein_heavy)

    return empirical_score_cached(
        ligand_heavy, protein_coords, protein_typing, ligand_conf_id, params
    ).total


def empirical_score_cached(
    ligand_heavy: Chem.Mol,
    protein_coords: np.ndarray,
    protein_typing: AtomTyping,
    ligand_conf_id: int = 0,
    params: EmpiricalParams | None = None,
    protein_tree: cKDTree | None = None,
    cutoff: float = INTERACTION_CUTOFF,
    ligand_typing: AtomTyping | None = None,
) -> ScoreComponents:
    """Compute docking score and component breakdown with pre-computed protein data.

    Reporting path. Input ligand must be heavy-atom only — call
    ``Chem.RemoveAllHs()`` before passing.

    Args:
        ligand_heavy: Heavy-atom-only ligand RDKit Mol with 3D coordinates.
        protein_coords: Pre-computed protein atom 3D coordinates (n_atoms, 3).
        protein_typing: Pre-computed protein atom typing from get_atom_typing().
        ligand_conf_id: Ligand conformer ID to use.
        params: Scoring parameters. If None, uses defaults.
        protein_tree: Optional KD-tree from ``build_protein_tree``. When given,
            uses the sparse cutoff path (cost scales with nearby atoms, not the
            full protein); numerically identical to the dense path within ~1e-8.
        cutoff: Euclidean neighbor cutoff in Angstroms (sparse path only).
        ligand_typing: Optional pre-computed ligand typing. Pass it to avoid
            recomputing SMARTS matches every call in an optimization loop
            (ligand topology is fixed; only coordinates change).

    Returns:
        ScoreComponents with per-term raw sums, weights, and total score.

    Raises:
        ValueError: If ligand molecule lacks 3D conformers.
    """
    if params is None:
        params = EmpiricalParams()

    if ligand_heavy.GetNumConformers() == 0:
        raise ValueError("Ligand molecule has no conformers")

    ligand_conf = ligand_heavy.GetConformer(ligand_conf_id)
    ligand_coords = np.array(ligand_conf.GetPositions())
    if ligand_typing is None:
        ligand_typing = get_atom_typing(ligand_heavy)

    if protein_tree is not None:
        i_idx, j_idx = _neighbor_pairs(ligand_coords, protein_tree, cutoff)
        # Gradient unused on the score-only path; inv_divisor is irrelevant.
        g1_raw, rep_raw, hydro_raw, hb_raw, _ = _sparse_score_grad(
            ligand_coords,
            protein_coords,
            i_idx,
            j_idx,
            ligand_typing,
            protein_typing,
            params,
            1.0,
            cutoff=cutoff,
        )
    else:
        distances = compute_surface_distances(
            ligand_coords,
            protein_coords,
            ligand_typing.radii,
            protein_typing.radii,
        )

        # Occupancy weight per protein atom, broadcast across ligand-atom rows.
        w = protein_typing.weights[None, :]
        g1_raw = float(
            np.sum(
                gauss1_term(distances, params.gauss1_offset, params.gauss1_width) * w
            )
        )
        rep_raw = float(np.sum(repulsion_term(distances) * w))
        hydro_raw = float(
            np.sum(
                hydrophobic_term(
                    distances,
                    ligand_typing.is_hydrophobic,
                    protein_typing.is_hydrophobic,
                    params.hydro_good,
                    params.hydro_bad,
                )
                * w
            )
        )
        hb_raw = float(
            np.sum(
                hbond_term(
                    distances,
                    ligand_typing.is_hbond_donor,
                    ligand_typing.is_hbond_acceptor,
                    protein_typing.is_hbond_donor,
                    protein_typing.is_hbond_acceptor,
                    params.hbond_good,
                )
                * w
            )
        )

    n_rot = rdMolDescriptors.CalcNumRotatableBonds(ligand_heavy, strict=False)

    return ScoreComponents(
        gauss1_raw=g1_raw,
        repulsion_raw=rep_raw,
        hydrophobic_raw=hydro_raw,
        hbond_raw=hb_raw,
        w_gauss1=params.w_gauss1,
        w_repulsion=params.w_repulsion,
        w_hydrophobic=params.w_hydrophobic,
        w_hbond=params.w_hbond,
        n_rot=n_rot,
        w_rot=params.w_rot,
    )


def empirical_score_and_grad_cached(
    ligand_heavy: Chem.Mol,
    protein_coords: np.ndarray,
    protein_typing: AtomTyping,
    ligand_conf_id: int = 0,
    params: EmpiricalParams | None = None,
    protein_tree: cKDTree | None = None,
    cutoff: float = INTERACTION_CUTOFF,
    ligand_typing: AtomTyping | None = None,
) -> tuple[float, NDArray]:
    """Compute docking score and per-heavy-atom gradient with pre-computed protein data.

    Optimizer path. Input ligand must be heavy-atom only — call
    ``Chem.RemoveAllHs()`` before passing. Score and gradient are computed
    in a single vectorized pass.

    Args:
        ligand_heavy: Heavy-atom-only ligand RDKit Mol with 3D coordinates.
        protein_coords: Pre-computed protein atom 3D coordinates (n_atoms, 3).
        protein_typing: Pre-computed protein atom typing from get_atom_typing().
        ligand_conf_id: Ligand conformer ID to use.
        params: Scoring parameters. If None, uses defaults.
        protein_tree: Optional KD-tree from ``build_protein_tree``. When given,
            uses the sparse cutoff path (cost scales with nearby atoms, not the
            full protein); numerically identical to the dense path within ~1e-8.
        cutoff: Euclidean neighbor cutoff in Angstroms (sparse path only).
        ligand_typing: Optional pre-computed ligand typing. Pass it to avoid
            recomputing SMARTS matches every call in an optimization loop
            (ligand topology is fixed; only coordinates change).

    Returns:
        Tuple of (score, atom_grad) where atom_grad has shape (n_heavy, 3).
        atom_grad[i] = dScore/d(pos_i).

    Raises:
        ValueError: If ligand molecule lacks 3D conformers.
    """
    if params is None:
        params = EmpiricalParams()

    if ligand_heavy.GetNumConformers() == 0:
        raise ValueError("Ligand molecule has no conformers")

    ligand_coords = np.array(ligand_heavy.GetConformer(ligand_conf_id).GetPositions())
    if ligand_typing is None:
        ligand_typing = get_atom_typing(ligand_heavy)
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(ligand_heavy, strict=False)
    torsion_divisor = 1.0 + params.w_rot * n_rot

    if protein_tree is not None:
        # --- Sparse path: only atom pairs within cutoff ---
        # The numba kernel fuses the gather, term sums, and gradient scatter into
        # a single pass; the gradient is scaled by 1/torsion_divisor internally.
        i_idx, j_idx = _neighbor_pairs(ligand_coords, protein_tree, cutoff)
        g1_raw, rep_raw, hydro_raw, hb_raw, atom_grad = _sparse_score_grad(
            ligand_coords,
            protein_coords,
            i_idx,
            j_idx,
            ligand_typing,
            protein_typing,
            params,
            1.0 / torsion_divisor,
            cutoff=cutoff,
        )
        score = (
            params.w_gauss1 * g1_raw
            + params.w_repulsion * rep_raw
            + params.w_hydrophobic * hydro_raw
            + params.w_hbond * hb_raw
        ) / torsion_divisor
        return float(score), atom_grad

    # --- Dense path: full (n_lig, n_prot) ---
    diff = ligand_coords[:, None, :] - protein_coords[None, :, :]
    eucl = np.linalg.norm(diff, axis=-1)
    d = eucl - ligand_typing.radii[:, None] - protein_typing.radii[None, :]

    hydro_pair = (
        ligand_typing.is_hydrophobic[:, None] & protein_typing.is_hydrophobic[None, :]
    )
    hbond_pair = (
        ligand_typing.is_hbond_donor[:, None]
        & protein_typing.is_hbond_acceptor[None, :]
    ) | (
        ligand_typing.is_hbond_acceptor[:, None]
        & protein_typing.is_hbond_donor[None, :]
    )
    (g1_raw, rep_raw, hydro_raw, hb_raw), df_dd = _pair_term_sums_and_grad(
        d,
        hydro_pair,
        hbond_pair,
        params,
        torsion_divisor,
        protein_typing.weights[None, :],
    )
    score = (
        params.w_gauss1 * g1_raw
        + params.w_repulsion * rep_raw
        + params.w_hydrophobic * hydro_raw
        + params.w_hbond * hb_raw
    ) / torsion_divisor

    # Chain rule: atom gradient g_i = sum_j df_dd[i,j] * unit[i,j]
    safe_eucl = np.where(eucl > 1e-8, eucl, 1.0)
    unit = diff / safe_eucl[:, :, None]
    unit = np.where(eucl[:, :, None] > 1e-8, unit, 0.0)
    atom_grad = np.einsum("ij,ijk->ik", df_dd, unit)  # (n_heavy, 3)

    return float(score), atom_grad


class NeighborList:
    """Reusable (Verlet) ligand/protein neighbor-pair list for one local minimize.

    The per-call KD-tree query in ``empirical_score_and_grad_cached`` dominates
    the optimizer hot path, yet the ligand barely moves between gradient evals.
    This builds the pair list once at ``cutoff + skin`` and reuses it, rebuilding
    only when a ligand atom has moved more than ``skin`` since the last build.
    The kernel gates each pair at ``cutoff`` (see ``score_grad_pairs``), so the
    scored set is identical to a fresh ``cutoff`` query for as long as no atom has
    drifted past the skin -- which the rebuild check guarantees. Results therefore
    match the per-call path to summation order (~1e-13).

    Built once per ``optimize_pose_cached`` start; persists across all L-BFGS-B
    iterations and basin hops of that start.
    """

    def __init__(
        self,
        protein_coords: np.ndarray,
        protein_typing: AtomTyping,
        ligand_typing: AtomTyping,
        cutoff: float = INTERACTION_CUTOFF,
        skin: float = NEIGHBOR_SKIN,
        protein_tree: cKDTree | None = None,
    ) -> None:
        self.protein_coords = np.ascontiguousarray(protein_coords)
        self.protein_typing = protein_typing
        self.ligand_typing = ligand_typing
        self.cutoff = cutoff
        self.skin = skin
        # Reuse the caller's tree when given (it spans the same atoms); the build
        # radius (cutoff + skin) is just a query argument, not baked into the tree.
        self._tree = (
            protein_tree
            if protein_tree is not None
            else build_protein_tree(self.protein_coords)
        )
        self._ref_coords: np.ndarray | None = None  # ligand coords at last build
        self.i_idx: NDArray[np.intp] = np.empty(0, dtype=np.intp)
        self.j_idx: NDArray[np.intp] = np.empty(0, dtype=np.intp)

    def update(self, ligand_coords: np.ndarray) -> None:
        """Rebuild the pair list iff a ligand atom moved more than ``skin``."""
        if self._ref_coords is not None:
            max_disp = np.sqrt(
                ((ligand_coords - self._ref_coords) ** 2).sum(axis=1).max()
            )
            if max_disp <= self.skin:
                return  # cached list still covers every within-cutoff pair
        self.i_idx, self.j_idx = _neighbor_pairs(
            ligand_coords, self._tree, self.cutoff + self.skin
        )
        self._ref_coords = ligand_coords.copy()


def empirical_score_and_grad_fast(
    ligand_coords: np.ndarray,
    neighbor_list: NeighborList,
    params: EmpiricalParams,
    torsion_divisor: float,
) -> tuple[float, NDArray]:
    """Score + per-atom gradient over a reused Verlet neighbor list.

    Optimizer hot path. Unlike ``empirical_score_and_grad_cached`` this takes
    coordinates directly -- no Mol, so no per-call conformer extraction or
    ``CalcNumRotatableBonds`` -- and reuses ``neighbor_list``'s KD-tree pairs
    across evals, paying the query only on rebuilds. Numerically equal to the
    cached path (same pair set, summation-order ~1e-13) at the same ``cutoff``.

    Args:
        ligand_coords: Ligand heavy-atom coordinates (n_lig, 3).
        neighbor_list: Verlet list for this minimize; updated in place.
        params: Scoring parameters.
        torsion_divisor: ``1 + w_rot * n_rot`` (fixed topology; computed once by
            the caller). Scales the score and, via ``1/torsion_divisor``, the grad.

    Returns:
        Tuple of (score, atom_grad) with atom_grad shape (n_lig, 3).
    """
    ligand_coords = np.ascontiguousarray(ligand_coords)
    neighbor_list.update(ligand_coords)
    g1_raw, rep_raw, hydro_raw, hb_raw, atom_grad = _sparse_score_grad(
        ligand_coords,
        neighbor_list.protein_coords,
        neighbor_list.i_idx,
        neighbor_list.j_idx,
        neighbor_list.ligand_typing,
        neighbor_list.protein_typing,
        params,
        1.0 / torsion_divisor,
        cutoff=neighbor_list.cutoff,
    )
    score = (
        params.w_gauss1 * g1_raw
        + params.w_repulsion * rep_raw
        + params.w_hydrophobic * hydro_raw
        + params.w_hbond * hb_raw
    ) / torsion_divisor
    return float(score), atom_grad


# =============================================================================
# Electrostatic Complementarity
# =============================================================================


# =============================================================================
# Extensibility
# =============================================================================


class ScoringFunction(Protocol):
    """Protocol for docking scoring functions."""

    def __call__(
        self,
        ligand_mol: Chem.Mol,
        protein_mol: Chem.Mol,
        ligand_conf_id: int = 0,
        protein_conf_id: int = 0,
    ) -> float: ...


def get_scoring_function(name: str = "empirical") -> ScoringFunction:
    """Get a scoring function by name.

    Args:
        name: Scoring function name ("empirical").

    Returns:
        Callable scoring function.

    Raises:
        ValueError: If scoring function not found.
    """
    functions: dict[str, ScoringFunction] = {
        "empirical": empirical_score,
    }
    if name not in functions:
        raise ValueError(
            f"Unknown scoring function: {name}. Available: {list(functions.keys())}"
        )
    return functions[name]
