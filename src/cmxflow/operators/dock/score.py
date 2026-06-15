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
from dataclasses import dataclass
from typing import Protocol, TypeAlias

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

    Raw values are the unweighted sums over all atom pairs. Weighted
    properties divide each raw sum by the torsion divisor so that
    ``total`` equals the score returned by the scoring function.

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
        return self.w_gauss1 * self.gauss1_raw / self._torsion_divisor

    @property
    def repulsion(self) -> float:
        return self.w_repulsion * self.repulsion_raw / self._torsion_divisor

    @property
    def hydrophobic(self) -> float:
        return self.w_hydrophobic * self.hydrophobic_raw / self._torsion_divisor

    @property
    def hbond(self) -> float:
        return self.w_hbond * self.hbond_raw / self._torsion_divisor

    @property
    def total(self) -> float:
        return self.gauss1 + self.repulsion + self.hydrophobic + self.hbond


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
    "[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O;H1;+0]),$([n;H1;+0]),$([n;!H0;+1])"
    ",Li+1,Na+1,K+1,Cs+1,Mg+2,Ca+2,Mn+2,Zn+2]"
)
HBOND_ACCEPTOR_SMARTS = (
    "[$([O;H1;v2]-[!$(*=[O,N,P,S])]),$([O;H0;v2]),$([O;-]),"
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
    """

    radii: NDArray[np.floating]
    is_hydrophobic: NDArray[np.bool_]
    is_hbond_donor: NDArray[np.bool_]
    is_hbond_acceptor: NDArray[np.bool_]


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
        float(np.sum(g1)),
        float(np.sum(rep)),
        float(np.sum(hydro)),
        float(np.sum(hb)),
    )

    df_dd = (
        params.w_gauss1 * (-2.0 * z / params.gauss1_width) * g1
        + params.w_repulsion * np.where(rep_mask, 2.0 * d, 0.0)
        + params.w_hydrophobic * np.where(hydro_trans, -1.0 / range_hydro, 0.0)
        + params.w_hbond * np.where(hb_trans, -1.0 / (-params.hbond_good), 0.0)
    ) / torsion_divisor

    return raws, df_dd


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
        d, pairs.hydro_pair, pairs.hbond_pair, params, torsion_divisor
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
        diff = ligand_coords[i_idx] - protein_coords[j_idx]
        d = (
            np.linalg.norm(diff, axis=-1)
            - ligand_typing.radii[i_idx]
            - protein_typing.radii[j_idx]
        )
        hydro_pair = (
            ligand_typing.is_hydrophobic[i_idx] & protein_typing.is_hydrophobic[j_idx]
        )
        hbond_pair = (
            ligand_typing.is_hbond_donor[i_idx]
            & protein_typing.is_hbond_acceptor[j_idx]
        ) | (
            ligand_typing.is_hbond_acceptor[i_idx]
            & protein_typing.is_hbond_donor[j_idx]
        )
        (g1_raw, rep_raw, hydro_raw, hb_raw), _ = _pair_term_sums_and_grad(
            d, hydro_pair, hbond_pair, params, 1.0
        )
    else:
        distances = compute_surface_distances(
            ligand_coords,
            protein_coords,
            ligand_typing.radii,
            protein_typing.radii,
        )

        g1_raw = float(
            np.sum(gauss1_term(distances, params.gauss1_offset, params.gauss1_width))
        )
        rep_raw = float(np.sum(repulsion_term(distances)))
        hydro_raw = float(
            np.sum(
                hydrophobic_term(
                    distances,
                    ligand_typing.is_hydrophobic,
                    protein_typing.is_hydrophobic,
                    params.hydro_good,
                    params.hydro_bad,
                )
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
    n_lig = len(ligand_coords)
    torsion_divisor = 1.0 + params.w_rot * n_rot

    if protein_tree is not None:
        # --- Sparse path: only atom pairs within cutoff ---
        i_idx, j_idx = _neighbor_pairs(ligand_coords, protein_tree, cutoff)
        diff = ligand_coords[i_idx] - protein_coords[j_idx]  # (n_pairs, 3)
        eucl = np.linalg.norm(diff, axis=-1)  # (n_pairs,)
        d = eucl - ligand_typing.radii[i_idx] - protein_typing.radii[j_idx]
        hydro_pair = (
            ligand_typing.is_hydrophobic[i_idx] & protein_typing.is_hydrophobic[j_idx]
        )
        hbond_pair = (
            ligand_typing.is_hbond_donor[i_idx]
            & protein_typing.is_hbond_acceptor[j_idx]
        ) | (
            ligand_typing.is_hbond_acceptor[i_idx]
            & protein_typing.is_hbond_donor[j_idx]
        )
        (g1_raw, rep_raw, hydro_raw, hb_raw), df_dd = _pair_term_sums_and_grad(
            d, hydro_pair, hbond_pair, params, torsion_divisor
        )
        score = (
            params.w_gauss1 * g1_raw
            + params.w_repulsion * rep_raw
            + params.w_hydrophobic * hydro_raw
            + params.w_hbond * hb_raw
        ) / torsion_divisor

        # Chain rule + scatter-add onto ligand atoms (bincount per axis)
        safe_eucl = np.where(eucl > 1e-8, eucl, 1.0)
        unit = np.where(eucl[:, None] > 1e-8, diff / safe_eucl[:, None], 0.0)
        contrib = df_dd[:, None] * unit  # (n_pairs, 3)
        atom_grad = np.zeros((n_lig, 3))
        for axis in range(3):
            atom_grad[:, axis] = np.bincount(
                i_idx, weights=contrib[:, axis], minlength=n_lig
            )
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
        d, hydro_pair, hbond_pair, params, torsion_divisor
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
