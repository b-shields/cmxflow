"""Vinardo scoring function for molecular docking.

This module implements the Vinardo empirical scoring function for
protein-ligand binding affinity estimation.

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
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

# Type aliases
Coords: TypeAlias = NDArray[np.floating]
DistanceMatrix: TypeAlias = NDArray[np.floating]


# =============================================================================
# Vinardo Parameters
# =============================================================================


@dataclass(frozen=True)
class VinardoParams:
    """Vinardo scoring function parameters.

    All default values are from Quiroga & Villarreal (2016).
    Parameters can be modified for experimentation.

    Attributes:
        w_gauss1: Weight for Gaussian attractive term.
        w_repulsion: Weight for repulsion term.
        w_hydrophobic: Weight for hydrophobic interactions.
        w_hbond: Weight for hydrogen bonding.
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
HYDROPHOBIC_SMARTS = "[#6,#9,#14,#15,#16,#17,#35,#53]"
HBOND_DONOR_SMARTS = (
    "[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])"
    ",Li+1,Na+1,K+1,Cs+1,Mg+2,Ca+2,Mn+2,Zn+2]"
)
HBOND_ACCEPTOR_SMARTS = (
    "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),"
    "$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]"
)


@dataclass
class AtomTyping:
    """Atom classification for Vinardo scoring.

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

    matches = mol.GetSubstructMatches(pattern)
    for match in matches:
        for atom_idx in match:
            mask[atom_idx] = True

    return mask


def get_atom_typing(mol: Chem.Mol) -> AtomTyping:
    """Classify atoms for Vinardo scoring.

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
    # Create pair mask for hydrophobic-hydrophobic interactions
    pair_mask = ligand_hydrophobic[:, np.newaxis] & protein_hydrophobic[np.newaxis, :]

    result = np.zeros_like(distances)

    # Inner region: full interaction
    inner_mask = (distances <= good_cutoff) & pair_mask
    result[inner_mask] = 1.0

    # Transition region: linear decay
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
    # Valid H-bond pairs: donor-acceptor or acceptor-donor
    pair_mask = (ligand_donor[:, np.newaxis] & protein_acceptor[np.newaxis, :]) | (
        ligand_acceptor[:, np.newaxis] & protein_donor[np.newaxis, :]
    )

    result = np.zeros_like(distances)

    # Inner region: full interaction
    inner_mask = (distances <= good_cutoff) & pair_mask
    result[inner_mask] = 1.0

    # Transition region: linear decay to 0
    trans_mask = (distances > good_cutoff) & (distances < 0) & pair_mask
    result[trans_mask] = -distances[trans_mask] / (-good_cutoff)

    return result


# =============================================================================
# Main Scoring Function
# =============================================================================


def vinardo_score(
    ligand_mol: Chem.Mol,
    protein_mol: Chem.Mol,
    ligand_conf_id: int = 0,
    protein_conf_id: int = 0,
    params: VinardoParams | None = None,
) -> float:
    """Compute Vinardo docking score for ligand-protein complex.

    Score = w_gauss1 * sum(Gauss1) + w_rep * sum(Repulsion)
          + w_hydro * sum(Hydrophobic) + w_hbond * sum(HBond)

    Lower (more negative) scores indicate better binding.

    Args:
        ligand_mol: Ligand RDKit Mol with 3D coordinates.
        protein_mol: Protein RDKit Mol with 3D coordinates.
        ligand_conf_id: Ligand conformer ID to use.
        protein_conf_id: Protein conformer ID to use.
        params: Vinardo parameters. If None, uses defaults.

    Returns:
        Vinardo docking score (kcal/mol-like units).

    Raises:
        ValueError: If molecules lack 3D conformers.
    """
    if params is None:
        params = VinardoParams()

    # United atom
    ligand_mol = Chem.RemoveHs(ligand_mol)

    # Validate conformers
    if ligand_mol.GetNumConformers() == 0:
        raise ValueError("Ligand molecule has no conformers")
    if protein_mol.GetNumConformers() == 0:
        raise ValueError("Protein molecule has no conformers")

    # Get coordinates
    ligand_conf = ligand_mol.GetConformer(ligand_conf_id)
    protein_conf = protein_mol.GetConformer(protein_conf_id)

    ligand_coords = np.array(ligand_conf.GetPositions())
    protein_coords = np.array(protein_conf.GetPositions())

    # Get atom typing
    ligand_typing = get_atom_typing(ligand_mol)
    protein_typing = get_atom_typing(protein_mol)

    # Compute surface distances
    distances = compute_surface_distances(
        ligand_coords,
        protein_coords,
        ligand_typing.radii,
        protein_typing.radii,
    )

    # Compute interaction terms
    gauss1 = gauss1_term(distances, params.gauss1_offset, params.gauss1_width)
    repulsion = repulsion_term(distances)
    hydrophobic = hydrophobic_term(
        distances,
        ligand_typing.is_hydrophobic,
        protein_typing.is_hydrophobic,
        params.hydro_good,
        params.hydro_bad,
    )
    hbond = hbond_term(
        distances,
        ligand_typing.is_hbond_donor,
        ligand_typing.is_hbond_acceptor,
        protein_typing.is_hbond_donor,
        protein_typing.is_hbond_acceptor,
        params.hbond_good,
    )

    # Sum all pair contributions and weight
    score = (
        params.w_gauss1 * np.sum(gauss1)
        + params.w_repulsion * np.sum(repulsion)
        + params.w_hydrophobic * np.sum(hydrophobic)
        + params.w_hbond * np.sum(hbond)
    )

    return float(score)


def vinardo_score_cached(
    ligand_mol: Chem.Mol,
    protein_coords: np.ndarray,
    protein_typing: AtomTyping,
    ligand_conf_id: int = 0,
    params: VinardoParams | None = None,
) -> float:
    """Compute Vinardo docking score with pre-computed protein data.

    This is a performance-optimized version of vinardo_score() that accepts
    pre-computed protein coordinates and atom typing. Use this when scoring
    multiple ligands against the same protein to avoid redundant computation.

    Score = w_gauss1 * sum(Gauss1) + w_rep * sum(Repulsion)
          + w_hydro * sum(Hydrophobic) + w_hbond * sum(HBond)

    Lower (more negative) scores indicate better binding.

    Args:
        ligand_mol: Ligand RDKit Mol with 3D coordinates.
        protein_coords: Pre-computed protein atom 3D coordinates as numpy
            array with shape (n_atoms, 3).
        protein_typing: Pre-computed protein atom typing from get_atom_typing().
        ligand_conf_id: Ligand conformer ID to use.
        params: Vinardo parameters. If None, uses defaults.

    Returns:
        Vinardo docking score (kcal/mol-like units).

    Raises:
        ValueError: If ligand molecule lacks 3D conformers.

    Example:
        >>> protein_coords = np.array(protein.GetConformer().GetPositions())
        >>> protein_typing = get_atom_typing(protein)
        >>> for ligand in ligands:
        ...     score = vinardo_score_cached(
        ...         ligand, protein_coords, protein_typing
        ...     )
    """
    if params is None:
        params = VinardoParams()

    # United atom
    ligand_mol = Chem.RemoveHs(ligand_mol)

    # Validate conformers
    if ligand_mol.GetNumConformers() == 0:
        raise ValueError("Ligand molecule has no conformers")

    ligand_conf = ligand_mol.GetConformer(ligand_conf_id)
    ligand_coords = np.array(ligand_conf.GetPositions())
    ligand_typing = get_atom_typing(ligand_mol)

    # Compute surface distances
    distances = compute_surface_distances(
        ligand_coords,
        protein_coords,
        ligand_typing.radii,
        protein_typing.radii,
    )

    # Compute interaction terms
    gauss1 = gauss1_term(distances, params.gauss1_offset, params.gauss1_width)
    repulsion = repulsion_term(distances)
    hydrophobic = hydrophobic_term(
        distances,
        ligand_typing.is_hydrophobic,
        protein_typing.is_hydrophobic,
        params.hydro_good,
        params.hydro_bad,
    )
    hbond = hbond_term(
        distances,
        ligand_typing.is_hbond_donor,
        ligand_typing.is_hbond_acceptor,
        protein_typing.is_hbond_donor,
        protein_typing.is_hbond_acceptor,
        params.hbond_good,
    )

    # Sum all pair contributions and weight
    score = (
        params.w_gauss1 * np.sum(gauss1)
        + params.w_repulsion * np.sum(repulsion)
        + params.w_hydrophobic * np.sum(hydrophobic)
        + params.w_hbond * np.sum(hbond)
    )

    return float(score)


# =============================================================================
# Extensibility
# =============================================================================


class ScoringFunction(Protocol):
    """Protocol for docking scoring functions.

    This protocol defines the interface for scoring functions that can be
    used with the pose optimization functions. Any callable matching this
    signature can be used as a scoring function.

    Example:
        def my_scoring_fn(
            ligand_mol: Chem.Mol,
            protein_mol: Chem.Mol,
            ligand_conf_id: int = 0,
            protein_conf_id: int = 0,
        ) -> float:
            # Custom scoring logic
            return score
    """

    def __call__(
        self,
        ligand_mol: Chem.Mol,
        protein_mol: Chem.Mol,
        ligand_conf_id: int = 0,
        protein_conf_id: int = 0,
    ) -> float:
        """Compute docking score for a ligand-protein complex.

        Args:
            ligand_mol: Ligand RDKit Mol with 3D coordinates.
            protein_mol: Protein RDKit Mol with 3D coordinates.
            ligand_conf_id: Ligand conformer ID to use.
            protein_conf_id: Protein conformer ID to use.

        Returns:
            Docking score (lower is better for binding).
        """
        ...


def get_scoring_function(name: str = "vinardo") -> ScoringFunction:
    """Get a scoring function by name.

    Args:
        name: Scoring function name ("vinardo", etc.).

    Returns:
        Callable scoring function.

    Raises:
        ValueError: If scoring function not found.
    """
    functions: dict[str, ScoringFunction] = {
        "vinardo": vinardo_score,
    }
    if name not in functions:
        raise ValueError(
            f"Unknown scoring function: {name}. Available: {list(functions.keys())}"
        )
    return functions[name]
