"""Electrostatic complementarity scoring for molecular docking.

This module computes the electrostatic complementarity (EC) between a
ligand and protein binding site. EC is defined as the Pearson correlation
between ligand and negated protein electrostatic potentials on the ligand
solvent-accessible surface (SAS).

EC values range from -1 to +1:
    +1: Perfect charge complementarity (favorable)
     0: No correlation
    -1: Anti-complementary charges (unfavorable)

Reference:
    Bauer & Mackey (2019). Electrostatic Complementarity as a Fast and
    Effective Tool to Optimize Binding and Selectivity of Protein-Ligand
    Complexes. J. Med. Chem. 62(6): 3036-3050.
"""

import logging

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

# Coulomb constant in kcal*A/(mol*e^2)
COULOMB_K = 332.06


# =============================================================================
# Surface Point Generation
# =============================================================================


def fibonacci_sphere(n_points: int) -> NDArray[np.floating]:
    """Generate approximately uniform points on a unit sphere.

    Uses the Fibonacci spiral method for quasi-uniform distribution.

    Args:
        n_points: Number of points to generate.

    Returns:
        Array of shape (n_points, 3) with unit-sphere coordinates.
    """
    indices = np.arange(n_points, dtype=np.float64)
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0

    # Latitude: evenly spaced in [-1, 1]
    z = 1.0 - 2.0 * indices / (n_points - 1) if n_points > 1 else np.array([0.0])
    r = np.sqrt(1.0 - z * z)

    # Longitude: golden angle increments
    theta = 2.0 * np.pi * indices / golden_ratio

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return np.column_stack([x, y, z])


def generate_sas_points(
    coords: NDArray[np.floating],
    radii: NDArray[np.floating],
    probe_radius: float = 1.4,
    n_sphere_points: int = 50,
) -> NDArray[np.floating]:
    """Generate solvent-accessible surface (SAS) points around atoms.

    For each atom, sphere points are placed at radius + probe_radius.
    Points buried inside other atoms' SAS spheres are removed.

    Args:
        coords: Atom coordinates (n_atoms, 3).
        radii: Van der Waals radii (n_atoms,).
        probe_radius: Solvent probe radius in Angstroms.
        n_sphere_points: Points per atom sphere.

    Returns:
        Non-buried surface points (n_surface, 3).
    """
    n_atoms = len(coords)
    if n_atoms == 0:
        return np.empty((0, 3), dtype=np.float64)

    unit_sphere = fibonacci_sphere(n_sphere_points)
    sas_radii = radii + probe_radius

    # Generate all candidate points: scale and translate unit sphere per atom
    # candidates shape: (n_atoms * n_sphere_points, 3)
    candidates = (
        unit_sphere[np.newaxis, :, :] * sas_radii[:, np.newaxis, np.newaxis]
        + coords[:, np.newaxis, :]
    ).reshape(-1, 3)

    # Parent atom index for each candidate
    parent_idx = np.repeat(np.arange(n_atoms), n_sphere_points)

    # Distance from each candidate to each atom center
    dists = cdist(candidates, coords)  # (n_candidates, n_atoms)

    # Mask parent atom distances (set to inf so they don't cause self-burial)
    dists[np.arange(len(candidates)), parent_idx] = np.inf

    # A candidate is buried if any non-parent atom's SAS sphere contains it
    buried = np.any(dists < sas_radii[np.newaxis, :], axis=1)

    return candidates[~buried]


# =============================================================================
# Charge Computation
# =============================================================================


def compute_gasteiger_charges(mol: Chem.Mol) -> NDArray[np.floating]:
    """Compute Gasteiger partial charges for a molecule.

    Args:
        mol: RDKit Mol object (should have explicit H for accuracy).

    Returns:
        Array of partial charges (n_atoms,).
    """
    AllChem.ComputeGasteigerCharges(mol)
    charges = np.array(
        [atom.GetDoubleProp("_GasteigerCharge") for atom in mol.GetAtoms()],
        dtype=np.float64,
    )
    return np.nan_to_num(charges, nan=0.0)


# =============================================================================
# Electrostatic Potential
# =============================================================================


def compute_esp_at_points(
    points: NDArray[np.floating],
    atom_coords: NDArray[np.floating],
    charges: NDArray[np.floating],
    cutoff: float = 10.0,
) -> NDArray[np.floating]:
    """Compute electrostatic potential at query points from atomic charges.

    Uses Coulomb's law with a distance cutoff for efficiency.
    Distances below 0.1 A are clamped to avoid singularities.

    Args:
        points: Query point coordinates (n_points, 3).
        atom_coords: Atom coordinates (n_atoms, 3).
        charges: Partial charges (n_atoms,).
        cutoff: Distance cutoff in Angstroms.

    Returns:
        Electrostatic potential at each query point (n_points,).
    """
    n_points = len(points)
    if n_points == 0 or len(atom_coords) == 0:
        return np.zeros(n_points, dtype=np.float64)

    point_tree = cKDTree(points)
    atom_tree = cKDTree(atom_coords)

    sparse_dist = point_tree.sparse_distance_matrix(
        atom_tree, cutoff, output_type="coo_matrix"
    )

    # Clamp minimum distance to avoid singularity
    sparse_dist.data = np.maximum(sparse_dist.data, 0.1)

    # Coulomb contributions: k * q / r
    contributions = COULOMB_K * charges[sparse_dist.col] / sparse_dist.data

    # Sum contributions per query point
    potentials = np.zeros(n_points, dtype=np.float64)
    np.add.at(potentials, sparse_dist.row, contributions)

    return potentials


# =============================================================================
# Electrostatic Complementarity
# =============================================================================


def electrostatic_complementarity(
    ligand_mol: Chem.Mol,
    protein_coords: NDArray[np.floating],
    protein_charges: NDArray[np.floating],
    probe_radius: float = 1.4,
    n_sphere_points: int = 50,
    esp_cutoff: float = 10.0,
) -> float:
    """Compute electrostatic complementarity between ligand and protein.

    EC is the Pearson correlation between ligand ESP and negated protein
    ESP on the ligand solvent-accessible surface. Higher values indicate
    better charge complementarity.

    Args:
        ligand_mol: Ligand RDKit Mol with 3D conformer.
        protein_coords: Protein atom coordinates (n_atoms, 3).
        protein_charges: Protein Gasteiger charges (n_atoms,).
        probe_radius: Solvent probe radius in Angstroms.
        n_sphere_points: Surface points per atom sphere.
        esp_cutoff: ESP distance cutoff in Angstroms.

    Returns:
        EC value in [-1, 1], or 0.0 for degenerate cases.
    """
    pt = Chem.GetPeriodicTable()

    # Ensure explicit H for accurate dipole representation
    if not any(a.GetAtomicNum() == 1 for a in ligand_mol.GetAtoms()):
        ligand_mol = Chem.AddHs(ligand_mol, addCoords=True)

    if ligand_mol.GetNumConformers() == 0:
        return 0.0

    # Ligand coordinates and VdW radii
    lig_coords = np.array(ligand_mol.GetConformer().GetPositions())
    lig_radii = np.array(
        [pt.GetRvdw(a.GetAtomicNum()) for a in ligand_mol.GetAtoms()],
        dtype=np.float64,
    )

    # Generate SAS points
    sas_points = generate_sas_points(
        lig_coords, lig_radii, probe_radius, n_sphere_points
    )

    if len(sas_points) < 3:
        return 0.0

    # Compute Gasteiger charges for ligand
    lig_charges = compute_gasteiger_charges(ligand_mol)

    # Compute ESP from ligand and protein at surface points
    esp_ligand = compute_esp_at_points(sas_points, lig_coords, lig_charges, esp_cutoff)
    esp_protein = compute_esp_at_points(
        sas_points, protein_coords, protein_charges, esp_cutoff
    )

    # Check for degenerate (constant) ESP
    if np.std(esp_ligand) < 1e-10 or np.std(esp_protein) < 1e-10:
        return 0.0

    # EC = correlation of ligand ESP with negated protein ESP
    corr_matrix = np.corrcoef(esp_ligand, -esp_protein)
    ec = corr_matrix[0, 1]

    if np.isnan(ec):
        return 0.0

    return float(ec)
