"""Template docking primitives.

Dock a ligand by reusing a *known* pose of a shared substructure (a "core").
Given a core molecule that carries 3D coordinates and whose graph is a
substructure of the ligand, :func:`transfer_template_pose` overlays the ligand's
matching atoms onto the core (via a fast ``GetSubstructMatch`` -- no maximum
common substructure search), relaxes the rest with the core held, and returns
the prepared ligand plus the matched heavy-atom indices. Those indices feed the
existing constrained pose search (``constrained_atom_indices`` with ``n_starts``
forced to 1) so only a single local search is run -- the speed and
series-consistency win behind scaffold-indexed docking.

The core is supplied by the scaffold index (a stored Bemis-Murcko scaffold pose,
see :mod:`cmxflow.operators.dock.scaffold_index`), which is guaranteed to be a
substructure of any molecule sharing its scaffold key -- so substructure matching
is exact and cheap, avoiding the cost of an MCS search on the hot path.

The primitives are pure (no I/O, no global state) and deterministic.
"""

from __future__ import annotations

import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from cmxflow.operators.dock.pose import (
    OptimizationResult,
    PoseParams,
    optimize_pose_cached,
)
from cmxflow.operators.dock.score import AtomTyping, EmpiricalParams

logger = logging.getLogger(__name__)


def transfer_template_pose(
    ligand: Chem.Mol,
    core: Chem.Mol,
    *,
    mmff_max_iters: int = 200,
) -> tuple[Chem.Mol, tuple[int, ...]] | None:
    """Overlay a ligand onto a template core pose and relax with the core held.

    Rigidly aligns the ligand onto ``core`` by their matching atoms, snaps the
    matched atoms exactly onto the core coordinates, then runs a short MMFF94s
    minimization that holds the matched atoms fixed (so substituents relax without
    moving the trusted core). Falls back to UFF, then to no minimization, when
    force-field parameters are unavailable. The core is held *exactly* here;
    the give that lets it shift to relieve clashes is the flat-bottom restraint
    applied later in the constrained dock, not this geometry cleanup.

    Args:
        ligand: Ligand RDKit Mol with a 3D conformer.
        core: Core Mol with a 3D conformer whose graph is a substructure of the
            ligand (e.g. a stored scaffold pose or an MCS submol).
        mmff_max_iters: Maximum force-field minimization iterations.

    Returns:
        ``(prepared_ligand_with_Hs, matched_heavy_atom_indices)`` where the core
        is overlaid on the template and the rest is relaxed, or ``None`` if the
        core does not match the ligand.
    """
    ligand_heavy = Chem.RemoveAllHs(ligand)
    match = ligand_heavy.GetSubstructMatch(core)
    if not match:
        logger.debug("Template core did not match ligand; skipping transfer.")
        return None

    # Work on a Hs-added copy (MMFF needs Hs); AddHs keeps heavy-atom indices, so
    # `match` (heavy indices) stays valid and the result matches the with-H input
    # convention of the normal docking path.
    mol_h = Chem.AddHs(ligand_heavy, addCoords=True)

    # core atom i  <->  ligand heavy atom match[i].
    atom_map = [(int(match[i]), i) for i in range(core.GetNumAtoms())]

    # Rigid overlay of the ligand onto the core, then snap matched atoms exactly.
    rdMolAlign.AlignMol(mol_h, core, atomMap=atom_map)
    conf = mol_h.GetConformer()
    core_conf = core.GetConformer()
    for core_i, lig_i in enumerate(match):
        conf.SetAtomPosition(int(lig_i), core_conf.GetAtomPosition(core_i))

    _relax_with_core_held(mol_h, match, mmff_max_iters)
    return mol_h, tuple(sorted(int(i) for i in match))


def _relax_with_core_held(
    mol_h: Chem.Mol,
    held: tuple[int, ...],
    max_iters: int,
) -> None:
    """Minimize ``mol_h`` in place holding ``held`` atoms fixed.

    Tries MMFF94s, then UFF, then leaves the snapped coordinates unchanged.
    """
    if AllChem.MMFFHasAllMoleculeParams(mol_h):
        props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94s")
        ff = AllChem.MMFFGetMoleculeForceField(mol_h, props)
    elif AllChem.UFFHasAllMoleculeParams(mol_h):
        ff = AllChem.UFFGetMoleculeForceField(mol_h)
    else:
        logger.debug("No MMFF/UFF params for ligand; docking from snapped pose.")
        return
    if ff is None:
        return
    for idx in held:
        ff.AddFixedPoint(int(idx))
    ff.Initialize()
    ff.Minimize(maxIts=max_iters)


def template_dock(
    ligand: Chem.Mol,
    core: Chem.Mol,
    protein_coords: np.ndarray,
    protein_typing: AtomTyping,
    *,
    constraint_weight: float,
    constraint_tol: float = 0.5,
    score_params: EmpiricalParams | None = None,
    max_iterations: int = 100,
    basin_hops: int = 0,
    optimize_torsions: bool = True,
    translation_bounds: tuple[float, float] = (-10.0, 10.0),
    seed: int = 0,
    protein_tree=None,
) -> OptimizationResult | None:
    """Template / MCS docking: transfer a core pose, then constrained local search.

    Overlays the ligand onto ``core`` (:func:`transfer_template_pose`) and runs a
    single constrained pose optimization (``n_starts=1``) that tethers the core to
    its template position with a flat-bottom restraint (free within
    ``constraint_tol``). Returns ``None`` if the core does not match the ligand.

    Args:
        ligand: Ligand Mol with a 3D conformer.
        core: Posed core Mol (substructure of the ligand).
        protein_coords: Pre-computed protein atom coordinates.
        protein_typing: Pre-computed protein atom typing.
        constraint_weight: Flat-bottom restraint weight (kcal/mol/Å²).
        constraint_tol: Flat-bottom radius (Å); core moves freely within it.
        score_params: Scoring parameters (defaults if None).
        max_iterations: L-BFGS-B iterations per local minimize.
        basin_hops: Optional iterated-local-search hops (0 = single minimize).
        optimize_torsions: Optimize rotatable torsions (False = rigid).
        translation_bounds: Translation search bounds (Å).
        seed: RNG seed for basin-hopping.
        protein_tree: Optional precomputed KDTree for sparse scoring.

    Returns:
        OptimizationResult for the constrained dock, or ``None`` on no match.
    """
    transferred = transfer_template_pose(ligand, core)
    if transferred is None:
        return None
    prepared, constrained_idx = transferred
    params = PoseParams(
        n_starts=1,
        max_iterations=max_iterations,
        translation_bounds=translation_bounds,
        optimize_torsions=optimize_torsions,
        basin_hops=basin_hops,
        constrained_atom_indices=constrained_idx,
        constraint_weight=constraint_weight,
        constraint_tol=constraint_tol,
        seed=seed,
    )
    return optimize_pose_cached(
        prepared,
        protein_coords,
        protein_typing,
        params=params,
        score_params=score_params,
        site_center=None,
        protein_tree=protein_tree,
    )
