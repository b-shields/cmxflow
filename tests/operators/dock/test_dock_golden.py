"""Golden-master docking tests for ``MoleculeDockBlock``.

Docks one real ligand into a real receptor and pins the resulting score and pose,
so any change that alters the docking result is caught. Three configs cover the
distinct search paths: flexible docking, rigid docking, and iterated local search
(basin hopping). The search is fully deterministic for a fixed config (seeded
conformer embedding and Sobol grid, deterministic L-BFGS-B), reproducing the same
minimum bit-for-bit; BLAS is pinned to a single thread to avoid floating-point
reduction-order drift.
"""

from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from threadpoolctl import threadpool_limits

from cmxflow.operators.dock import MoleculeDockBlock
from cmxflow.sources.reader import read_molecules

DATA = Path(__file__).parent / "data" / "abl1"

# Each case: (config, golden docking_score, golden-pose fixture file). Configs are
# kept light except "flex", which is the production config. Golden values captured
# on CHEMBL355330 and verified bit-identical across runs.
GOLDEN_CASES = {
    "flex": (
        dict(
            n_starts=32,
            basin_hops=0,
            max_distance_geometry_samples=32,
            sobol_max_tries=2048,
            diversity_rmsd=1.0,
        ),
        -10.1956373547,
        "golden_pose_flex.npy",
    ),
    "rigid": (
        dict(
            rigid=True,
            n_starts=4,
            sobol_max_tries=512,
            max_distance_geometry_samples=32,
            diversity_rmsd=1.0,
            basin_hops=0,
        ),
        -0.3049939334,
        "golden_pose_rigid.npy",
    ),
    "ils": (
        dict(
            rigid=False,
            n_starts=2,
            max_distance_geometry_samples=4,
            sobol_max_tries=512,
            diversity_rmsd=1.0,
            basin_hops=3,
        ),
        -4.4360881952,
        "golden_pose_ils.npy",
    ),
}


def _dock(config: dict) -> Chem.Mol:
    """Dock the frozen abl1 active with ``config`` and return the docked mol."""
    block = MoleculeDockBlock(
        receptor=str(DATA / "receptor.pdb"),
        site_reference=str(DATA / "crystal_ligand.mol2"),
        **config,
    )
    mol = next(read_molecules(DATA / "active_ligand.sdf", wrap=False))
    with threadpool_limits(limits=1):
        out = block._forward(mol)
    assert out is not None
    return out


def _assert_golden(out: Chem.Mol, golden_score: float, pose_file: str) -> None:
    """Assert the docked result matches the pinned golden score and pose.

    Tolerances catch any behaviour change (which shifts scores by tenths of a
    kcal/mol and poses by tenths of an angstrom) while tolerating last-ULP
    BLAS/libm differences across CI machines, which the optimizer can amplify
    within a flat basin. The pose gets an extra order of slack over the score
    because atoms can drift further than the score within an equienergetic well.
    """
    assert float(out.GetProp("docking_score")) == pytest.approx(golden_score, abs=1e-3)
    golden_pose = np.load(DATA / pose_file)
    pose = np.array(Chem.RemoveAllHs(out).GetConformer().GetPositions())
    rmsd = float(np.sqrt(np.mean(np.sum((pose - golden_pose) ** 2, axis=1))))
    assert rmsd < 1e-2


@pytest.mark.parametrize("case", list(GOLDEN_CASES))
def test_dock_reaches_golden_minimum(case: str) -> None:
    """Each search path reaches its exact pinned score and pose."""
    config, golden_score, pose_file = GOLDEN_CASES[case]
    _assert_golden(_dock(config), golden_score, pose_file)


def test_defaults_reach_golden_minimum() -> None:
    """A default-constructed block reproduces the production (flex) golden."""
    _, golden_score, pose_file = GOLDEN_CASES["flex"]
    _assert_golden(_dock({}), golden_score, pose_file)


# Scaffold-constraint path. Docks the co-crystallized ligand with its fused
# bicyclic core pinned by a SMARTS constraint. The constrained core must stay on
# its input position while the same search *without* the constraint both relocates
# that core and reaches a lower score -- proving the constraint actively holds the
# scaffold rather than the search simply never leaving it.
CONSTRAINT_SMARTS = "c1ncc2cccnc2n1"  # fused bicyclic scaffold of the crystal ligand
# Light, exploratory search shared by the constrained and free runs.
CONSTRAINT_SEARCH = dict(
    n_starts=8, max_distance_geometry_samples=8, sobol_max_tries=512, basin_hops=0
)
CONSTRAINED_GOLDEN_SCORE = -12.2233718177
CONSTRAINED_GOLDEN_POSE = "golden_pose_constrained.npy"


def _dock_crystal(config: dict) -> Chem.Mol:
    """Dock the frozen abl1 crystal ligand with ``config`` and return the result."""
    block = MoleculeDockBlock(
        receptor=str(DATA / "receptor.pdb"),
        site_reference=str(DATA / "crystal_ligand.mol2"),
        **config,
    )
    mol = next(read_molecules(DATA / "crystal_ligand.mol2", wrap=False))
    with threadpool_limits(limits=1):
        out = block._forward(mol)
    assert out is not None
    return out


def _crystal_core() -> tuple[list[int], np.ndarray]:
    """Return (core heavy-atom indices, input positions) for the crystal ligand."""
    mol = next(read_molecules(DATA / "crystal_ligand.mol2", wrap=False))
    heavy = Chem.RemoveAllHs(mol)
    matches = heavy.GetSubstructMatches(Chem.MolFromSmarts(CONSTRAINT_SMARTS))
    idx = sorted({i for match in matches for i in match})
    return idx, np.array(heavy.GetConformer().GetPositions())[idx]


def _core_displacement(out: Chem.Mol) -> float:
    """Max distance any constrained core atom moved from its input position."""
    idx, in_core = _crystal_core()
    out_core = np.array(Chem.RemoveAllHs(out).GetConformer().GetPositions())[idx]
    return float(np.linalg.norm(out_core - in_core, axis=1).max())


def test_constraint_holds_core_at_golden_minimum() -> None:
    """A scaffold-constrained dock pins the core and reaches its golden pose.

    Also covers the ``n_starts -> 1`` forcing in ``_forward``: with a constraint
    matched, only the aligned input pose is refined regardless of configured starts.
    """
    out = _dock_crystal(
        dict(
            constraint_smarts=CONSTRAINT_SMARTS,
            constraint_weight=1000.0,
            **CONSTRAINT_SEARCH,
        )
    )
    _assert_golden(out, CONSTRAINED_GOLDEN_SCORE, CONSTRAINED_GOLDEN_POSE)
    assert _core_displacement(out) < 0.05  # core held on its input position
    assert out.GetIntProp("docking_n_starts_used") == 1  # constraint forces 1 start


def test_free_docking_moves_core_and_scores_lower() -> None:
    """Guard: the same search without the constraint leaves the input core and
    reaches a lower score, so the constrained test above is not vacuous."""
    free = _dock_crystal(dict(CONSTRAINT_SEARCH))
    assert _core_displacement(free) > 0.15  # free search relocates the core
    free_score = float(free.GetProp("docking_score"))
    assert free_score < CONSTRAINED_GOLDEN_SCORE - 0.1  # and finds a better pose
