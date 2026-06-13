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
    """Assert the docked result matches the pinned golden score and pose."""
    assert float(out.GetProp("docking_score")) == pytest.approx(golden_score, abs=1e-6)
    golden_pose = np.load(DATA / pose_file)
    pose = np.array(Chem.RemoveAllHs(out).GetConformer().GetPositions())
    rmsd = float(np.sqrt(np.mean(np.sum((pose - golden_pose) ** 2, axis=1))))
    assert rmsd < 1e-4


@pytest.mark.parametrize("case", list(GOLDEN_CASES))
def test_dock_reaches_golden_minimum(case: str) -> None:
    """Each search path reaches its exact pinned score and pose."""
    config, golden_score, pose_file = GOLDEN_CASES[case]
    _assert_golden(_dock(config), golden_score, pose_file)


def test_defaults_reach_golden_minimum() -> None:
    """A default-constructed block reproduces the production (flex) golden."""
    _, golden_score, pose_file = GOLDEN_CASES["flex"]
    _assert_golden(_dock({}), golden_score, pose_file)
