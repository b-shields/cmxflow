"""Golden-master docking test for ``MoleculeDockBlock``.

Docks one real ligand into a real receptor and pins the resulting score and pose,
so any change that alters the docking result is caught. The search is fully
deterministic for a fixed config (seeded conformer embedding and Sobol grid,
deterministic L-BFGS-B), reproducing the same minimum bit-for-bit; BLAS is pinned
to a single thread to avoid floating-point reduction-order drift.
"""

from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from threadpoolctl import threadpool_limits

from cmxflow.operators.dock import MoleculeDockBlock
from cmxflow.sources.reader import read_molecules

DATA = Path(__file__).parent / "data" / "abl1"

# Production docking config.
DG_CONFIG = dict(
    n_starts=32,
    basin_hops=0,
    init_mode="dg",
    max_distance_geometry_samples=32,
    sobol_max_tries=2048,
    max_score_per_heavy_atom=5.0,
    diversity_rmsd=1.0,
)

# Captured once with DG_CONFIG on CHEMBL355330 (bit-identical across runs).
GOLDEN_SCORE = -10.1956373547


def _dock(**overrides: object) -> Chem.Mol:
    """Dock the frozen abl1 active and return the docked molecule."""
    block = MoleculeDockBlock(
        receptor=str(DATA / "receptor.pdb"),
        site_reference=str(DATA / "crystal_ligand.mol2"),
        **{**DG_CONFIG, **overrides},
    )
    mol = next(read_molecules(DATA / "active_ligand.sdf", wrap=False))
    with threadpool_limits(limits=1):
        out = block._forward(mol)
    assert out is not None
    return out


def _assert_golden(out: Chem.Mol) -> None:
    """Assert the docked result is the golden score and pose."""
    score = float(out.GetProp("docking_score"))
    empirical = float(out.GetProp("docking_empirical"))
    assert score == pytest.approx(GOLDEN_SCORE, abs=1e-6)
    # No strain/EC folded into docking_score in this config, so they match.
    assert empirical == pytest.approx(GOLDEN_SCORE, abs=1e-6)

    golden_pose = np.load(DATA / "golden_pose_dg.npy")
    pose = np.array(Chem.RemoveAllHs(out).GetConformer().GetPositions())
    rmsd = float(np.sqrt(np.mean(np.sum((pose - golden_pose) ** 2, axis=1))))
    assert rmsd < 1e-4


def test_dg_dock_reaches_golden_minimum() -> None:
    """The DG search reaches the exact pinned minimum for the explicit config."""
    _assert_golden(_dock())
