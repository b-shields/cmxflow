"""Golden-master docking tests for ``MoleculeDockBlock``.

Docks one real ligand into a real receptor and pins the resulting score and pose,
so any change that alters the docking result is caught. Three configs cover the
distinct search paths: flexible docking, rigid docking, and iterated local search
(basin hopping). The search is fully deterministic for a fixed config (seeded
conformer embedding and Sobol grid, deterministic L-BFGS-B), reproducing the same
minimum bit-for-bit; BLAS is pinned to a single thread to avoid floating-point
reduction-order drift.
"""

import sys
from pathlib import Path
from typing import Any

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


# Goldens for stochastic search paths (flex, ils) are pinned on 3.13. On
# earlier versions the Sobol/conformer trajectories diverge enough to land in
# a different basin, so we skip rather than re-pin per version.
_needs_313 = pytest.mark.skipif(
    sys.version_info < (3, 13),
    reason=(
        "golden scores pinned on Python 3.13; trajectory diverges on earlier versions"
    ),
)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param("flex", marks=_needs_313),
        "rigid",
        pytest.param("ils", marks=_needs_313),
    ],
)
def test_dock_reaches_golden_minimum(case: str) -> None:
    """Each search path reaches its exact pinned score and pose."""
    config, golden_score, pose_file = GOLDEN_CASES[case]
    _assert_golden(_dock(config), golden_score, pose_file)


@_needs_313
def test_defaults_reach_golden_minimum() -> None:
    """A default-constructed block reproduces the production (flex) golden."""
    _, golden_score, pose_file = GOLDEN_CASES["flex"]
    _assert_golden(_dock({}), golden_score, pose_file)


# Scaffold-indexed (template) docking. Docking the active is a cache miss (full
# dock, caches its scaffold pose); a congener sharing that scaffold then hits the
# cache and is docked with a single constrained local search. Tests run in a tmp
# cwd so the conventional ./.cmxflow/scaffold_index.db is created/discovered there.
INDEX_SEARCH: dict[str, Any] = dict(
    n_starts=8, max_distance_geometry_samples=8, sobol_max_tries=512, basin_hops=0
)
INDEXED_GOLDEN_SCORE = -8.5587969796
INDEXED_GOLDEN_POSE = "golden_pose_indexed.npy"


def _index_block(**extra: Any) -> MoleculeDockBlock:
    return MoleculeDockBlock(
        receptor=str(DATA / "receptor.pdb"),
        site_reference=str(DATA / "crystal_ligand.mol2"),
        **INDEX_SEARCH,
        **extra,
    )


def test_indexed_dock_miss_then_hit(monkeypatch, tmp_path) -> None:
    """Miss caches the scaffold; a congener hits it and reaches its golden pose."""
    monkeypatch.chdir(tmp_path)
    block = _index_block(index_poses=True)
    active = next(read_molecules(DATA / "active_ligand.sdf", wrap=False))
    congener = next(read_molecules(DATA / "congener_ligand.sdf", wrap=False))
    with threadpool_limits(limits=1):
        miss = block._forward(active)
        hit = block._forward(congener)
        hit_again = block._forward(
            next(read_molecules(DATA / "congener_ligand.sdf", wrap=False))
        )

    assert miss is not None and hit is not None and hit_again is not None
    assert (tmp_path / ".cmxflow" / "scaffold_index.db").exists()
    assert miss.GetBoolProp("docking_indexed") is False  # active scaffold is novel
    assert hit.GetBoolProp("docking_indexed") is True  # congener shares it -> hit
    assert hit.GetIntProp("docking_n_starts_used") == 1  # single constrained search
    _assert_golden(hit, INDEXED_GOLDEN_SCORE, INDEXED_GOLDEN_POSE)
    # Deterministic on the warm cache.
    assert hit_again.GetProp("docking_score") == hit.GetProp("docking_score")


def test_reference_scaffold_is_seeded(monkeypatch, tmp_path) -> None:
    """The site_reference scaffold is seeded, so a molecule sharing it hits first."""
    monkeypatch.chdir(tmp_path)
    block = _index_block(index_poses=True)
    crystal = next(read_molecules(DATA / "crystal_ligand.mol2", wrap=False))
    with threadpool_limits(limits=1):
        out = block._forward(crystal)
    # No prior dock, but the reference seed means the crystal scaffold is a hit.
    assert out is not None
    assert out.GetBoolProp("docking_indexed") is True
    assert out.GetIntProp("docking_n_starts_used") == 1


def test_namespace_separates_targets() -> None:
    """Cache keys are namespaced by receptor/reference so one shared DB can serve
    multiple targets without pose collisions, but identical inputs share a key."""
    base = _index_block(index_poses=True)
    same = _index_block(index_poses=True)
    other_receptor = MoleculeDockBlock(
        receptor=str(DATA / "crystal_ligand.mol2"),  # different receptor path
        site_reference=str(DATA / "crystal_ligand.mol2"),
        **INDEX_SEARCH,
        index_poses=True,
    )
    assert base._index_namespace() == same._index_namespace()
    assert base._index_namespace() != other_receptor._index_namespace()
