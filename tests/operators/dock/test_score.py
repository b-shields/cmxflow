"""Unit tests for empirical score components."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from cmxflow.operators.dock.score import (
    AtomTyping,
    EmpiricalParams,
    NeighborList,
    ScoreComponents,
    build_protein_tree,
    empirical_score_and_grad_cached,
    empirical_score_and_grad_fast,
    empirical_score_cached,
    get_atom_typing,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_mol_3d(smiles: str, seed: int = 42) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    return mol


def _make_protein_typing(
    n_atoms: int,
    hydrophobic: bool = False,
    donor: bool = False,
    acceptor: bool = False,
) -> AtomTyping:
    return AtomTyping(
        radii=np.full(n_atoms, 1.7),
        is_hydrophobic=np.full(n_atoms, hydrophobic),
        is_hbond_donor=np.full(n_atoms, donor),
        is_hbond_acceptor=np.full(n_atoms, acceptor),
    )


def _benzene_system():
    """Heavy-atom benzene ligand + 3 hydrophobic protein atoms ~3 Å away."""
    ligand = Chem.RemoveAllHs(_make_mol_3d("c1ccccc1"))
    protein_coords = np.array([[4.0, 0.0, 0.0], [4.0, 1.4, 0.0], [4.0, -1.4, 0.0]])
    protein_typing = _make_protein_typing(3, hydrophobic=True)
    return ligand, protein_coords, protein_typing


# =============================================================================
# TestEmpiricalScoreCachedComponents
# =============================================================================


class TestEmpiricalScoreCachedComponents:
    """Tests for empirical_score_cached which always returns ScoreComponents."""

    def test_returns_score_components(self) -> None:
        """empirical_score_cached always returns ScoreComponents."""
        ligand, pc, pt = _benzene_system()
        result = empirical_score_cached(ligand, pc, pt)
        assert isinstance(result, ScoreComponents)

    def test_total_is_float(self) -> None:
        """comps.total is a float."""
        ligand, pc, pt = _benzene_system()
        comps = empirical_score_cached(ligand, pc, pt)
        assert isinstance(comps.total, float)

    def test_components_sum_to_total(self) -> None:
        """comps.total is the component sum divided by the torsion divisor.

        Components are raw (pre-divisor) weighted terms; total applies the
        torsion divisor once to their sum.
        """
        ligand, pc, pt = _benzene_system()
        comps = empirical_score_cached(ligand, pc, pt)
        divisor = 1.0 + comps.w_rot * comps.n_rot
        reconstructed = (
            comps.gauss1 + comps.repulsion + comps.hydrophobic + comps.hbond
        ) / divisor
        assert reconstructed == pytest.approx(comps.total)

    def test_hydrophobic_raw_zero_when_no_hydrophobic_pairs(self) -> None:
        """hydrophobic_raw == 0 when protein has no hydrophobic atoms."""
        ligand, pc, _ = _benzene_system()
        comps = empirical_score_cached(
            ligand, pc, _make_protein_typing(3, hydrophobic=False)
        )
        assert comps.hydrophobic_raw == 0.0

    def test_hbond_raw_zero_when_no_donor_acceptor_pairs(self) -> None:
        """hbond_raw == 0 when neither molecule has donors or acceptors."""
        ligand, pc, _ = _benzene_system()
        comps = empirical_score_cached(ligand, pc, _make_protein_typing(3))
        assert comps.hbond_raw == 0.0

    def test_hydrophobic_raw_nonzero_when_close_hydrophobic_pairs(self) -> None:
        """Sanity: hydrophobic_raw > 0 when atoms are within the good cutoff."""
        ligand, pc, pt = _benzene_system()
        comps = empirical_score_cached(ligand, pc, pt)
        assert comps.hydrophobic_raw > 0.0

    def test_weights_in_components_match_params(self) -> None:
        """Weights in ScoreComponents must match the EmpiricalParams passed in."""
        ligand, pc, pt = _benzene_system()
        params = EmpiricalParams(
            w_gauss1=-0.050, w_repulsion=0.900, w_hydrophobic=-0.040, w_hbond=-0.700
        )
        comps = empirical_score_cached(ligand, pc, pt, params=params)
        assert comps.w_gauss1 == pytest.approx(-0.050)
        assert comps.w_repulsion == pytest.approx(0.900)
        assert comps.w_hydrophobic == pytest.approx(-0.040)
        assert comps.w_hbond == pytest.approx(-0.700)


# =============================================================================
# TestSparseCutoffPath
# =============================================================================


def _protein_cloud(ligand: Chem.Mol, n: int = 200, seed: int = 7) -> tuple:
    """Random protein cloud around the ligand with mixed atom typing.

    Returns (protein_coords, protein_typing) spanning near-contact to far
    (>cutoff) atoms so both the close-range terms and the cutoff are exercised.
    """
    rng = np.random.default_rng(seed)
    center = np.array(ligand.GetConformer().GetPositions()).mean(axis=0)
    coords = center + rng.uniform(-12.0, 12.0, size=(n, 3))
    typing = AtomTyping(
        radii=rng.choice([1.6, 1.7, 2.0], size=n),
        is_hydrophobic=rng.random(n) < 0.4,
        is_hbond_donor=rng.random(n) < 0.2,
        is_hbond_acceptor=rng.random(n) < 0.2,
    )
    return coords, typing


class TestSparseCutoffPath:
    """The KD-tree cutoff path must match the dense path to numerical precision."""

    def test_score_matches_dense(self) -> None:
        ligand = Chem.RemoveAllHs(_make_mol_3d("c1ccccc1CCO"))
        pc, pt = _protein_cloud(ligand)
        tree = build_protein_tree(pc)
        dense = empirical_score_cached(ligand, pc, pt).total
        sparse = empirical_score_cached(ligand, pc, pt, protein_tree=tree).total
        assert sparse == pytest.approx(dense, abs=1e-8)

    def test_grad_matches_dense(self) -> None:
        ligand = Chem.RemoveAllHs(_make_mol_3d("c1ccccc1CCO"))
        pc, pt = _protein_cloud(ligand)
        tree = build_protein_tree(pc)
        s_dense, g_dense = empirical_score_and_grad_cached(ligand, pc, pt)
        s_sparse, g_sparse = empirical_score_and_grad_cached(
            ligand, pc, pt, protein_tree=tree
        )
        assert s_sparse == pytest.approx(s_dense, abs=1e-8)
        np.testing.assert_allclose(g_sparse, g_dense, atol=1e-8)

    def test_cached_ligand_typing_matches(self) -> None:
        """Passing pre-computed ligand_typing must not change the result."""
        from cmxflow.operators.dock.score import get_atom_typing

        ligand = Chem.RemoveAllHs(_make_mol_3d("c1ccccc1CCO"))
        pc, pt = _protein_cloud(ligand)
        tree = build_protein_tree(pc)
        lt = get_atom_typing(ligand)
        base = empirical_score_cached(ligand, pc, pt, protein_tree=tree).total
        cached = empirical_score_cached(
            ligand, pc, pt, protein_tree=tree, ligand_typing=lt
        ).total
        assert cached == pytest.approx(base, abs=1e-12)

    def test_no_neighbors_in_range(self) -> None:
        """Ligand far from all protein atoms: zero interaction, zero gradient."""
        ligand = Chem.RemoveAllHs(_make_mol_3d("c1ccccc1CCO"))
        pc = np.array(ligand.GetConformer().GetPositions()).mean(axis=0) + np.array(
            [[100.0, 0.0, 0.0], [100.0, 2.0, 0.0]]
        )
        pt = _make_protein_typing(2, hydrophobic=True, donor=True, acceptor=True)
        tree = build_protein_tree(pc)
        score, grad = empirical_score_and_grad_cached(ligand, pc, pt, protein_tree=tree)
        assert score == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_allclose(grad, 0.0, atol=1e-12)


# =============================================================================
# TestNeighborListFastPath
# =============================================================================


def _moved_mol(ligand: Chem.Mol, shift: np.ndarray) -> tuple[Chem.Mol, np.ndarray]:
    """Copy ``ligand`` translated by ``shift``; return (mol, coords)."""
    coords = np.array(ligand.GetConformer().GetPositions()) + shift
    mol = Chem.Mol(ligand)
    conf = mol.GetConformer()
    for i, xyz in enumerate(coords):
        conf.SetAtomPosition(i, xyz.tolist())
    return mol, coords


class TestNeighborListFastPath:
    """The reused-Verlet fast path must equal a fresh per-call KD-tree query.

    Same scored pair set (kernel gates at ``cutoff``), so results match the
    cached path to summation order regardless of skin rebuilds. A rotatable
    ligand is used so the torsion divisor is non-trivial (!= 1).
    """

    def _setup(self):
        ligand = Chem.RemoveAllHs(_make_mol_3d("c1ccccc1CCO"))  # 1 rotatable bond
        pc, pt = _protein_cloud(ligand)
        lt = get_atom_typing(ligand)
        n_rot = rdMolDescriptors.CalcNumRotatableBonds(ligand, strict=False)
        params = EmpiricalParams()
        divisor = 1.0 + params.w_rot * n_rot
        assert divisor != 1.0  # guards that the divisor is actually exercised
        return ligand, pc, pt, lt, params, divisor

    def _cached(self, mol, pc, pt, lt):
        tree = build_protein_tree(pc)
        return empirical_score_and_grad_cached(
            mol, pc, pt, protein_tree=tree, ligand_typing=lt
        )

    def test_matches_cached_first_eval(self) -> None:
        """Fresh NeighborList scores identically to the per-call sparse path."""
        ligand, pc, pt, lt, params, divisor = self._setup()
        s_ref, g_ref = self._cached(ligand, pc, pt, lt)
        coords = np.array(ligand.GetConformer().GetPositions())
        nbr = NeighborList(pc, pt, lt)
        s_fast, g_fast = empirical_score_and_grad_fast(coords, nbr, params, divisor)
        assert s_fast == pytest.approx(s_ref, abs=1e-9)
        np.testing.assert_allclose(g_fast, g_ref, atol=1e-9)

    def test_reuse_within_skin_no_rebuild(self) -> None:
        """A sub-skin move reuses the list yet still matches a fresh query."""
        ligand, pc, pt, lt, params, divisor = self._setup()
        nbr = NeighborList(pc, pt, lt, skin=2.0)
        coords0 = np.array(ligand.GetConformer().GetPositions())
        empirical_score_and_grad_fast(coords0, nbr, params, divisor)
        ref_build = nbr._ref_coords  # noqa: SLF001 - asserting rebuild policy

        mol1, coords1 = _moved_mol(ligand, np.array([0.5, 0.0, 0.0]))  # < skin
        s_fast, g_fast = empirical_score_and_grad_fast(coords1, nbr, params, divisor)
        assert nbr._ref_coords is ref_build  # no rebuild  # noqa: SLF001
        s_ref, g_ref = self._cached(mol1, pc, pt, lt)
        assert s_fast == pytest.approx(s_ref, abs=1e-9)
        np.testing.assert_allclose(g_fast, g_ref, atol=1e-9)

    def test_rebuild_past_skin_matches_fresh(self) -> None:
        """A move beyond the skin rebuilds and still matches a fresh query."""
        ligand, pc, pt, lt, params, divisor = self._setup()
        nbr = NeighborList(pc, pt, lt, skin=2.0)
        coords0 = np.array(ligand.GetConformer().GetPositions())
        empirical_score_and_grad_fast(coords0, nbr, params, divisor)
        ref_build = nbr._ref_coords  # noqa: SLF001

        mol1, coords1 = _moved_mol(ligand, np.array([3.0, 0.0, 0.0]))  # > skin
        s_fast, g_fast = empirical_score_and_grad_fast(coords1, nbr, params, divisor)
        assert nbr._ref_coords is not ref_build  # rebuilt  # noqa: SLF001
        s_ref, g_ref = self._cached(mol1, pc, pt, lt)
        assert s_fast == pytest.approx(s_ref, abs=1e-9)
        np.testing.assert_allclose(g_fast, g_ref, atol=1e-9)

    def test_empty_list_far_ligand(self) -> None:
        """Ligand beyond cutoff+skin of every protein atom: zero score/grad."""
        ligand = Chem.RemoveAllHs(_make_mol_3d("c1ccccc1CCO"))
        lt = get_atom_typing(ligand)
        pc = np.array(ligand.GetConformer().GetPositions()).mean(axis=0) + np.array(
            [[100.0, 0.0, 0.0], [100.0, 2.0, 0.0]]
        )
        pt = _make_protein_typing(2, hydrophobic=True, donor=True, acceptor=True)
        coords = np.array(ligand.GetConformer().GetPositions())
        nbr = NeighborList(pc, pt, lt)
        score, grad = empirical_score_and_grad_fast(coords, nbr, EmpiricalParams(), 1.0)
        assert score == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_allclose(grad, 0.0, atol=1e-12)
