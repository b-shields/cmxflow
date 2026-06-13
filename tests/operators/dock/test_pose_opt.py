"""Verification tests for pose optimization improvements.

Covers:
- Atom gradient finite-difference check (empirical_score_and_grad_cached)
- DOF gradient finite-difference check at r=0 and ||r||=π/2
- Constraint gradient finite-difference check
- Constraint behaviour (tight constraint holds atoms in place)
- SMARTS constraint interface (index resolution, H rejection, no-match passthrough)
"""

from unittest.mock import patch

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.transform import Rotation

from cmxflow.operators.dock.pose import (
    PoseParams,
    _apply_constraint_penalty,
    _build_intramolecular_pairs,
    _compute_dof_gradient,
    _constraint_penalty_value,
    _get_downstream_atoms,
    _rigid_fragments,
    _SingleDOFStep,
    apply_rigid_transform,
    apply_torsion_changes,
    get_rotatable_bonds,
    optimize_dg_restarts,
    optimize_pose_cached,
)
from cmxflow.operators.dock.score import (
    AtomTyping,
    EmpiricalParams,
    empirical_score_and_grad_cached,
    get_atom_typing,
    intramolecular_score_and_grad,
)

# =============================================================================
# Helpers
# =============================================================================

EPS = 1e-4  # finite-difference step
ATOL = 1e-4  # tolerance for gradient agreement


def _make_ligand(smiles: str = "c1ccccc1CCO", seed: int = 42) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    return mol


def _make_system(smiles: str = "c1ccccc1CCO"):
    """Return (ligand_heavy, protein_coords, protein_typing, params)."""
    ligand_heavy = Chem.RemoveAllHs(_make_ligand(smiles))
    protein_coords = np.array(ligand_heavy.GetConformer().GetPositions()) + np.array(
        [3.5, 0.0, 0.0]
    )
    protein_typing = get_atom_typing(ligand_heavy)  # same typing for simplicity
    params = EmpiricalParams()
    return ligand_heavy, protein_coords, protein_typing, params


# =============================================================================
# Step 1 verification: atom gradient finite-difference check
# =============================================================================


class TestAtomGradient:
    """empirical_score_and_grad_cached atom_grad vs finite differences."""

    def test_atom_grad_agrees_with_fd(self) -> None:
        ligand_heavy, protein_coords, protein_typing, params = _make_system()
        score0, atom_grad = empirical_score_and_grad_cached(
            ligand_heavy, protein_coords, protein_typing, params=params
        )

        conf = ligand_heavy.GetConformer(0)
        coords = np.array(conf.GetPositions())
        n_atoms = coords.shape[0]

        for i in range(n_atoms):
            for d in range(3):
                # +eps
                coords_p = coords.copy()
                coords_p[i, d] += EPS
                for j, pos in enumerate(coords_p):
                    conf.SetAtomPosition(j, tuple(pos))
                sp, _ = empirical_score_and_grad_cached(
                    ligand_heavy, protein_coords, protein_typing, params=params
                )

                # -eps
                coords_m = coords.copy()
                coords_m[i, d] -= EPS
                for j, pos in enumerate(coords_m):
                    conf.SetAtomPosition(j, tuple(pos))
                sm, _ = empirical_score_and_grad_cached(
                    ligand_heavy, protein_coords, protein_typing, params=params
                )

                fd = (sp - sm) / (2 * EPS)
                assert atom_grad[i, d] == pytest.approx(
                    fd, abs=ATOL
                ), f"atom {i} dim {d}: analytical={atom_grad[i,d]:.6f} fd={fd:.6f}"

                # restore
                for j, pos in enumerate(coords):
                    conf.SetAtomPosition(j, tuple(pos))


# =============================================================================
# Step 3 verification: DOF gradient finite-difference check
# =============================================================================


def _make_pose_system():
    """Return pose system tuple for DOF gradient tests."""
    ligand_mol = _make_ligand("c1ccccc1CCO")
    ligand_heavy = Chem.RemoveAllHs(ligand_mol)
    p0_heavy = np.array(ligand_heavy.GetConformer(0).GetPositions())
    centroid = np.mean(p0_heavy, axis=0)
    dihedrals = get_rotatable_bonds(ligand_heavy)
    subtrees = [_get_downstream_atoms(ligand_heavy, j, k) for (_, j, k, _) in dihedrals]
    protein_coords = p0_heavy + np.array([3.5, 0.0, 0.0])
    protein_typing = get_atom_typing(ligand_heavy)
    score_params = EmpiricalParams()
    return (
        ligand_mol,
        ligand_heavy,
        p0_heavy,
        centroid,
        dihedrals,
        subtrees,
        protein_coords,
        protein_typing,
        score_params,
    )


def _eval_objective(
    x,
    ligand_heavy,
    p0_heavy,
    centroid,
    dihedrals,
    subtrees,
    protein_coords,
    protein_typing,
    score_params,
    ligand_conf_id=0,
):
    """Evaluate objective (score only) at pose vector x."""
    n_torsions = len(dihedrals)
    T = x[:3]
    rot = Rotation.from_rotvec(x[3:6])
    mol = apply_rigid_transform(ligand_heavy, T, rot, ligand_conf_id, center=centroid)
    if dihedrals and n_torsions > 0:
        initial_torsions = np.array(
            [
                float(
                    Chem.rdMolTransforms.GetDihedralDeg(
                        ligand_heavy.GetConformer(0), *d
                    )
                )
                for d in dihedrals
            ]
        )
        new_torsions = initial_torsions + x[6:]
        mol = apply_torsion_changes(
            mol, dict(zip(dihedrals, new_torsions)), ligand_conf_id
        )
    score, atom_grad = empirical_score_and_grad_cached(
        mol, protein_coords, protein_typing, ligand_conf_id, score_params
    )
    return score, atom_grad, np.array(mol.GetConformer(0).GetPositions())


class TestDofGradient:
    """Full DOF gradient (T, r, θ) vs finite differences."""

    def _check_dof_grad(
        self,
        x0,
        ligand_heavy,
        p0_heavy,
        centroid,
        dihedrals,
        subtrees,
        protein_coords,
        protein_typing,
        score_params,
    ):
        score0, atom_grad0, pos0 = _eval_objective(
            x0,
            ligand_heavy,
            p0_heavy,
            centroid,
            dihedrals,
            subtrees,
            protein_coords,
            protein_typing,
            score_params,
        )
        analytical = _compute_dof_gradient(
            x0, pos0, atom_grad0, centroid, dihedrals, subtrees
        )

        dim = len(x0)
        fd_grad = np.zeros(dim)
        for k in range(dim):
            xp = x0.copy()
            xp[k] += EPS
            sp, _, _ = _eval_objective(
                xp,
                ligand_heavy,
                p0_heavy,
                centroid,
                dihedrals,
                subtrees,
                protein_coords,
                protein_typing,
                score_params,
            )
            xm = x0.copy()
            xm[k] -= EPS
            sm, _, _ = _eval_objective(
                xm,
                ligand_heavy,
                p0_heavy,
                centroid,
                dihedrals,
                subtrees,
                protein_coords,
                protein_typing,
                score_params,
            )
            fd_grad[k] = (sp - sm) / (2 * EPS)

        np.testing.assert_allclose(
            analytical, fd_grad, atol=ATOL, err_msg="DOF gradient mismatch"
        )

    def test_dof_grad_at_zero(self) -> None:
        """DOF gradient check at r=0 (the aligned starting pose)."""
        (
            ligand_mol,
            ligand_heavy,
            p0_heavy,
            centroid,
            dihedrals,
            subtrees,
            protein_coords,
            protein_typing,
            score_params,
        ) = _make_pose_system()
        n_torsions = len(dihedrals)
        x0 = np.zeros(6 + n_torsions)
        self._check_dof_grad(
            x0,
            ligand_heavy,
            p0_heavy,
            centroid,
            dihedrals,
            subtrees,
            protein_coords,
            protein_typing,
            score_params,
        )

    def test_dof_grad_at_nonzero_rotation(self) -> None:
        """DOF gradient check at ||r|| = π/2 (verifies J_r^{-T} correction)."""
        (
            ligand_mol,
            ligand_heavy,
            p0_heavy,
            centroid,
            dihedrals,
            subtrees,
            protein_coords,
            protein_typing,
            score_params,
        ) = _make_pose_system()
        n_torsions = len(dihedrals)
        x0 = np.zeros(6 + n_torsions)
        # Set rotvec to 90° rotation about [1, 0, 0]
        x0[3] = np.pi / 2
        self._check_dof_grad(
            x0,
            ligand_heavy,
            p0_heavy,
            centroid,
            dihedrals,
            subtrees,
            protein_coords,
            protein_typing,
            score_params,
        )

    def test_dof_grad_at_nonzero_rotation_and_torsions(self) -> None:
        """DOF grad with rotation AND torsions both off zero.

        Regression: the rotation gradient must use the post-torsion body-frame
        lever arms (R^T @ (pos - centroid - T)), not the pre-torsion p0. With
        the stale p0 lever arms the rotation gradient is wrong whenever torsions
        have displaced atoms -- the regime exercised here.
        """
        (
            ligand_mol,
            ligand_heavy,
            p0_heavy,
            centroid,
            dihedrals,
            subtrees,
            protein_coords,
            protein_typing,
            score_params,
        ) = _make_pose_system()
        n_torsions = len(dihedrals)
        assert n_torsions > 0, "test ligand must have rotatable bonds"
        x0 = np.zeros(6 + n_torsions)
        x0[3] = 0.4  # rotation about x
        x0[4] = -0.3  # rotation about y
        x0[6:] = np.linspace(20.0, -25.0, n_torsions)  # nonzero torsion changes
        self._check_dof_grad(
            x0,
            ligand_heavy,
            p0_heavy,
            centroid,
            dihedrals,
            subtrees,
            protein_coords,
            protein_typing,
            score_params,
        )


# =============================================================================
# Step 8 verification: constraint penalties
# =============================================================================


class TestConstraintPenalty:

    def _make_params(self, indices, weight):
        return PoseParams(
            constrained_atom_indices=tuple(indices),
            constraint_weight=weight,
        )

    def test_penalty_zero_when_no_constraint(self) -> None:
        pos = np.zeros((5, 3))
        p0 = np.ones((5, 3))
        params = self._make_params([], 0.0)
        assert _constraint_penalty_value(pos, p0, params) == 0.0

    def test_penalty_value_correct(self) -> None:
        pos = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        p0 = np.zeros((2, 3))
        params = self._make_params([0], 10.0)
        # λ * ||[1,0,0]||² = 10 * 1 = 10
        assert _constraint_penalty_value(pos, p0, params) == pytest.approx(10.0)

    def test_penalty_and_grad_agree_with_fd(self) -> None:
        """Constraint penalty gradient agrees with finite differences."""
        pos = np.array([[1.2, -0.5, 0.3], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        p0 = np.zeros((3, 3))
        params = self._make_params([0, 2], weight=5.0)

        _, atom_grad = _apply_constraint_penalty(0.0, np.zeros((3, 3)), pos, p0, params)

        for i in range(3):
            for d in range(3):
                pos_p = pos.copy()
                pos_p[i, d] += EPS
                pos_m = pos.copy()
                pos_m[i, d] -= EPS
                fd = (
                    _constraint_penalty_value(pos_p, p0, params)
                    - _constraint_penalty_value(pos_m, p0, params)
                ) / (2 * EPS)
                assert atom_grad[i, d] == pytest.approx(fd, abs=ATOL)

    def test_penalty_value_equals_apply_penalty_score(self) -> None:
        """_constraint_penalty_value and _apply_constraint_penalty give same penalty."""
        pos = np.random.default_rng(0).standard_normal((4, 3))
        p0 = np.zeros((4, 3))
        params = self._make_params([0, 1, 3], weight=7.0)

        v = _constraint_penalty_value(pos, p0, params)
        score_out, _ = _apply_constraint_penalty(0.0, np.zeros((4, 3)), pos, p0, params)
        assert v == pytest.approx(score_out)

    def test_tight_constraint_holds_atoms(self) -> None:
        """With weight=1000, constrained atoms move < 0.01 Å from initial position."""
        ligand_mol = _make_ligand("c1ccccc1CCO")
        ligand_heavy = Chem.RemoveAllHs(ligand_mol)
        p0_heavy = np.array(ligand_heavy.GetConformer(0).GetPositions())
        protein_coords = p0_heavy + np.array([3.5, 0.0, 0.0])
        protein_typing = get_atom_typing(ligand_heavy)

        # Constrain first 3 heavy atoms with a tight penalty
        constrained = tuple(range(3))
        pose_params = PoseParams(
            n_starts=1,
            constrained_atom_indices=constrained,
            constraint_weight=1000.0,
        )
        result = optimize_pose_cached(
            ligand_mol,
            protein_coords,
            protein_typing,
            params=pose_params,
        )
        result_heavy = Chem.RemoveAllHs(result.mol)
        final_pos = np.array(result_heavy.GetConformer(0).GetPositions())
        for i in constrained:
            disp = np.linalg.norm(final_pos[i] - p0_heavy[i])
            assert disp < 0.05, f"Constrained atom {i} moved {disp:.3f} Å"

    def test_unconstrained_atoms_free_to_move(self) -> None:
        """A tight subset constraint holds those atoms while the rest move freely."""
        ligand_mol = _make_ligand("c1ccccc1CCO")
        ligand_heavy = Chem.RemoveAllHs(ligand_mol)
        p0_heavy = np.array(ligand_heavy.GetConformer(0).GetPositions())
        protein_coords = p0_heavy + np.array([3.5, 0.0, 0.0])
        protein_typing = get_atom_typing(ligand_heavy)

        n_heavy = p0_heavy.shape[0]
        constrained = tuple(range(min(3, n_heavy)))
        result = optimize_pose_cached(
            ligand_mol,
            protein_coords,
            protein_typing,
            params=PoseParams(
                n_starts=1,
                constrained_atom_indices=constrained,
                constraint_weight=1000.0,
                optimize_torsions=True,
            ),
        )
        final_pos = np.array(
            Chem.RemoveAllHs(result.mol).GetConformer(0).GetPositions()
        )
        disp = np.linalg.norm(final_pos - p0_heavy, axis=1)

        # Constrained atoms stay on their input position ...
        assert disp[list(constrained)].max() < 0.05
        # ... while at least one unconstrained atom moves substantially.
        assert disp[len(constrained) :].max() > 0.5


# =============================================================================
# SMARTS constraint interface (Step 7)
# =============================================================================


class TestConstraintSmarts:

    def _make_constraint_block(self, smarts: str, weight: float = 10.0):
        from cmxflow.operators.dock import MoleculeDockBlock

        return MoleculeDockBlock(constraint_smarts=smarts, constraint_weight=weight)

    def test_smarts_resolves_to_heavy_atom_indices(self) -> None:
        """SMARTS pattern correctly resolves to heavy-atom indices."""

        block = self._make_constraint_block("c1ccccc1", weight=10.0)
        mol = _make_ligand("c1ccccc1CCO")
        block._protein_coords = np.zeros((3, 3))
        block._protein_typing = AtomTyping(
            radii=np.full(3, 1.7),
            is_hydrophobic=np.zeros(3, dtype=bool),
            is_hbond_donor=np.zeros(3, dtype=bool),
            is_hbond_acceptor=np.zeros(3, dtype=bool),
        )
        block._protein_ec_coords = np.zeros((3, 3))
        block._protein_ec_charges = np.zeros(3)

        # Trigger lazy compile via _forward
        block._forward(mol)
        assert block._constraint_smarts_mol is not None
        ligand_heavy_pre = Chem.RemoveAllHs(mol)
        matches = ligand_heavy_pre.GetSubstructMatches(block._constraint_smarts_mol)
        assert len(matches) > 0
        assert len({idx for match in matches for idx in match}) == 6

    def test_no_match_molecule_proceeds_without_constraint(self) -> None:
        """Molecule that doesn't match SMARTS docks normally (no constraint)."""

        block = self._make_constraint_block("[#7]", weight=10.0)  # N — not in propane
        mol = _make_ligand("CCC")
        block._protein_coords = np.zeros((3, 3))
        block._protein_typing = AtomTyping(
            radii=np.full(3, 1.7),
            is_hydrophobic=np.zeros(3, dtype=bool),
            is_hbond_donor=np.zeros(3, dtype=bool),
            is_hbond_acceptor=np.zeros(3, dtype=bool),
        )
        block._protein_ec_coords = np.zeros((3, 3))
        block._protein_ec_charges = np.zeros(3)

        result = block._forward(mol)
        assert result is not None

    def test_explicit_h_smarts_raises_on_first_forward(self) -> None:
        """SMARTS with explicit H raises ValueError on first docking call."""

        block = self._make_constraint_block("[#6]-[H]", weight=1.0)
        mol = _make_ligand("CCO")
        block._protein_coords = np.zeros((3, 3))
        block._protein_typing = AtomTyping(
            radii=np.full(3, 1.7),
            is_hydrophobic=np.zeros(3, dtype=bool),
            is_hbond_donor=np.zeros(3, dtype=bool),
            is_hbond_acceptor=np.zeros(3, dtype=bool),
        )
        block._protein_ec_coords = np.zeros((3, 3))
        block._protein_ec_charges = np.zeros(3)

        with pytest.raises(ValueError, match="hydrogen"):
            block._forward(mol)

    def test_invalid_smarts_raises_on_first_forward(self) -> None:
        """Invalid SMARTS string raises ValueError on first docking call."""

        block = self._make_constraint_block("[invalid(smarts", weight=1.0)
        mol = _make_ligand("CCO")
        block._protein_coords = np.zeros((3, 3))
        block._protein_typing = AtomTyping(
            radii=np.full(3, 1.7),
            is_hydrophobic=np.zeros(3, dtype=bool),
            is_hbond_donor=np.zeros(3, dtype=bool),
            is_hbond_acceptor=np.zeros(3, dtype=bool),
        )
        block._protein_ec_coords = np.zeros((3, 3))
        block._protein_ec_charges = np.zeros(3)

        with pytest.raises(ValueError, match="Invalid constraint_smarts"):
            block._forward(mol)


# =============================================================================
# Amide bond exclusion (Step 2)
# =============================================================================


class TestAmideBondExclusion:

    def test_amide_bond_excluded_from_rotatable(self) -> None:
        """Amide C-N bond is not in the rotatable bond list."""
        # N-methylacetamide: CH3-C(=O)-NH-CH3 — amide N-C bond should be excluded
        mol = Chem.RemoveAllHs(Chem.MolFromSmiles("CC(=O)NC"))
        dihedrals = get_rotatable_bonds(mol)
        # Get atom indices for the N-C(=O) bond
        amide_bonds = set()
        for bond in mol.GetBonds():
            a = bond.GetBeginAtom()
            b = bond.GetEndAtom()
            if (a.GetAtomicNum() == 7 and b.GetAtomicNum() == 6) or (
                a.GetAtomicNum() == 6 and b.GetAtomicNum() == 7
            ):
                # Check if the C has a =O neighbor
                for nb in (
                    b.GetNeighbors() if b.GetAtomicNum() == 6 else a.GetNeighbors()
                ):
                    if nb.GetAtomicNum() == 8:
                        bond_pair = frozenset([a.GetIdx(), b.GetIdx()])
                        amide_bonds.add(bond_pair)
        for _, j, k, _ in dihedrals:
            assert (
                frozenset([j, k]) not in amide_bonds
            ), f"Amide bond ({j},{k}) should not be in rotatable bonds"

    def test_ester_bond_included_in_rotatable(self) -> None:
        """Ester C(=O)-O-C bond is included (smina-compatible)."""
        # Methyl acetate: CH3-C(=O)-O-CH3
        mol = Chem.RemoveAllHs(Chem.MolFromSmiles("CC(=O)OC"))
        dihedrals = get_rotatable_bonds(mol)
        # Should have at least one rotatable bond (the acyl-oxygen or alkyl-oxygen)
        assert len(dihedrals) > 0


# =============================================================================
# Intramolecular energy (Phase 2)
# =============================================================================


class TestIntramolecularPairs:
    """Conf-dependent pair selection: 1-4+ and crossing a rotatable bond."""

    def test_butane_single_pair(self) -> None:
        """n-butane (C0-C1-C2-C3): only one movable 1-4 pair, (0, 3)."""
        mol = Chem.RemoveAllHs(Chem.MolFromSmiles("CCCC"))
        dih = get_rotatable_bonds(mol)
        pairs = _build_intramolecular_pairs(mol, get_atom_typing(mol), dih)
        got = set(zip(pairs.i_idx.tolist(), pairs.j_idx.tolist()))
        assert got == {(0, 3)}

    def test_benzene_no_pairs(self) -> None:
        """A rigid ring has no rotatable bonds and no movable intra pairs."""
        mol = Chem.RemoveAllHs(Chem.MolFromSmiles("c1ccccc1"))
        dih = get_rotatable_bonds(mol)
        pairs = _build_intramolecular_pairs(mol, get_atom_typing(mol), dih)
        assert pairs.i_idx.size == 0

    def test_ring_internal_pairs_excluded(self) -> None:
        """Ethylbenzene: ring-internal pairs are constant, hence excluded."""
        mol = Chem.RemoveAllHs(Chem.MolFromSmiles("c1ccccc1CC"))
        dih = get_rotatable_bonds(mol)
        frag = _rigid_fragments(mol, dih)
        pairs = _build_intramolecular_pairs(mol, get_atom_typing(mol), dih)
        # every selected pair crosses a fragment boundary (i.e. a rotatable bond)
        for i, j in zip(pairs.i_idx.tolist(), pairs.j_idx.tolist()):
            assert frag[i] != frag[j]


class TestIntramolecularGradient:
    """intramolecular_score_and_grad atom gradient vs finite differences."""

    def test_grad_matches_fd(self) -> None:
        mol = Chem.RemoveAllHs(_make_ligand("OCCCCCCO"))
        dih = get_rotatable_bonds(mol)
        typing = get_atom_typing(mol)
        pairs = _build_intramolecular_pairs(mol, typing, dih)
        assert pairs.i_idx.size > 0  # ensure the test exercises real pairs
        coords = np.array(mol.GetConformer().GetPositions())
        params = EmpiricalParams()

        _, grad = intramolecular_score_and_grad(coords, pairs, params)
        fd = np.zeros_like(coords)
        for a in range(coords.shape[0]):
            for k in range(3):
                cp = coords.copy()
                cp[a, k] += EPS
                sp = intramolecular_score_and_grad(cp, pairs, params)[0]
                cp[a, k] -= 2 * EPS
                sm = intramolecular_score_and_grad(cp, pairs, params)[0]
                fd[a, k] = (sp - sm) / (2 * EPS)
        np.testing.assert_allclose(grad, fd, atol=ATOL)

    def test_self_clash_penalized(self) -> None:
        """Overlapping non-bonded atoms give a positive (repulsive) energy."""
        mol = Chem.RemoveAllHs(_make_ligand("OCCCCCCO"))
        dih = get_rotatable_bonds(mol)
        pairs = _build_intramolecular_pairs(mol, get_atom_typing(mol), dih)
        coords = np.array(mol.GetConformer().GetPositions())
        # Collapse atom 0 onto the far end to force a clash.
        coords[0] = coords[-1]
        score, _ = intramolecular_score_and_grad(coords, pairs, EmpiricalParams())
        assert score > 0.0


# =============================================================================
# Basin-hopping / iterated local search (Phase 3)
# =============================================================================


class TestBasinHopping:
    """ILS path: never worse than a single local minimize, and deterministic."""

    def test_not_worse_than_single_minimize(self) -> None:
        """Basin-hopping includes hop 0 (= single minimize), so it can't be worse.

        Uses w_intra=0 so the reported intermolecular score equals the optimized
        objective and the inequality is exact.
        """
        lig, pc, pt, _ = _make_system("c1ccccc1CCCCO")
        single = optimize_pose_cached(
            lig, pc, pt, params=PoseParams(basin_hops=0, w_intra=0.0, max_iterations=50)
        )
        ils = optimize_pose_cached(
            lig,
            pc,
            pt,
            params=PoseParams(basin_hops=15, w_intra=0.0, max_iterations=50, seed=0),
        )
        assert ils.score <= single.score + 1e-6

    def test_deterministic_with_seed(self) -> None:
        """Same seed → identical basin-hopping result."""
        lig, pc, pt, _ = _make_system("c1ccccc1CCCCO")
        params = PoseParams(basin_hops=10, max_iterations=50, seed=7)
        a = optimize_pose_cached(lig, pc, pt, params=params)
        b = optimize_pose_cached(lig, pc, pt, params=params)
        assert a.score == pytest.approx(b.score)


class TestSingleDOFStep:
    """Vina-style proposal: each hop perturbs exactly one DOF group."""

    def _make_stepper(self, n_torsions: int) -> _SingleDOFStep:
        dim = 6 + n_torsions
        lo = np.full(dim, -1e9)
        hi = np.full(dim, 1e9)
        return _SingleDOFStep(
            n_torsions, lo, hi, 2.0, 0.5, 60.0, np.random.default_rng(0)
        )

    def test_mutates_exactly_one_group(self) -> None:
        """Over many calls, only one of {T, R, single torsion} changes."""
        n_torsions = 4
        step = self._make_stepper(n_torsions)
        x0 = np.zeros(6 + n_torsions)
        groups_seen = set()
        for _ in range(200):
            x = step(x0)
            t_changed = np.any(x[:3] != x0[:3])
            r_changed = np.any(x[3:6] != x0[3:6])
            tor_changed = [x[6 + k] != x0[6 + k] for k in range(n_torsions)]
            n_changed = int(t_changed) + int(r_changed) + sum(tor_changed)
            assert n_changed == 1, "exactly one group must move per hop"
            if t_changed:
                groups_seen.add("T")
            elif r_changed:
                groups_seen.add("R")
            else:
                groups_seen.add(f"tor{tor_changed.index(True)}")
        # Sampling should eventually touch every group.
        assert groups_seen == {"T", "R", "tor0", "tor1", "tor2", "tor3"}

    def test_respects_bounds(self) -> None:
        """Proposals are clipped into [lo, hi]."""
        dim = 6 + 2
        lo = np.full(dim, -0.5)
        hi = np.full(dim, 0.5)
        step = _SingleDOFStep(2, lo, hi, 5.0, 5.0, 360.0, np.random.default_rng(1))
        x0 = np.zeros(dim)
        for _ in range(100):
            x = step(x0)
            assert np.all(x >= lo - 1e-12)
            assert np.all(x <= hi + 1e-12)

    def test_rigid_only_picks_translation_or_rotation(self) -> None:
        """With no torsions, only translation/rotation groups are selectable."""
        step = self._make_stepper(0)
        x0 = np.zeros(6)
        for _ in range(50):
            x = step(x0)
            assert not np.any(x[6:] != x0[6:])  # no torsion components exist


class TestDgRestarts:
    """optimize_dg_restarts: input-pose start 0, count parity, rigid delegation."""

    def _system(self, smiles: str = "CCOCCOCc1ccccc1"):
        """Return (ligand_heavy, protein_coords, protein_typing, score_params)."""
        ligand_heavy = Chem.RemoveAllHs(_make_ligand(smiles))
        protein_coords = (
            np.array(ligand_heavy.GetConformer().GetPositions()) + 8.0
        )  # offset so the protein does not overlap the ligand
        protein_typing = get_atom_typing(ligand_heavy)
        return ligand_heavy, protein_coords, protein_typing, EmpiricalParams()

    def test_first_start_is_input_pose(self) -> None:
        lig, prot, typ, sp = self._system()
        starts = optimize_dg_restarts(
            lig,
            prot,
            typ,
            PoseParams(n_starts=5, seed=0),
            sp,
            max_tries=64,
            max_distance_geometry_samples=8,
        )
        assert len(starts) >= 1
        first = np.array(starts[0][1].GetConformer().GetPositions())
        ref = np.array(lig.GetConformer().GetPositions())
        assert np.allclose(first, ref)

    def test_count_matches_sobol_parity(self) -> None:
        """Returns n_starts starts (input pose plus n_starts - 1 sampled)."""
        lig, prot, typ, sp = self._system()
        params = PoseParams(n_starts=9, seed=0)
        starts = optimize_dg_restarts(
            lig, prot, typ, params, sp, max_tries=64, max_distance_geometry_samples=8
        )
        assert len(starts) == params.n_starts - 1

    def test_n_starts_one_returns_only_input(self) -> None:
        lig, prot, typ, sp = self._system()
        starts = optimize_dg_restarts(
            lig,
            prot,
            typ,
            PoseParams(n_starts=1, seed=0),
            sp,
            max_tries=64,
            max_distance_geometry_samples=8,
        )
        assert len(starts) == 1

    def test_rigid_delegates_to_sobol(self) -> None:
        """rigid=True forwards to the rigid Sobol path (no conformer diversity)."""
        lig, prot, typ, sp = self._system()
        params = PoseParams(n_starts=5, seed=0)
        sentinel = [(0.0, lig)]
        with patch(
            "cmxflow.operators.dock.pose._rigid_sobol_restarts",
            return_value=sentinel,
        ) as mock_sobol:
            out = optimize_dg_restarts(
                lig, prot, typ, params, sp, rigid=True, max_tries=64
            )
        mock_sobol.assert_called_once()
        assert out is sentinel

    def test_starts_are_diverse_from_input(self) -> None:
        """Non-input starts explore conformers/placements away from the input."""
        lig, prot, typ, sp = self._system()
        starts = optimize_dg_restarts(
            lig,
            prot,
            typ,
            PoseParams(n_starts=9, seed=0),
            sp,
            max_tries=128,
            max_distance_geometry_samples=8,
        )
        ref = np.array(lig.GetConformer().GetPositions())
        moved = [
            float(np.abs(np.array(m.GetConformer().GetPositions()) - ref).max())
            for _, m in starts[1:]
        ]
        assert any(d > 0.5 for d in moved)

    def test_starts_have_rigid_placement_diversity(self) -> None:
        """Non-input starts span multiple rigid placements, not one fixed pose.

        Regression guard: an earlier nested-loop+early-stop selection drained a
        single Sobol placement (all starts shared one centroid). Ranking the full
        grid must instead spread starts across distinct placements.
        """
        lig, prot, typ, sp = self._system()
        starts = optimize_dg_restarts(
            lig,
            prot,
            typ,
            PoseParams(n_starts=9, seed=0),
            sp,
            max_tries=256,
            max_distance_geometry_samples=8,
        )
        centroids = np.array(
            [
                np.array(m.GetConformer().GetPositions()).mean(axis=0)
                for _, m in starts[1:]
            ]
        )
        # More than one distinct rigid placement among the non-input starts.
        spread = float(np.linalg.norm(centroids - centroids.mean(axis=0), axis=1).max())
        assert spread > 0.5
