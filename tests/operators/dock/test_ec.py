"""Tests for electrostatic complementarity scoring."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.transform import Rotation

from cmxflow.operators.dock import MoleculeDockBlock
from cmxflow.operators.dock.ec import (
    compute_esp_at_points,
    compute_gasteiger_charges,
    electrostatic_complementarity,
    fibonacci_sphere,
    generate_sas_points,
)
from cmxflow.operators.dock.pose import OptimizationResult
from cmxflow.operators.dock.score import AtomTyping

# =============================================================================
# fibonacci_sphere
# =============================================================================


class TestFibonacciSphere:
    """Tests for fibonacci_sphere."""

    def test_correct_shape(self) -> None:
        """Test output shape matches requested number of points."""
        pts = fibonacci_sphere(100)
        assert pts.shape == (100, 3)

    def test_points_on_unit_sphere(self) -> None:
        """Test all points lie on the unit sphere."""
        pts = fibonacci_sphere(200)
        norms = np.linalg.norm(pts, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_deterministic(self) -> None:
        """Test that results are deterministic."""
        pts1 = fibonacci_sphere(50)
        pts2 = fibonacci_sphere(50)
        np.testing.assert_array_equal(pts1, pts2)

    def test_single_point(self) -> None:
        """Test single point case."""
        pts = fibonacci_sphere(1)
        assert pts.shape == (1, 3)


# =============================================================================
# generate_sas_points
# =============================================================================


class TestGenerateSasPoints:
    """Tests for generate_sas_points."""

    def test_single_atom(self) -> None:
        """Test SAS points around a single atom."""
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([1.5])
        pts = generate_sas_points(coords, radii, probe_radius=1.4, n_sphere_points=50)
        # All 50 points should survive (no neighbors to bury them)
        assert len(pts) == 50
        # Points should be at distance radius + probe
        dists = np.linalg.norm(pts, axis=1)
        np.testing.assert_allclose(dists, 2.9, atol=1e-10)

    def test_two_overlapping_atoms(self) -> None:
        """Test that overlapping atoms produce fewer surface points."""
        # Two atoms close together — some points should be buried
        coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        radii = np.array([1.5, 1.5])
        pts = generate_sas_points(coords, radii, probe_radius=1.4, n_sphere_points=50)
        # Should have fewer than 100 (2 * 50) points due to burial
        assert len(pts) < 100
        assert len(pts) > 0

    def test_empty_input(self) -> None:
        """Test empty atom array."""
        coords = np.empty((0, 3))
        radii = np.empty(0)
        pts = generate_sas_points(coords, radii)
        assert pts.shape == (0, 3)

    def test_correct_shape(self) -> None:
        """Test output is (n, 3)."""
        coords = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        radii = np.array([1.7, 1.7])
        pts = generate_sas_points(coords, radii)
        assert pts.ndim == 2
        assert pts.shape[1] == 3


# =============================================================================
# compute_gasteiger_charges
# =============================================================================


class TestComputeGasteigerCharges:
    """Tests for compute_gasteiger_charges."""

    def test_ethanol_charges(self) -> None:
        """Test charges for ethanol."""
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        charges = compute_gasteiger_charges(mol)
        assert len(charges) == mol.GetNumAtoms()
        # Net charge should be ~0 for neutral molecule
        assert abs(np.sum(charges)) < 0.05

    def test_neutral_molecule_net_charge(self) -> None:
        """Test that neutral molecule charges sum to ~0."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        mol = Chem.AddHs(mol)
        charges = compute_gasteiger_charges(mol)
        assert abs(np.sum(charges)) < 0.05

    def test_no_nan_in_output(self) -> None:
        """Test that NaN values are replaced with 0."""
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        charges = compute_gasteiger_charges(mol)
        assert not np.any(np.isnan(charges))


# =============================================================================
# compute_esp_at_points
# =============================================================================


class TestComputeEspAtPoints:
    """Tests for compute_esp_at_points."""

    def test_single_charge_known_distance(self) -> None:
        """Test ESP from a single charge at a known distance."""
        points = np.array([[2.0, 0.0, 0.0]])
        atom_coords = np.array([[0.0, 0.0, 0.0]])
        charges = np.array([1.0])
        esp = compute_esp_at_points(points, atom_coords, charges, cutoff=10.0)
        # ESP = k * q / r = 332.06 * 1.0 / 2.0 = 166.03
        np.testing.assert_allclose(esp[0], 166.03, atol=0.01)

    def test_cutoff_excludes_distant_atoms(self) -> None:
        """Test that atoms beyond cutoff don't contribute."""
        points = np.array([[0.0, 0.0, 0.0]])
        atom_coords = np.array([[20.0, 0.0, 0.0]])
        charges = np.array([1.0])
        esp = compute_esp_at_points(points, atom_coords, charges, cutoff=10.0)
        assert esp[0] == 0.0

    def test_zero_charges_zero_esp(self) -> None:
        """Test that zero charges produce zero ESP."""
        points = np.array([[1.0, 0.0, 0.0]])
        atom_coords = np.array([[0.0, 0.0, 0.0]])
        charges = np.array([0.0])
        esp = compute_esp_at_points(points, atom_coords, charges, cutoff=10.0)
        assert esp[0] == 0.0

    def test_empty_inputs(self) -> None:
        """Test empty point and atom arrays."""
        esp = compute_esp_at_points(
            np.empty((0, 3)), np.array([[0.0, 0.0, 0.0]]), np.array([1.0])
        )
        assert len(esp) == 0

        esp = compute_esp_at_points(
            np.array([[0.0, 0.0, 0.0]]), np.empty((0, 3)), np.empty(0)
        )
        assert esp[0] == 0.0


# =============================================================================
# electrostatic_complementarity
# =============================================================================


class TestElectrostaticComplementarity:
    """Tests for electrostatic_complementarity."""

    def _make_mol_3d(self, smiles: str) -> Chem.Mol:
        """Create a molecule with 3D coordinates."""
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        return mol

    def test_opposite_charges_positive_ec(self) -> None:
        """Test that opposite charges yield positive EC."""
        # Positive ligand near negative protein charges
        ligand = self._make_mol_3d("CCO")
        # Create "protein" with negative charges near the ligand
        prot_coords = np.array(ligand.GetConformer().GetPositions()) + 3.0
        prot_charges = -compute_gasteiger_charges(ligand)

        ec = electrostatic_complementarity(ligand, prot_coords, prot_charges)
        # Should tend positive (complementary)
        assert isinstance(ec, float)
        assert -1.0 <= ec <= 1.0

    def test_identical_charges_negative_ec(self) -> None:
        """Test that identical charges yield negative EC."""
        ligand = self._make_mol_3d("CCO")
        # Protein with same charges as ligand (anti-complementary)
        prot_coords = np.array(ligand.GetConformer().GetPositions()) + 3.0
        prot_charges = compute_gasteiger_charges(ligand)

        ec = electrostatic_complementarity(ligand, prot_coords, prot_charges)
        assert isinstance(ec, float)
        assert -1.0 <= ec <= 1.0

    def test_result_in_range(self) -> None:
        """Test EC is always in [-1, 1]."""
        ligand = self._make_mol_3d("c1ccccc1O")
        prot_coords = np.array([[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
        prot_charges = np.array([0.5, -0.3])
        ec = electrostatic_complementarity(ligand, prot_coords, prot_charges)
        assert -1.0 <= ec <= 1.0

    def test_no_conformer_returns_zero(self) -> None:
        """Test molecule without conformer returns 0.0."""
        mol = Chem.MolFromSmiles("CCO")
        prot_coords = np.array([[0.0, 0.0, 0.0]])
        prot_charges = np.array([1.0])
        ec = electrostatic_complementarity(mol, prot_coords, prot_charges)
        assert ec == 0.0

    def test_zero_protein_charges_returns_zero(self) -> None:
        """Test that all-zero protein charges return 0.0."""
        ligand = self._make_mol_3d("CCO")
        prot_coords = np.array(ligand.GetConformer().GetPositions()) + 3.0
        prot_charges = np.zeros(ligand.GetNumAtoms())
        ec = electrostatic_complementarity(ligand, prot_coords, prot_charges)
        assert ec == 0.0


# =============================================================================
# MoleculeDockBlock EC integration
# =============================================================================


class TestMoleculeDockBlockEC:
    """Integration tests for EC in MoleculeDockBlock.

    EC is factored entirely out of the search: the optimizer is EC-agnostic and
    EC is computed once on the final pose as a reporting-only score.
    """

    def _block_with_protein(self) -> MoleculeDockBlock:
        block = MoleculeDockBlock()
        block._protein_coords = np.zeros((3, 3))
        block._protein_typing = AtomTyping(
            radii=np.array([1.7, 1.7, 1.7]),
            is_hydrophobic=np.array([False, False, False]),
            is_hbond_donor=np.array([False, False, False]),
            is_hbond_acceptor=np.array([False, False, False]),
        )
        block._protein_ec_coords = np.zeros((3, 3))
        block._protein_ec_charges = np.zeros(3)
        return block

    def _ligand(self) -> Chem.Mol:
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        return mol

    def _opt_result(self, mol: Chem.Mol, score: float = -5.0) -> OptimizationResult:
        return OptimizationResult(
            mol=mol,
            score=score,
            initial_score=-3.0,
            translation=np.zeros(3),
            rotation=Rotation.identity(),
            converged=True,
        )

    def test_no_ec_passed_to_optimizer(self) -> None:
        """The optimizer is EC-agnostic: no EC kwargs are threaded into it."""
        block = self._block_with_protein()
        mol = self._ligand()

        with (
            patch(
                "cmxflow.operators.dock.dock.optimize_pose_cached",
                return_value=self._opt_result(mol),
            ) as mock_opt,
            patch(
                "cmxflow.operators.dock.dock.optimize_dg_restarts",
                return_value=[(-5.0, mol)],
            ),
        ):
            result = block._forward(mol)

        assert result is not None
        call_kwargs = mock_opt.call_args[1]
        assert "w_ec" not in call_kwargs
        assert "protein_ec_coords" not in call_kwargs
        assert "protein_ec_charges" not in call_kwargs

    def test_ec_computed_as_postprocess(self) -> None:
        """docking_ec comes from a single post-search EC call on the final pose;
        docking_score/docking_empirical stay pure empirical."""
        block = self._block_with_protein()
        mol = self._ligand()

        with (
            patch(
                "cmxflow.operators.dock.dock.optimize_pose_cached",
                return_value=self._opt_result(mol, score=-6.2),
            ),
            patch(
                "cmxflow.operators.dock.dock.optimize_dg_restarts",
                return_value=[(-6.2, mol)],
            ),
            patch(
                "cmxflow.operators.dock.dock.electrostatic_complementarity",
                return_value=0.42,
            ) as mock_ec,
        ):
            result = block._forward(mol)

        assert result is not None
        # Exactly one EC evaluation, on the final optimized pose.
        mock_ec.assert_called_once()
        assert mock_ec.call_args[0][0] is result
        assert result.GetDoubleProp("docking_ec") == pytest.approx(0.42)
        # EC never touches the reported empirical score.
        assert result.GetDoubleProp("docking_score") == pytest.approx(-6.2)
        assert result.GetDoubleProp("docking_empirical") == pytest.approx(-6.2)

    def test_docking_ec_always_present(self) -> None:
        """Test that docking_ec property is always set."""
        block = self._block_with_protein()
        mol = self._ligand()

        with (
            patch(
                "cmxflow.operators.dock.dock.optimize_pose_cached",
                return_value=self._opt_result(mol),
            ),
            patch(
                "cmxflow.operators.dock.dock.optimize_dg_restarts",
                return_value=[(-5.0, mol)],
            ),
        ):
            result = block._forward(mol)
            assert result is not None
            assert result.HasProp("docking_ec")

    def test_check_output_requires_docking_ec(self) -> None:
        """Test that check_output validates docking_ec."""
        from cmxflow.operators.dock import MoleculeDockBlock

        block = MoleculeDockBlock()

        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)

        # Missing docking_ec should fail
        mol.SetDoubleProp("docking_initial_pose_score", -3.0)
        mol.SetDoubleProp("docking_score", -5.0)
        mol.SetBoolProp("docking_converged", True)
        assert block.check_output(mol) is False

        # With docking_ec should pass
        mol.SetDoubleProp("docking_ec", 0.5)
        assert block.check_output(mol) is True

    def test_ec_data_cached_after_load_receptor(self) -> None:
        """Test that EC protein data is cached after _load_receptor."""
        from cmxflow.operators.dock import MoleculeDockBlock

        block = MoleculeDockBlock()

        # Create a minimal PDB-like mol
        mol = Chem.MolFromSmiles("CC")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)

        with (
            patch("cmxflow.operators.dock.dock.Chem.MolFromPDBFile", return_value=mol),
            patch("cmxflow.operators.dock.dock.Path") as mock_path,
        ):
            mock_path.return_value.exists.return_value = True
            block.input_files["receptor"] = Path("dummy.pdb")
            block._load_receptor()

        assert block._protein_ec_coords is not None
        assert block._protein_ec_charges is not None
        # EC data should include H atoms (more atoms than heavy-atom-only)
        assert block._protein_coords is not None
        assert len(block._protein_ec_coords) >= len(block._protein_coords)


# =============================================================================
# optimize_pose_cached is EC-agnostic
# =============================================================================


class TestOptimizePoseCachedIsEcAgnostic:
    """The pose optimizer must know nothing about EC after the refactor."""

    def test_signature_has_no_ec_params(self) -> None:
        """optimize_pose_cached exposes no EC parameters."""
        import inspect

        from cmxflow.operators.dock.pose import optimize_pose_cached

        params = inspect.signature(optimize_pose_cached).parameters
        assert "w_ec" not in params
        assert "protein_ec_coords" not in params
        assert "protein_ec_charges" not in params

    def test_result_has_no_ec_field(self) -> None:
        """OptimizationResult no longer carries an EC value."""
        import dataclasses

        fields = {f.name for f in dataclasses.fields(OptimizationResult)}
        assert "ec" not in fields
