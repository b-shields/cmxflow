"""Tests for the ShapeOverlayScoreBlock."""

import tempfile
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cmxflow.scores.shape import ShapeOverlayScoreBlock


def _create_mol_with_conformer(smiles: str, name: str | None = None) -> Chem.Mol:
    """Create a molecule with a 3D conformer."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)  # type: ignore[attr-defined]
    AllChem.MMFFOptimizeMolecule(mol)  # type: ignore[attr-defined]
    if name:
        mol.SetProp("_Name", name)
    return mol


def _create_mol_without_conformer(smiles: str) -> Chem.Mol:
    """Create a molecule without a 3D conformer."""
    return Chem.MolFromSmiles(smiles)


def _write_sdf(mols: list[Chem.Mol], path: Path) -> None:
    """Write molecules to an SDF file."""
    writer = Chem.SDWriter(str(path))
    for mol in mols:
        writer.write(mol)
    writer.close()


class TestShapeOverlayScoreBlockInit:
    """Tests for ShapeOverlayScoreBlock initialization."""

    def test_init_creates_reference_input(self) -> None:
        """Test that initialization creates a query input file requirement."""
        block = ShapeOverlayScoreBlock()
        assert "query" in block.input_files
        assert block._reference_mols is None
        assert block._reference_names is None


class TestShapeOverlayScoreBlockObjective:
    """Tests for ShapeOverlayScoreBlock.objective method."""

    def test_shape_overlay_single_molecule(self) -> None:
        """Test shape overlay scoring with a single input molecule."""
        # Create reference molecule
        ref_mol = _create_mol_with_conformer("CCO", name="ethanol")

        # Create input molecule (same as reference for high similarity)
        input_mol = _create_mol_with_conformer("CCO")

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = Path(tmpdir) / "reference.sdf"
            _write_sdf([ref_mol], ref_path)

            block = ShapeOverlayScoreBlock()
            block.input_files["query"] = ref_path

            score = block.objective(iter([input_mol]))

            # Same molecule should have high similarity (close to 1.0)
            assert score > 0.8

    def test_shape_overlay_multiple_molecules(self) -> None:
        """Test shape overlay scoring averages over multiple molecules."""
        # Create reference molecule
        ref_mol = _create_mol_with_conformer("CCO", name="ethanol")

        # Create input molecules (one similar, one different)
        similar_mol = _create_mol_with_conformer("CCO")
        different_mol = _create_mol_with_conformer("c1ccccc1")  # benzene

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = Path(tmpdir) / "reference.sdf"
            _write_sdf([ref_mol], ref_path)

            block = ShapeOverlayScoreBlock()
            block.input_files["query"] = ref_path

            score = block.objective(iter([similar_mol, different_mol]))

            # Average should be between individual scores
            assert 0.0 < score < 1.0

    def test_shape_overlay_empty_stream(self) -> None:
        """Test shape overlay scoring handles empty input gracefully."""
        ref_mol = _create_mol_with_conformer("CCO", name="ethanol")

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = Path(tmpdir) / "reference.sdf"
            _write_sdf([ref_mol], ref_path)

            block = ShapeOverlayScoreBlock()
            block.input_files["query"] = ref_path

            score = block.objective(iter([]))

            assert score == 0.0

    def test_shape_overlay_skips_molecules_without_3d(self) -> None:
        """Test that molecules without 3D conformers are skipped."""
        ref_mol = _create_mol_with_conformer("CCO", name="ethanol")
        mol_2d = _create_mol_without_conformer("CCO")
        mol_3d = _create_mol_with_conformer("CCO")

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = Path(tmpdir) / "reference.sdf"
            _write_sdf([ref_mol], ref_path)

            block = ShapeOverlayScoreBlock()
            block.input_files["query"] = ref_path

            score = block.objective(iter([mol_2d, mol_3d]))

            # Should only score the 3D molecule
            assert score > 0.8


class TestShapeOverlayScoreBlockForward:
    """Tests for ShapeOverlayScoreBlock.forward method."""

    def test_forward_annotates_molecule(self) -> None:
        """Test that forward annotates molecule with score and reference."""
        ref_mol = _create_mol_with_conformer("CCO", name="ethanol")
        input_mol = _create_mol_with_conformer("CCO")

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = Path(tmpdir) / "reference.sdf"
            _write_sdf([ref_mol], ref_path)

            block = ShapeOverlayScoreBlock()
            block.input_files["query"] = ref_path

            result = block.forward(input_mol)

            assert result is not None
            assert result.HasProp("shape_overlay_score")
            assert result.HasProp("shape_overlay_reference")
            assert result.GetDoubleProp("shape_overlay_score") > 0.8
            assert result.GetProp("shape_overlay_reference") == "ethanol"

    def test_forward_returns_none_for_2d_molecule(self) -> None:
        """Test that forward returns None for molecules without 3D conformers."""
        ref_mol = _create_mol_with_conformer("CCO", name="ethanol")
        mol_2d = _create_mol_without_conformer("CCO")

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = Path(tmpdir) / "reference.sdf"
            _write_sdf([ref_mol], ref_path)

            block = ShapeOverlayScoreBlock()
            block.input_files["query"] = ref_path

            result = block.forward(mol_2d)

            assert result is None

    def test_forward_returns_none_for_none_input(self) -> None:
        """Test that forward returns None for None input."""
        block = ShapeOverlayScoreBlock()
        result = block.forward(None)  # type: ignore[arg-type]
        assert result is None


class TestShapeOverlayScoreBlockValidation:
    """Tests for ShapeOverlayScoreBlock validation."""

    def test_requires_3d_conformers_in_reference(self) -> None:
        """Test that reference molecules must have 3D conformers."""
        ref_mol = _create_mol_without_conformer("CCO")
        ref_mol.SetProp("_Name", "ethanol_2d")

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = Path(tmpdir) / "reference.sdf"
            _write_sdf([ref_mol], ref_path)

            block = ShapeOverlayScoreBlock()
            block.input_files["query"] = ref_path

            with pytest.raises(ValueError, match="does not have 3D conformers"):
                block.objective(iter([_create_mol_with_conformer("CCO")]))

    def test_requires_valid_reference_file(self) -> None:
        """Test that reference file must contain valid molecules."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = Path(tmpdir) / "reference.sdf"
            # Create empty SDF file
            with open(ref_path, "w") as f:
                f.write("")

            block = ShapeOverlayScoreBlock()
            block.input_files["query"] = ref_path

            # Empty file raises OSError from RDKit or ValueError from our code
            with pytest.raises((ValueError, OSError)):
                block.objective(iter([_create_mol_with_conformer("CCO")]))


class TestShapeOverlayScoreBlockCache:
    """Tests for ShapeOverlayScoreBlock caching behavior."""

    def test_reset_cache_clears_reference(self) -> None:
        """Test that reset_cache clears cached reference molecules."""
        ref_mol = _create_mol_with_conformer("CCO", name="ethanol")

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = Path(tmpdir) / "reference.sdf"
            _write_sdf([ref_mol], ref_path)

            block = ShapeOverlayScoreBlock()
            block.input_files["query"] = ref_path

            # Trigger lazy loading
            block.objective(iter([_create_mol_with_conformer("CCO")]))

            assert block._reference_mols is not None

            block.reset_cache()

            assert block._reference_mols is None
            assert block._reference_names is None


class TestShapeOverlayScoreBlockMultipleReferences:
    """Tests for ShapeOverlayScoreBlock with multiple reference molecules."""

    def test_finds_best_reference(self) -> None:
        """Test that the best matching reference is identified."""
        # Create two different reference molecules
        ref_ethanol = _create_mol_with_conformer("CCO", name="ethanol")
        ref_methanol = _create_mol_with_conformer("CO", name="methanol")

        # Input is ethanol - should match ethanol reference exactly
        input_mol = _create_mol_with_conformer("CCO")

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = Path(tmpdir) / "reference.sdf"
            _write_sdf([ref_ethanol, ref_methanol], ref_path)

            block = ShapeOverlayScoreBlock()
            block.input_files["query"] = ref_path

            result = block.forward(input_mol)

            assert result is not None
            # Ethanol should match ethanol reference better than methanol
            assert result.GetProp("shape_overlay_reference") == "ethanol"
