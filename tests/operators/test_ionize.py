"""Tests for the IonizeMoleculeBlock operator."""

import pickle
from unittest.mock import MagicMock, patch

from rdkit import Chem

from cmxflow import wrap_mol
from cmxflow.operators.ionize import IonizeMoleculeBlock


class TestIonizeMoleculeBlockInit:
    """Tests for block initialization."""

    def test_block_name(self) -> None:
        """Test that block has correct name."""
        block = IonizeMoleculeBlock()
        assert block.name == "IonizeMolecule"

    def test_default_params(self) -> None:
        """Test default parameter values."""
        block = IonizeMoleculeBlock()
        assert block.ph_min == 6.4
        assert block.ph_max == 8.4
        assert block.params["precision"].get() == 1.0
        assert block.params["max_variants"].get() == 128

    def test_custom_ph_range(self) -> None:
        """Test custom pH range initialization."""
        block = IonizeMoleculeBlock(ph_min=5.0, ph_max=9.0)
        assert block.ph_min == 5.0
        assert block.ph_max == 9.0

    def test_pickle_round_trip(self) -> None:
        """Test that block survives pickle serialization."""
        block = IonizeMoleculeBlock(ph_min=5.0, ph_max=9.0)
        data = pickle.dumps(block)
        restored = pickle.loads(data)
        assert restored.name == "IonizeMolecule"
        assert restored.ph_min == 5.0
        assert restored.ph_max == 9.0
        assert restored.params["precision"].get() == 1.0
        assert restored._amide_pattern is not None


class TestTertiaryAmideFix:
    """Tests for tertiary amide deprotonation fix."""

    def test_nacetylpiperidine_not_protonated(self) -> None:
        """Test that N-acetylpiperidine tertiary amide N is not left protonated."""
        block = IonizeMoleculeBlock()
        # Simulate a protonated tertiary amide (what dimorphite_dl would produce)
        protonated = Chem.MolFromSmiles("CC(=O)[NH+]1CCCCC1")
        fixed = block._fix_tertiary_amides(protonated)
        smiles = Chem.MolToSmiles(fixed)
        assert "[NH+]" not in smiles
        assert "+" not in smiles

    def test_secondary_amide_unaffected(self) -> None:
        """Test that secondary amides (with H2) are not altered."""
        block = IonizeMoleculeBlock()
        # Regular protonated secondary amine (2 H's) - should NOT match our pattern
        # Our pattern targets NX4+ with exactly H1 bonded to C=O
        mol = Chem.MolFromSmiles("CC(=O)NC")
        original = Chem.MolToSmiles(mol)
        fixed = block._fix_tertiary_amides(mol)
        assert Chem.MolToSmiles(fixed) == original

    def test_smarts_pattern_matches_protonated_tertiary_amide(self) -> None:
        """Test that the SMARTS pattern matches protonated tertiary amide."""
        block = IonizeMoleculeBlock()
        protonated = Chem.MolFromSmiles("CC(=O)[NH+]1CCCCC1")
        assert protonated.HasSubstructMatch(block._amide_pattern)

    def test_smarts_pattern_does_not_match_regular_amine(self) -> None:
        """Test that the SMARTS pattern does not match protonated basic amine."""
        block = IonizeMoleculeBlock()
        # Protonated piperidine (no C=O neighbor)
        mol = Chem.MolFromSmiles("[NH2+]1CCCCC1")
        assert not mol.HasSubstructMatch(block._amide_pattern)

    def test_multiple_amide_fixes(self) -> None:
        """Test that multiple protonated tertiary amides are all fixed."""
        block = IonizeMoleculeBlock()
        # Two protonated tertiary amides
        mol = Chem.MolFromSmiles("CC(=O)[NH+]1CCCCC1.CC(=O)[NH+]1CCCCC1")
        fixed = block._fix_tertiary_amides(mol)
        smiles = Chem.MolToSmiles(fixed)
        assert "[NH+]" not in smiles


class TestIonizeMoleculeBlockForward:
    """Tests for the forward method with mocked dimorphite_dl."""

    @patch.dict("sys.modules", {"dimorphite_dl": MagicMock()})
    def test_carboxylic_acid_deprotonated(self) -> None:
        """Test that a carboxylic acid produces deprotonated variant."""
        import sys

        mock_dl = sys.modules["dimorphite_dl"]
        mock_dl.protonate_smiles.return_value = ["OC(=O)c1ccccc1", "[O-]C(=O)c1ccccc1"]

        block = IonizeMoleculeBlock()
        mol = Chem.MolFromSmiles("OC(=O)c1ccccc1")
        results = list(block.forward(mol))
        assert len(results) == 2
        smiles_set = {Chem.MolToSmiles(r) for r in results}
        assert any("-" in s for s in smiles_set)

    @patch.dict("sys.modules", {"dimorphite_dl": MagicMock()})
    def test_amine_protonated(self) -> None:
        """Test that an amine produces protonated variant."""
        import sys

        mock_dl = sys.modules["dimorphite_dl"]
        mock_dl.protonate_smiles.return_value = ["CCN", "CC[NH3+]"]

        block = IonizeMoleculeBlock()
        mol = Chem.MolFromSmiles("CCN")
        results = list(block.forward(mol))
        assert len(results) == 2
        smiles_set = {Chem.MolToSmiles(r) for r in results}
        assert any("+" in s for s in smiles_set)

    @patch.dict("sys.modules", {"dimorphite_dl": MagicMock()})
    def test_neutral_molecule_passthrough(self) -> None:
        """Test that a neutral molecule passes through."""
        import sys

        mock_dl = sys.modules["dimorphite_dl"]
        mock_dl.protonate_smiles.return_value = ["c1ccccc1"]

        block = IonizeMoleculeBlock()
        mol = Chem.MolFromSmiles("c1ccccc1")
        results = list(block.forward(mol))
        assert len(results) == 1
        assert Chem.MolToSmiles(results[0]) == "c1ccccc1"

    @patch.dict("sys.modules", {"dimorphite_dl": MagicMock()})
    def test_property_preservation(self) -> None:
        """Test that properties from input are copied to variants."""
        import sys

        mock_dl = sys.modules["dimorphite_dl"]
        mock_dl.protonate_smiles.return_value = ["CCO", "CC[OH2+]"]

        block = IonizeMoleculeBlock()
        mol = Chem.MolFromSmiles("CCO")
        mol.SetProp("name", "ethanol")
        mol.SetDoubleProp("score", 0.95)
        mol.SetIntProp("rank", 1)

        results = list(block.forward(mol))
        for r in results:
            assert r.GetProp("name") == "ethanol"
            assert r.GetDoubleProp("score") == 0.95
            assert r.GetIntProp("rank") == 1

    @patch.dict("sys.modules", {"dimorphite_dl": MagicMock()})
    def test_property_preservation_from_mol(self) -> None:
        """Test property preservation from a Mol (wrapped) input."""
        import sys

        mock_dl = sys.modules["dimorphite_dl"]
        mock_dl.protonate_smiles.return_value = ["CCO"]

        block = IonizeMoleculeBlock()
        mol = wrap_mol(Chem.MolFromSmiles("CCO"))
        mol.SetProp("source", "test")

        results = list(block.forward(mol))
        assert len(results) == 1
        assert results[0].GetProp("source") == "test"


class TestIonizeMoleculeBlockIntegration:
    """Integration tests for iterator behavior."""

    @patch.dict("sys.modules", {"dimorphite_dl": MagicMock()})
    def test_iterator_with_multiple_mols(self) -> None:
        """Test that the block processes an iterator of molecules."""
        import sys

        mock_dl = sys.modules["dimorphite_dl"]
        # First call returns 2 variants, second returns 1
        mock_dl.protonate_smiles.side_effect = [
            ["CCO", "CC[OH2+]"],
            ["CCN", "CC[NH3+]"],
        ]

        block = IonizeMoleculeBlock()
        mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCN")]
        results = list(block(iter(mols)))
        # 2 + 2 = 4 total variants
        assert len(results) == 4

    @patch.dict("sys.modules", {"dimorphite_dl": MagicMock()})
    def test_deduplication_after_fix(self) -> None:
        """Test that duplicates produced after amide fix are removed."""
        import sys

        mock_dl = sys.modules["dimorphite_dl"]
        # Return duplicates that become identical after canonicalization
        mock_dl.protonate_smiles.return_value = ["CCO", "OCC"]

        block = IonizeMoleculeBlock()
        mol = Chem.MolFromSmiles("CCO")
        results = list(block.forward(mol))
        assert len(results) == 1

    @patch.dict("sys.modules", {"dimorphite_dl": MagicMock()})
    def test_1_to_n_expansion(self) -> None:
        """Test that one molecule can produce multiple variants."""
        import sys

        mock_dl = sys.modules["dimorphite_dl"]
        mock_dl.protonate_smiles.return_value = [
            "OC(=O)c1ccccc1",
            "[O-]C(=O)c1ccccc1",
        ]

        block = IonizeMoleculeBlock()
        mol = Chem.MolFromSmiles("OC(=O)c1ccccc1")
        results = list(block.forward(mol))
        assert len(results) == 2


class TestErrorHandling:
    """Tests for error handling."""

    @patch.dict("sys.modules", {"dimorphite_dl": MagicMock()})
    def test_dimorphite_failure_falls_back(self) -> None:
        """Test that dimorphite_dl failure falls back to original SMILES."""
        import sys

        mock_dl = sys.modules["dimorphite_dl"]
        mock_dl.protonate_smiles.side_effect = RuntimeError("dimorphite failed")

        block = IonizeMoleculeBlock()
        mol = Chem.MolFromSmiles("CCO")
        results = list(block.forward(mol))
        assert len(results) == 1
        assert Chem.MolToSmiles(results[0]) == "CCO"

    @patch.dict("sys.modules", {"dimorphite_dl": MagicMock()})
    def test_invalid_smiles_from_dimorphite_skipped(self) -> None:
        """Test that invalid SMILES from dimorphite_dl are skipped."""
        import sys

        mock_dl = sys.modules["dimorphite_dl"]
        mock_dl.protonate_smiles.return_value = ["CCO", "INVALID_SMILES", "CCN"]

        block = IonizeMoleculeBlock()
        mol = Chem.MolFromSmiles("CCO")
        results = list(block.forward(mol))
        # CCO and CCN are valid, INVALID_SMILES is skipped
        assert len(results) == 2

    @patch.dict("sys.modules", {"dimorphite_dl": MagicMock()})
    def test_none_input_filtered(self) -> None:
        """Test that None input is filtered by check_input."""
        block = IonizeMoleculeBlock()
        assert block.check_input(None) is False

    def test_forward_raises_not_implemented(self) -> None:
        """Test that _forward raises NotImplementedError."""
        block = IonizeMoleculeBlock()
        mol = Chem.MolFromSmiles("CCO")
        try:
            block._forward(mol)
            assert False, "Should have raised NotImplementedError"
        except NotImplementedError:
            pass
