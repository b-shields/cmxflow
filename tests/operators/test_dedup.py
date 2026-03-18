"""Tests for MoleculeDeduplicateBlock."""

import logging

import pytest
from rdkit import Chem

from cmxflow.operators.dedup import MoleculeDeduplicateBlock


class TestMoleculeDeduplicateForward:
    """Tests for MoleculeDeduplicateBlock._forward method."""

    def test_unique_molecules_pass_through(self) -> None:
        """Test that three distinct molecules all pass through."""
        block = MoleculeDeduplicateBlock()

        mols = [Chem.MolFromSmiles(s) for s in ["CCO", "CCC", "c1ccccc1"]]
        results = [block._forward(m) for m in mols]

        assert all(r is not None for r in results)

    def test_duplicate_removed(self) -> None:
        """Test that the same SMILES twice results in the second being dropped."""
        block = MoleculeDeduplicateBlock()

        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("CCO")

        assert block._forward(mol1) is not None
        assert block._forward(mol2) is None

    def test_keeps_first_occurrence(self) -> None:
        """Test that the first molecule (with its properties) is kept."""
        block = MoleculeDeduplicateBlock()

        mol1 = Chem.MolFromSmiles("CCO")
        mol1.SetProp("source", "first")
        mol2 = Chem.MolFromSmiles("CCO")
        mol2.SetProp("source", "second")

        result1 = block._forward(mol1)
        result2 = block._forward(mol2)

        assert result1 is not None
        assert result1.GetProp("source") == "first"
        assert result2 is None

    def test_canonical_smiles_dedup(self) -> None:
        """Test that OCC and CCO are deduplicated (same canonical SMILES)."""
        block = MoleculeDeduplicateBlock()

        mol1 = Chem.MolFromSmiles("OCC")
        mol2 = Chem.MolFromSmiles("CCO")

        assert block._forward(mol1) is not None
        assert block._forward(mol2) is None

    def test_different_molecules_not_deduped(self) -> None:
        """Test that structurally different molecules both pass."""
        block = MoleculeDeduplicateBlock()

        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("CCCO")

        assert block._forward(mol1) is not None
        assert block._forward(mol2) is not None

    def test_reset_cache_clears_seen(self) -> None:
        """Test that reset_cache allows previously seen molecules to pass again."""
        block = MoleculeDeduplicateBlock()

        mol = Chem.MolFromSmiles("CCO")
        assert block._forward(mol) is not None
        assert block._forward(Chem.MolFromSmiles("CCO")) is None

        block.reset_cache()

        assert block._forward(Chem.MolFromSmiles("CCO")) is not None


class TestMoleculeDeduplicateIntegration:
    """Integration tests for MoleculeDeduplicateBlock."""

    def test_iterator_integration(self) -> None:
        """Test full iterator with duplicates, verify count."""
        block = MoleculeDeduplicateBlock()

        smiles = ["CCO", "CCC", "CCO", "c1ccccc1", "CCC"]
        mols = [Chem.MolFromSmiles(s) for s in smiles]

        results = list(block(iter(mols)))

        assert len(results) == 3

    def test_property_preservation(self) -> None:
        """Test that properties survive via MoleculeBlock.forward."""
        block = MoleculeDeduplicateBlock()

        mol = Chem.MolFromSmiles("CCO")
        mol.SetProp("name", "ethanol")
        mol.SetDoubleProp("MolWt", 46.07)

        result = block.forward(mol)

        assert result is not None
        assert result.GetProp("name") == "ethanol"
        assert result.GetDoubleProp("MolWt") == pytest.approx(46.07)

    def test_malformed_input_skipped(self) -> None:
        """Test that None is filtered by inherited check_input."""
        block = MoleculeDeduplicateBlock()

        results = list(block(iter([None, Chem.MolFromSmiles("CCO"), None])))

        assert len(results) == 1

    def test_block_name(self) -> None:
        """Test that block.name is correct."""
        block = MoleculeDeduplicateBlock()
        assert block.name == "MoleculeDeduplicate"

    def test_no_required_inputs(self) -> None:
        """Test that no input_files or input_text are required."""
        block = MoleculeDeduplicateBlock()
        assert len(block.input_files) == 0
        assert len(block.input_text) == 0

    def test_duplicate_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that info log message is emitted when duplicate is removed."""
        block = MoleculeDeduplicateBlock()

        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("CCO")

        block._forward(mol1)

        with caplog.at_level(logging.INFO, logger="cmxflow.operators.dedup"):
            block._forward(mol2)

        assert any("Removing duplicate" in record.message for record in caplog.records)
