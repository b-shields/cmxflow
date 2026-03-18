"""Tests for PropertyHeadBlock and PropertyTailBlock operators."""

import pytest
from rdkit import Chem

from cmxflow.operators.select import PropertyHeadBlock, PropertyTailBlock


class TestPropertyHeadBlock:
    """Tests for PropertyHeadBlock."""

    def test_returns_top_n_highest_values(self) -> None:
        """Test that head returns molecules with highest property values."""
        block = PropertyHeadBlock()
        block.input_text["property"] = "score"
        block.input_text["count"] = "3"

        mols = []
        for i, score in enumerate([1.0, 5.0, 3.0, 2.0, 4.0]):
            mol = Chem.MolFromSmiles("C" * (i + 1))
            mol.SetDoubleProp("score", score)
            mols.append(mol)

        results = list(block(iter(mols)))

        assert len(results) == 3
        # Should be sorted descending: 5.0, 4.0, 3.0
        scores = [m.GetDoubleProp("score") for m in results]
        assert scores == [5.0, 4.0, 3.0]

    def test_count_zero_returns_all_sorted(self) -> None:
        """Test that count=0 returns all molecules sorted descending."""
        block = PropertyHeadBlock()
        block.input_text["property"] = "score"
        block.input_text["count"] = "0"

        mols = []
        for score in [1.0, 3.0, 2.0]:
            mol = Chem.MolFromSmiles("C")
            mol.SetDoubleProp("score", score)
            mols.append(mol)

        results = list(block(iter(mols)))

        assert len(results) == 3
        scores = [m.GetDoubleProp("score") for m in results]
        assert scores == [3.0, 2.0, 1.0]  # Descending

    def test_empty_count_returns_all_sorted(self) -> None:
        """Test that empty count returns all molecules sorted."""
        block = PropertyHeadBlock()
        block.input_text["property"] = "score"
        block.input_text["count"] = ""

        mols = []
        for score in [1.0, 3.0, 2.0]:
            mol = Chem.MolFromSmiles("C")
            mol.SetDoubleProp("score", score)
            mols.append(mol)

        results = list(block(iter(mols)))

        assert len(results) == 3

    def test_raises_keyerror_missing_property(self) -> None:
        """Test that molecules without the property raise KeyError."""
        block = PropertyHeadBlock()
        block.input_text["property"] = "score"
        block.input_text["count"] = "10"

        mol1 = Chem.MolFromSmiles("C")
        mol1.SetDoubleProp("score", 5.0)

        mol2 = Chem.MolFromSmiles("CC")
        # No score property

        with pytest.raises(KeyError, match="Molecule missing property: score"):
            list(block(iter([mol1, mol2])))

    def test_count_larger_than_input(self) -> None:
        """Test count larger than input size returns all molecules."""
        block = PropertyHeadBlock()
        block.input_text["property"] = "score"
        block.input_text["count"] = "100"

        mols = []
        for score in [1.0, 2.0]:
            mol = Chem.MolFromSmiles("C")
            mol.SetDoubleProp("score", score)
            mols.append(mol)

        results = list(block(iter(mols)))

        assert len(results) == 2

    def test_block_name(self) -> None:
        """Test that block has correct name."""
        block = PropertyHeadBlock()
        assert block.name == "PropertyHead"

    def test_input_text_registered(self) -> None:
        """Test that property and count are registered as input_text."""
        block = PropertyHeadBlock()
        assert "property" in block.input_text
        assert "count" in block.input_text


class TestPropertyTailBlock:
    """Tests for PropertyTailBlock."""

    def test_returns_bottom_n_lowest_values(self) -> None:
        """Test that tail returns molecules with lowest property values."""
        block = PropertyTailBlock()
        block.input_text["property"] = "energy"
        block.input_text["count"] = "3"

        mols = []
        for i, energy in enumerate([10.0, 5.0, 15.0, 3.0, 8.0]):
            mol = Chem.MolFromSmiles("C" * (i + 1))
            mol.SetDoubleProp("energy", energy)
            mols.append(mol)

        results = list(block(iter(mols)))

        assert len(results) == 3
        # Should be sorted ascending: 3.0, 5.0, 8.0
        energies = [m.GetDoubleProp("energy") for m in results]
        assert energies == [3.0, 5.0, 8.0]

    def test_count_zero_returns_all_sorted_ascending(self) -> None:
        """Test that count=0 returns all molecules sorted ascending."""
        block = PropertyTailBlock()
        block.input_text["property"] = "energy"
        block.input_text["count"] = "0"

        mols = []
        for energy in [3.0, 1.0, 2.0]:
            mol = Chem.MolFromSmiles("C")
            mol.SetDoubleProp("energy", energy)
            mols.append(mol)

        results = list(block(iter(mols)))

        assert len(results) == 3
        energies = [m.GetDoubleProp("energy") for m in results]
        assert energies == [1.0, 2.0, 3.0]  # Ascending

    def test_block_name(self) -> None:
        """Test that block has correct name."""
        block = PropertyTailBlock()
        assert block.name == "PropertyTail"

    def test_raises_keyerror_missing_property(self) -> None:
        """Test that molecules without the property raise KeyError."""
        block = PropertyTailBlock()
        block.input_text["property"] = "energy"
        block.input_text["count"] = "5"

        mol1 = Chem.MolFromSmiles("C")
        # No energy property

        with pytest.raises(KeyError, match="Molecule missing property: energy"):
            list(block(iter([mol1])))


class TestPropertySelectIntegration:
    """Integration tests for property select blocks."""

    def test_int_property(self) -> None:
        """Test selection on integer property."""
        block = PropertyHeadBlock()
        block.input_text["property"] = "rank"
        block.input_text["count"] = "2"

        mols = []
        for rank in [3, 1, 2]:
            mol = Chem.MolFromSmiles("C")
            mol.SetIntProp("rank", rank)
            mols.append(mol)

        results = list(block(iter(mols)))

        assert len(results) == 2
        ranks = [m.GetIntProp("rank") for m in results]
        assert ranks == [3, 2]  # Highest first

    def test_string_numeric_property(self) -> None:
        """Test selection on string property that is numeric."""
        block = PropertyHeadBlock()
        block.input_text["property"] = "score"
        block.input_text["count"] = "2"

        mols = []
        for score in ["1.5", "3.0", "2.0"]:
            mol = Chem.MolFromSmiles("C")
            mol.SetProp("score", score)
            mols.append(mol)

        results = list(block(iter(mols)))

        assert len(results) == 2
        # Should pick 3.0 and 2.0 (highest)
        scores = [float(m.GetProp("score")) for m in results]
        assert scores == [3.0, 2.0]

    def test_repr_shows_required_inputs(self) -> None:
        """Test that repr shows property and count inputs."""
        block = PropertyHeadBlock()
        repr_str = repr(block)
        assert "property" in repr_str
        assert "count" in repr_str

    def test_empty_iterator(self) -> None:
        """Test handling of empty input."""
        block = PropertyHeadBlock()
        block.input_text["property"] = "score"
        block.input_text["count"] = "5"

        results = list(block(iter([])))

        assert len(results) == 0

    def test_no_property_passes_all(self) -> None:
        """Test that empty property passes all molecules through."""
        block = PropertyHeadBlock()
        block.input_text["property"] = ""
        block.input_text["count"] = "2"

        mols = [Chem.MolFromSmiles("C"), Chem.MolFromSmiles("CC")]

        results = list(block(iter(mols)))

        # All passed through when no property specified
        assert len(results) == 2

    def test_non_numeric_property_raises_keyerror(self) -> None:
        """Test that non-numeric property value raises KeyError."""
        block = PropertyHeadBlock()
        block.input_text["property"] = "name"
        block.input_text["count"] = "2"

        mol = Chem.MolFromSmiles("C")
        mol.SetProp("name", "methane")  # Not numeric

        with pytest.raises(KeyError, match="cannot be converted to numeric"):
            list(block(iter([mol])))

    def test_invalid_count_uses_zero(self) -> None:
        """Test that invalid count string defaults to 0 (all)."""
        block = PropertyHeadBlock()
        block.input_text["property"] = "score"
        block.input_text["count"] = "abc"  # Invalid

        mols = []
        for score in [1.0, 2.0, 3.0]:
            mol = Chem.MolFromSmiles("C")
            mol.SetDoubleProp("score", score)
            mols.append(mol)

        results = list(block(iter(mols)))

        # Should return all molecules since invalid count defaults to 0
        assert len(results) == 3
