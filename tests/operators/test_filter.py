"""Tests for the PropertyFilterBlock operator."""

import pytest
from rdkit import Chem

from cmxflow.operators.filter import (
    FilterExpressionError,
    PropertyFilterBlock,
    parse_filter_expression,
)


class TestFilterExpressionParsing:
    """Tests for filter expression parsing."""

    def test_simple_greater_than(self) -> None:
        """Test parsing simple greater than expression."""
        conditions = parse_filter_expression("MW>200")
        assert len(conditions) == 1
        assert conditions[0].property_name == "MW"
        assert conditions[0].operator == ">"
        assert conditions[0].value == 200.0

    def test_simple_less_than_or_equal(self) -> None:
        """Test parsing less than or equal expression."""
        conditions = parse_filter_expression("logP<=5")
        assert len(conditions) == 1
        assert conditions[0].property_name == "logP"
        assert conditions[0].operator == "<="
        assert conditions[0].value == 5.0

    def test_simple_equality(self) -> None:
        """Test parsing equality expression."""
        conditions = parse_filter_expression("HBD==2")
        assert len(conditions) == 1
        assert conditions[0].property_name == "HBD"
        assert conditions[0].operator == "=="
        assert conditions[0].value == 2.0

    def test_simple_not_equal(self) -> None:
        """Test parsing not equal expression."""
        conditions = parse_filter_expression("charge!=0")
        assert len(conditions) == 1
        assert conditions[0].property_name == "charge"
        assert conditions[0].operator == "!="
        assert conditions[0].value == 0.0

    def test_reverse_comparison(self) -> None:
        """Test parsing reversed comparison (value op property)."""
        conditions = parse_filter_expression("200<MW")
        assert len(conditions) == 1
        assert conditions[0].property_name == "MW"
        assert conditions[0].operator == ">"
        assert conditions[0].value == 200.0

    def test_reverse_comparison_less_equal(self) -> None:
        """Test parsing reversed less than or equal."""
        conditions = parse_filter_expression("5>=logP")
        assert len(conditions) == 1
        assert conditions[0].property_name == "logP"
        assert conditions[0].operator == "<="
        assert conditions[0].value == 5.0

    def test_range_expression(self) -> None:
        """Test parsing range expression."""
        conditions = parse_filter_expression("200<MW<500")
        assert len(conditions) == 2
        # First condition: MW > 200
        assert conditions[0].property_name == "MW"
        assert conditions[0].operator == ">"
        assert conditions[0].value == 200.0
        # Second condition: MW < 500
        assert conditions[1].property_name == "MW"
        assert conditions[1].operator == "<"
        assert conditions[1].value == 500.0

    def test_range_expression_inclusive(self) -> None:
        """Test parsing inclusive range expression."""
        conditions = parse_filter_expression("0<=logP<=5")
        assert len(conditions) == 2
        assert conditions[0].property_name == "logP"
        assert conditions[0].operator == ">="
        assert conditions[0].value == 0.0
        assert conditions[1].property_name == "logP"
        assert conditions[1].operator == "<="
        assert conditions[1].value == 5.0

    def test_multiple_conditions(self) -> None:
        """Test parsing multiple comma-separated conditions."""
        conditions = parse_filter_expression("MW>200, logP>0")
        assert len(conditions) == 2
        assert conditions[0].property_name == "MW"
        assert conditions[0].operator == ">"
        assert conditions[1].property_name == "logP"
        assert conditions[1].operator == ">"

    def test_multiple_conditions_with_range(self) -> None:
        """Test parsing multiple conditions including a range."""
        conditions = parse_filter_expression("200<MW<500, logP>0, HBD<=5")
        assert len(conditions) == 4
        # Range produces 2 conditions
        assert conditions[0].property_name == "MW"
        assert conditions[1].property_name == "MW"
        assert conditions[2].property_name == "logP"
        assert conditions[3].property_name == "HBD"

    def test_whitespace_handling(self) -> None:
        """Test that whitespace is handled correctly."""
        conditions = parse_filter_expression("  MW > 200  ,  logP <= 5  ")
        assert len(conditions) == 2
        assert conditions[0].property_name == "MW"
        assert conditions[1].property_name == "logP"

    def test_negative_values(self) -> None:
        """Test parsing expressions with negative values."""
        conditions = parse_filter_expression("logP>-2")
        assert len(conditions) == 1
        assert conditions[0].value == -2.0

    def test_float_values(self) -> None:
        """Test parsing expressions with float values."""
        conditions = parse_filter_expression("logP<=5.5")
        assert len(conditions) == 1
        assert conditions[0].value == 5.5

    def test_empty_expression(self) -> None:
        """Test that empty expression returns no conditions."""
        assert parse_filter_expression("") == []
        assert parse_filter_expression("   ") == []

    def test_invalid_both_numbers(self) -> None:
        """Test that expression with two numbers raises error."""
        with pytest.raises(FilterExpressionError, match="both sides are numbers"):
            parse_filter_expression("200>100")

    def test_invalid_both_properties(self) -> None:
        """Test that expression with two properties raises error."""
        with pytest.raises(
            FilterExpressionError, match="both sides are property names"
        ):
            parse_filter_expression("MW>logP")

    def test_invalid_syntax(self) -> None:
        """Test that invalid syntax raises error."""
        with pytest.raises(FilterExpressionError, match="Invalid filter expression"):
            parse_filter_expression("MW>>200")

    def test_invalid_missing_operator(self) -> None:
        """Test that missing operator raises error."""
        with pytest.raises(FilterExpressionError, match="Invalid filter expression"):
            parse_filter_expression("MW200")

    def test_underscore_in_property_name(self) -> None:
        """Test property names with underscores."""
        conditions = parse_filter_expression("Mol_Wt>200")
        assert conditions[0].property_name == "Mol_Wt"

    def test_property_name_with_numbers(self) -> None:
        """Test property names containing numbers."""
        conditions = parse_filter_expression("prop2d>0")
        assert conditions[0].property_name == "prop2d"


class TestPropertyFilterBlockForward:
    """Tests for PropertyFilterBlock._forward method."""

    def test_passes_molecule_with_matching_property(self) -> None:
        """Test that molecules passing the filter are returned."""
        block = PropertyFilterBlock()
        block.input_text["filters"] = "MolWt>40"

        mol = Chem.MolFromSmiles("CCO")
        mol.SetDoubleProp("MolWt", 46.07)

        result = block._forward(mol)
        assert result is not None

    def test_filters_molecule_not_matching(self) -> None:
        """Test that molecules failing the filter return None."""
        block = PropertyFilterBlock()
        block.input_text["filters"] = "MolWt>100"

        mol = Chem.MolFromSmiles("CCO")
        mol.SetDoubleProp("MolWt", 46.07)

        result = block._forward(mol)
        assert result is None

    def test_filters_molecule_missing_property(self) -> None:
        """Test that molecules with missing properties are filtered."""
        block = PropertyFilterBlock()
        block.input_text["filters"] = "MolWt>40"

        mol = Chem.MolFromSmiles("CCO")
        # No MolWt property set

        result = block._forward(mol)
        assert result is None

    def test_passes_with_empty_filter(self) -> None:
        """Test that empty filter passes all molecules."""
        block = PropertyFilterBlock()
        block.input_text["filters"] = ""

        mol = Chem.MolFromSmiles("CCO")

        result = block._forward(mol)
        assert result is not None

    def test_multiple_conditions_all_pass(self) -> None:
        """Test multiple conditions where all pass."""
        block = PropertyFilterBlock()
        block.input_text["filters"] = "MolWt>40, logP>-1"

        mol = Chem.MolFromSmiles("CCO")
        mol.SetDoubleProp("MolWt", 46.07)
        mol.SetDoubleProp("logP", -0.3)

        result = block._forward(mol)
        assert result is not None

    def test_multiple_conditions_one_fails(self) -> None:
        """Test multiple conditions where one fails."""
        block = PropertyFilterBlock()
        block.input_text["filters"] = "MolWt>40, logP>0"

        mol = Chem.MolFromSmiles("CCO")
        mol.SetDoubleProp("MolWt", 46.07)
        mol.SetDoubleProp("logP", -0.3)

        result = block._forward(mol)
        assert result is None

    def test_range_filter_passes(self) -> None:
        """Test range filter that molecule passes."""
        block = PropertyFilterBlock()
        block.input_text["filters"] = "200<MolWt<500"

        mol = Chem.MolFromSmiles("c1ccccc1")  # benzene
        mol.SetDoubleProp("MolWt", 300.0)

        result = block._forward(mol)
        assert result is not None

    def test_range_filter_fails_low(self) -> None:
        """Test range filter that molecule fails (too low)."""
        block = PropertyFilterBlock()
        block.input_text["filters"] = "200<MolWt<500"

        mol = Chem.MolFromSmiles("CCO")
        mol.SetDoubleProp("MolWt", 46.07)

        result = block._forward(mol)
        assert result is None

    def test_range_filter_fails_high(self) -> None:
        """Test range filter that molecule fails (too high)."""
        block = PropertyFilterBlock()
        block.input_text["filters"] = "200<MolWt<500"

        mol = Chem.MolFromSmiles("CCO")
        mol.SetDoubleProp("MolWt", 600.0)

        result = block._forward(mol)
        assert result is None

    def test_int_property(self) -> None:
        """Test filtering on integer property."""
        block = PropertyFilterBlock()
        block.input_text["filters"] = "HBD<=3"

        mol = Chem.MolFromSmiles("CCO")
        mol.SetIntProp("HBD", 1)

        result = block._forward(mol)
        assert result is not None

    def test_string_numeric_property(self) -> None:
        """Test filtering on string property that is numeric."""
        block = PropertyFilterBlock()
        block.input_text["filters"] = "score>0.5"

        mol = Chem.MolFromSmiles("CCO")
        mol.SetProp("score", "0.75")

        result = block._forward(mol)
        assert result is not None

    def test_equality_operator(self) -> None:
        """Test equality operator."""
        block = PropertyFilterBlock()
        block.input_text["filters"] = "charge==0"

        mol = Chem.MolFromSmiles("CCO")
        mol.SetIntProp("charge", 0)

        result = block._forward(mol)
        assert result is not None

        mol2 = Chem.MolFromSmiles("CCO")
        mol2.SetIntProp("charge", 1)

        result2 = block._forward(mol2)
        assert result2 is None

    def test_not_equal_operator(self) -> None:
        """Test not equal operator."""
        block = PropertyFilterBlock()
        block.input_text["filters"] = "charge!=0"

        mol = Chem.MolFromSmiles("CCO")
        mol.SetIntProp("charge", 1)

        result = block._forward(mol)
        assert result is not None

    def test_reset_cache_clears_conditions(self) -> None:
        """Test that reset_cache clears parsed conditions."""
        block = PropertyFilterBlock()
        block.input_text["filters"] = "MolWt>100"

        mol = Chem.MolFromSmiles("CCO")
        mol.SetDoubleProp("MolWt", 46.07)

        # First call - parses and caches
        result1 = block._forward(mol)
        assert result1 is None

        # Change filter
        block.input_text["filters"] = "MolWt>40"
        # Still uses cached conditions
        result2 = block._forward(mol)
        assert result2 is None

        # Reset cache
        block.reset_cache()
        # Now uses new filter
        result3 = block._forward(mol)
        assert result3 is not None


class TestPropertyFilterBlockIntegration:
    """Integration tests for PropertyFilterBlock."""

    def test_call_with_iterator(self) -> None:
        """Test that the block works with an iterator of molecules."""
        block = PropertyFilterBlock()
        block.input_text["filters"] = "MolWt>50"

        # Create molecules with MolWt property
        smiles_list = ["C", "CCO", "CCCCCCCC"]  # MW ~16, ~46, ~114
        mols = []
        for i, s in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(s)
            # Approximate molecular weights
            weights = [16.04, 46.07, 114.23]
            mol.SetDoubleProp("MolWt", weights[i])
            mols.append(mol)

        results = list(block(iter(mols)))

        # Only the octane (MW ~114) should pass
        assert len(results) == 1

    def test_block_name(self) -> None:
        """Test that block has correct name."""
        block = PropertyFilterBlock()
        assert block.name == "PropertyFilter"

    def test_input_text_registered(self) -> None:
        """Test that 'filters' is registered as input_text."""
        block = PropertyFilterBlock()
        assert "filters" in block.input_text

    def test_repr_shows_filter_input(self) -> None:
        """Test that repr shows filter input requirement."""
        block = PropertyFilterBlock()
        repr_str = repr(block)
        assert "filters" in repr_str

    def test_invalid_filter_raises_at_forward(self) -> None:
        """Test that invalid filter raises error when _forward is called."""
        block = PropertyFilterBlock()
        block.input_text["filters"] = "invalid>>syntax"

        mol = Chem.MolFromSmiles("CCO")
        mol.SetDoubleProp("MolWt", 46.07)

        with pytest.raises(FilterExpressionError):
            block._forward(mol)
