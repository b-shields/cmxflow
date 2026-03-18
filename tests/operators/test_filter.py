"""Tests for the PropertyFilterBlock and SubstructureFilterBlock operators."""

import pytest
from rdkit import Chem

from cmxflow.operators.filter import (
    FilterExpressionError,
    PropertyFilterBlock,
    SubstructureFilterBlock,
    SubstructureFilterError,
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

    def test_raises_keyerror_missing_property(self) -> None:
        """Test that molecules with missing properties raise KeyError."""
        block = PropertyFilterBlock()
        block.input_text["filters"] = "MolWt>40"

        mol = Chem.MolFromSmiles("CCO")
        # No MolWt property set

        with pytest.raises(KeyError, match="Molecule missing property: MolWt"):
            block._forward(mol)

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


class TestSubstructureFilterBlockSmarts:
    """Tests for SubstructureFilterBlock with SMARTS patterns."""

    def test_remove_mode_filters_match(self) -> None:
        """Test that remove mode filters out matching molecules."""
        block = SubstructureFilterBlock()
        block.input_text["query"] = "[OH]"  # hydroxyl group
        block.input_text["mode"] = "remove"

        mol = Chem.MolFromSmiles("CCO")  # ethanol - has OH
        result = block._forward(mol)
        assert result is None

    def test_remove_mode_keeps_non_match(self) -> None:
        """Test that remove mode keeps non-matching molecules."""
        block = SubstructureFilterBlock()
        block.input_text["query"] = "[OH]"  # hydroxyl group
        block.input_text["mode"] = "remove"

        mol = Chem.MolFromSmiles("CC")  # ethane - no OH
        result = block._forward(mol)
        assert result is not None

    def test_keep_mode_keeps_match(self) -> None:
        """Test that keep mode keeps matching molecules."""
        block = SubstructureFilterBlock()
        block.input_text["query"] = "[OH]"
        block.input_text["mode"] = "keep"

        mol = Chem.MolFromSmiles("CCO")
        result = block._forward(mol)
        assert result is not None

    def test_keep_mode_filters_non_match(self) -> None:
        """Test that keep mode filters out non-matching molecules."""
        block = SubstructureFilterBlock()
        block.input_text["query"] = "[OH]"
        block.input_text["mode"] = "keep"

        mol = Chem.MolFromSmiles("CC")
        result = block._forward(mol)
        assert result is None

    def test_carboxylic_acid_pattern(self) -> None:
        """Test matching carboxylic acid SMARTS."""
        block = SubstructureFilterBlock()
        block.input_text["query"] = "[CX3](=O)[OX2H1]"
        block.input_text["mode"] = "keep"

        # Acetic acid has carboxylic acid
        mol = Chem.MolFromSmiles("CC(=O)O")
        result = block._forward(mol)
        assert result is not None

        # Ethanol does not
        mol2 = Chem.MolFromSmiles("CCO")
        result2 = block._forward(mol2)
        assert result2 is None

    def test_invalid_smarts_raises_error(self) -> None:
        """Test that invalid SMARTS pattern raises error."""
        block = SubstructureFilterBlock()
        block.input_text["query"] = "[invalid"

        mol = Chem.MolFromSmiles("CCO")
        with pytest.raises(SubstructureFilterError, match="Invalid SMARTS pattern"):
            block._forward(mol)

    def test_default_mode_is_remove(self) -> None:
        """Test that default mode is 'remove'."""
        block = SubstructureFilterBlock()
        block.input_text["query"] = "[OH]"

        mol = Chem.MolFromSmiles("CCO")
        result = block._forward(mol)
        assert result is None  # should be filtered out (remove mode)

    def test_fluorine_pattern(self) -> None:
        """Test matching fluorine atoms."""
        block = SubstructureFilterBlock()
        block.input_text["query"] = "[F]"
        block.input_text["mode"] = "remove"

        # Fluorobenzene has fluorine
        mol = Chem.MolFromSmiles("Fc1ccccc1")
        result = block._forward(mol)
        assert result is None

        # Benzene does not
        mol2 = Chem.MolFromSmiles("c1ccccc1")
        result2 = block._forward(mol2)
        assert result2 is not None


class TestSubstructureFilterBlockCatalogs:
    """Tests for SubstructureFilterBlock with RDKit catalogs."""

    def test_pains_filter(self) -> None:
        """Test PAINS catalog filtering."""
        block = SubstructureFilterBlock()
        block.input_text["query"] = "PAINS"
        block.input_text["mode"] = "remove"

        # rhodanine is a known PAINS structure
        mol = Chem.MolFromSmiles("O=C1NC(=S)SC1=Cc1ccccc1")
        result = block._forward(mol)
        assert result is None

    def test_multiple_catalogs(self) -> None:
        """Test multiple catalogs."""
        block = SubstructureFilterBlock()
        block.input_text["query"] = "PAINS BRENK"
        block.input_text["mode"] = "remove"

        # Simple molecule should pass
        mol = Chem.MolFromSmiles("CCO")
        result = block._forward(mol)
        assert result is not None

    def test_catalog_case_insensitive(self) -> None:
        """Test that catalog names are case-insensitive."""
        block = SubstructureFilterBlock()
        block.input_text["query"] = "pains Brenk NIH"
        block.input_text["mode"] = "remove"

        mol = Chem.MolFromSmiles("CCO")
        result = block._forward(mol)
        assert result is not None

    def test_keep_mode_with_catalog(self) -> None:
        """Test keep mode with catalog matching."""
        block = SubstructureFilterBlock()
        block.input_text["query"] = "PAINS"
        block.input_text["mode"] = "keep"

        # rhodanine matches PAINS - should be kept
        mol = Chem.MolFromSmiles("O=C1NC(=S)SC1=Cc1ccccc1")
        result = block._forward(mol)
        assert result is not None

        # Simple molecule - should be filtered out
        mol2 = Chem.MolFromSmiles("CCO")
        result2 = block._forward(mol2)
        assert result2 is None

    def test_zinc_catalog(self) -> None:
        """Test ZINC catalog."""
        block = SubstructureFilterBlock()
        block.input_text["query"] = "ZINC"
        block.input_text["mode"] = "remove"

        mol = Chem.MolFromSmiles("CCO")
        result = block._forward(mol)
        assert result is not None


class TestSubstructureFilterBlockCombined:
    """Tests for SubstructureFilterBlock with pattern + catalogs (OR logic)."""

    def test_or_logic_pattern_matches(self) -> None:
        """Test that pattern match alone triggers filter."""
        block = SubstructureFilterBlock()
        block.input_text["query"] = "PAINS [OH]"
        block.input_text["mode"] = "remove"

        # Has OH but no PAINS
        mol = Chem.MolFromSmiles("CCO")
        result = block._forward(mol)
        assert result is None  # filtered due to pattern match

    def test_or_logic_catalog_matches(self) -> None:
        """Test that catalog match alone triggers filter."""
        block = SubstructureFilterBlock()
        block.input_text["query"] = "PAINS [Br]"
        block.input_text["mode"] = "remove"

        # rhodanine matches PAINS but no Br
        mol = Chem.MolFromSmiles("O=C1NC(=S)SC1=Cc1ccccc1")
        result = block._forward(mol)
        assert result is None  # filtered due to catalog match

    def test_or_logic_neither_matches(self) -> None:
        """Test that molecule passes when neither matches."""
        block = SubstructureFilterBlock()
        block.input_text["query"] = "PAINS [Br]"
        block.input_text["mode"] = "remove"

        # No Br and no PAINS
        mol = Chem.MolFromSmiles("CCCC")
        result = block._forward(mol)
        assert result is not None

    def test_or_logic_both_match(self) -> None:
        """Test when both pattern and catalog match."""
        block = SubstructureFilterBlock()
        block.input_text["query"] = "PAINS c1ccccc1"
        block.input_text["mode"] = "remove"

        # rhodanine has benzene and matches PAINS
        mol = Chem.MolFromSmiles("O=C1NC(=S)SC1=Cc1ccccc1")
        result = block._forward(mol)
        assert result is None


class TestSubstructureFilterBlockModes:
    """Tests for SubstructureFilterBlock mode validation."""

    def test_invalid_mode_raises_error(self) -> None:
        """Test that invalid mode raises error."""
        block = SubstructureFilterBlock()
        block.input_text["query"] = "[OH]"
        block.input_text["mode"] = "invalid"

        mol = Chem.MolFromSmiles("CCO")
        with pytest.raises(SubstructureFilterError, match="Invalid mode"):
            block._forward(mol)

    def test_mode_case_insensitive(self) -> None:
        """Test that mode is case-insensitive."""
        block = SubstructureFilterBlock()
        block.input_text["query"] = "[OH]"
        block.input_text["mode"] = "REMOVE"

        mol = Chem.MolFromSmiles("CCO")
        result = block._forward(mol)
        assert result is None

    def test_mode_whitespace_handling(self) -> None:
        """Test that mode handles whitespace."""
        block = SubstructureFilterBlock()
        block.input_text["query"] = "[OH]"
        block.input_text["mode"] = "  keep  "

        mol = Chem.MolFromSmiles("CCO")
        result = block._forward(mol)
        assert result is not None


class TestSubstructureFilterBlockEmptyConfig:
    """Tests for SubstructureFilterBlock with empty configuration."""

    def test_empty_query_passes_all(self) -> None:
        """Test that empty query passes all molecules."""
        block = SubstructureFilterBlock()

        mol = Chem.MolFromSmiles("CCO")
        result = block._forward(mol)
        assert result is not None

    def test_catalog_only(self) -> None:
        """Test with only catalog in query."""
        block = SubstructureFilterBlock()
        block.input_text["query"] = "PAINS"
        block.input_text["mode"] = "remove"

        mol = Chem.MolFromSmiles("CCO")
        result = block._forward(mol)
        assert result is not None

    def test_pattern_only(self) -> None:
        """Test with only pattern in query."""
        block = SubstructureFilterBlock()
        block.input_text["query"] = "[OH]"
        block.input_text["mode"] = "remove"

        mol = Chem.MolFromSmiles("CCO")
        result = block._forward(mol)
        assert result is None


class TestSubstructureFilterBlockIntegration:
    """Integration tests for SubstructureFilterBlock."""

    def test_call_with_iterator(self) -> None:
        """Test that the block works with an iterator of molecules."""
        block = SubstructureFilterBlock()
        block.input_text["query"] = "[OH]"
        block.input_text["mode"] = "remove"

        smiles_list = ["CCO", "CC", "C(=O)O", "CCCC"]  # 2 have OH
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]

        results = list(block(iter(mols)))

        # Only CC and CCCC should pass (no OH)
        assert len(results) == 2

    def test_block_name(self) -> None:
        """Test that block has correct name."""
        block = SubstructureFilterBlock()
        assert block.name == "SubstructureFilter"

    def test_input_text_registered(self) -> None:
        """Test that inputs are registered."""
        block = SubstructureFilterBlock()
        assert "query" in block.input_text
        assert "mode" in block.input_text
        assert "pattern" not in block.input_text
        assert "catalogs" not in block.input_text
        assert "annotate" not in block.input_text

    def test_check_output(self) -> None:
        """Test check_output validation."""
        block = SubstructureFilterBlock()
        mol = Chem.MolFromSmiles("CCO")

        assert block.check_output(mol) is True
        assert block.check_output(None) is False
        assert block.check_output("not a mol") is False

    def test_metadata_preserved(self) -> None:
        """Test that molecule properties are preserved."""
        block = SubstructureFilterBlock()
        block.input_text["query"] = "[F]"
        block.input_text["mode"] = "remove"

        mol = Chem.MolFromSmiles("CCO")
        mol.SetProp("name", "ethanol")
        mol.SetDoubleProp("MolWt", 46.07)

        result = block._forward(mol)
        assert result is not None
        assert result.GetProp("name") == "ethanol"
        assert result.GetDoubleProp("MolWt") == 46.07
