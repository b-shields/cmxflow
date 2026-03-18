"""Tests for the RDKitBlock operator."""

import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors  # type: ignore[attr-defined]

from cmxflow.operators.method import RDKitBlock


class TestRDKitBlockInit:
    """Tests for RDKitBlock initialization."""

    def test_init_with_callable(self) -> None:
        """Test initialization with a callable that has a proper name."""

        def my_descriptor(mol: Chem.Mol) -> float:
            return float(Descriptors.MolWt(mol))  # type: ignore[attr-defined]

        block = RDKitBlock(my_descriptor)
        assert block._method == my_descriptor
        assert block._property_name == "my_descriptor"

    def test_init_with_lambda_requires_name(self) -> None:
        """Test that lambdas require explicit name."""
        with pytest.raises(ValueError, match="Cannot determine method name"):
            RDKitBlock(Descriptors.MolWt)  # type: ignore[attr-defined]

    def test_init_with_lambda_and_name(self) -> None:
        """Test initialization with lambda when name is provided."""
        block = RDKitBlock(Descriptors.MolWt, name="MolWt")  # type: ignore[attr-defined]
        assert block._property_name == "MolWt"

    def test_init_with_string(self) -> None:
        """Test initialization with a string path."""
        block = RDKitBlock("rdkit.Chem.Descriptors.MolWt")
        assert block._property_name == "MolWt"

    def test_init_with_custom_name(self) -> None:
        """Test initialization with custom property name."""
        block = RDKitBlock(Descriptors.MolWt, name="molecular_weight")  # type: ignore[attr-defined]
        assert block._property_name == "molecular_weight"

    def test_invalid_string_path(self) -> None:
        """Test that invalid string paths raise ValueError."""
        with pytest.raises(ValueError, match="Invalid method path"):
            RDKitBlock("MolWt")

    def test_nonexistent_module(self) -> None:
        """Test that nonexistent modules raise ImportError."""
        with pytest.raises(ImportError):
            RDKitBlock("nonexistent.module.Method")

    def test_nonexistent_method(self) -> None:
        """Test that nonexistent methods raise AttributeError."""
        with pytest.raises(AttributeError):
            RDKitBlock("rdkit.Chem.Descriptors.NonexistentMethod")


class TestRDKitBlockForward:
    """Tests for RDKitBlock._forward method."""

    def test_float_result(self) -> None:
        """Test that float results are stored as DoubleProp."""
        block = RDKitBlock("rdkit.Chem.Descriptors.MolWt")
        mol = Chem.MolFromSmiles("CCO")
        result = block._forward(mol)

        assert result is not None
        assert result.HasProp("MolWt")
        assert abs(result.GetDoubleProp("MolWt") - 46.069) < 0.01

    def test_int_result(self) -> None:
        """Test that int results are stored as IntProp."""
        block = RDKitBlock("rdkit.Chem.Descriptors.HeavyAtomCount")
        mol = Chem.MolFromSmiles("CCO")
        result = block._forward(mol)

        assert result is not None
        assert result.HasProp("HeavyAtomCount")
        assert result.GetIntProp("HeavyAtomCount") == 3

    def test_string_method(self) -> None:
        """Test using a string path to specify the method."""
        block = RDKitBlock("rdkit.Chem.Descriptors.MolWt")
        mol = Chem.MolFromSmiles("CCO")
        result = block._forward(mol)

        assert result is not None
        assert result.HasProp("MolWt")

    def test_custom_property_name(self) -> None:
        """Test that custom property names are used."""
        block = RDKitBlock("rdkit.Chem.Descriptors.MolWt", name="weight")
        mol = Chem.MolFromSmiles("CCO")
        result = block._forward(mol)

        assert result is not None
        assert result.HasProp("weight")
        assert not result.HasProp("MolWt")

    def test_mol_returning_method(self) -> None:
        """Test methods that return a Mol object."""

        def add_hs(mol: Chem.Mol) -> Chem.Mol:
            return Chem.AddHs(mol)

        block = RDKitBlock(add_hs)
        mol = Chem.MolFromSmiles("CCO")
        original_atom_count = mol.GetNumAtoms()

        result = block._forward(mol)

        assert result is not None
        assert isinstance(result, Chem.Mol)
        assert result.GetNumAtoms() > original_atom_count

    def test_none_result_filters_molecule(self) -> None:
        """Test that None results filter out the molecule."""

        def always_none(mol: Chem.Mol) -> None:
            return None

        block = RDKitBlock(always_none)
        mol = Chem.MolFromSmiles("CCO")
        result = block._forward(mol)

        assert result is None

    def test_bool_result(self) -> None:
        """Test that bool results are stored as BoolProp."""

        def has_oxygen(mol: Chem.Mol) -> bool:
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == "O":
                    return True
            return False

        block = RDKitBlock(has_oxygen)
        mol = Chem.MolFromSmiles("CCO")
        result = block._forward(mol)

        assert result is not None
        assert result.HasProp("has_oxygen")
        assert result.GetBoolProp("has_oxygen") is True

    def test_string_result(self) -> None:
        """Test that string results are stored as Prop."""

        def get_smiles(mol: Chem.Mol) -> str:
            return str(Chem.MolToSmiles(mol))

        block = RDKitBlock(get_smiles, name="canonical_smiles")
        mol = Chem.MolFromSmiles("CCO")
        result = block._forward(mol)

        assert result is not None
        assert result.HasProp("canonical_smiles")
        assert result.GetProp("canonical_smiles") == "CCO"


class TestRDKitBlockIntegration:
    """Integration tests for RDKitBlock."""

    def test_call_with_iterator(self) -> None:
        """Test that the block works with an iterator of molecules."""
        block = RDKitBlock("rdkit.Chem.Descriptors.MolWt")
        smiles_list = ["CCO", "CCCO", "CCCCO"]
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]

        results = list(block(iter(mols)))

        assert len(results) == 3
        for result in results:
            assert result.HasProp("MolWt")
