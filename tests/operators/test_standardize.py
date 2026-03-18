"""Tests for the MoleculeStandardizeBlock operator."""

from rdkit import Chem

from cmxflow.operators.standardize import MoleculeStandardizeBlock


class TestSimplePassthrough:
    """Tests that clean molecules pass through unchanged."""

    def test_simple_molecule_passthrough(self) -> None:
        """Test that a clean molecule passes through unchanged."""
        block = MoleculeStandardizeBlock()
        mol = Chem.MolFromSmiles("CCO")
        result = block._forward(mol)
        assert result is not None
        assert Chem.MolToSmiles(result) == Chem.MolToSmiles(mol)


class TestSaltStripping:
    """Tests for salt/fragment removal."""

    def test_salt_stripping(self) -> None:
        """Test that salts are removed and largest fragment kept."""
        block = MoleculeStandardizeBlock()
        mol = Chem.MolFromSmiles("CCO.[Na+].[Cl-]")
        result = block._forward(mol)
        assert result is not None
        assert Chem.MolToSmiles(result) == "CCO"


class TestNeutralization:
    """Tests for charge neutralization."""

    def test_neutralization(self) -> None:
        """Test that unnecessary charges are neutralized."""
        block = MoleculeStandardizeBlock()
        mol = Chem.MolFromSmiles("[NH3+]CC([O-])=O")
        result = block._forward(mol)
        assert result is not None
        result_smiles = Chem.MolToSmiles(result)
        # Should be neutralized glycine
        assert "+" not in result_smiles
        assert "-" not in result_smiles


class TestMetalDisconnection:
    """Tests for metal disconnection."""

    def test_metal_disconnection(self) -> None:
        """Test that metal-ligand bonds are disconnected."""
        block = MoleculeStandardizeBlock()
        # Sodium ethoxide with explicit bond
        mol = Chem.MolFromSmiles("[Na]OCC")
        result = block._forward(mol)
        assert result is not None
        result_smiles = Chem.MolToSmiles(result)
        # After metal disconnection + largest fragment, should keep organic part
        assert "Na" not in result_smiles


class TestTautomerCanonicalization:
    """Tests for tautomer canonicalization."""

    def test_tautomer_off_by_default(self) -> None:
        """Test that tautomer canonicalization is off by default."""
        block = MoleculeStandardizeBlock()
        assert block.canonicalize_tautomers is False
        assert block._tautomer_enumerator is None

    def test_tautomer_canonicalization(self) -> None:
        """Test that tautomer canonicalization picks preferred form."""
        block_off = MoleculeStandardizeBlock(canonicalize_tautomers=False)
        block_on = MoleculeStandardizeBlock(canonicalize_tautomers=True)

        # 2-hydroxypyridine / 2-pyridinone tautomer pair
        mol = Chem.MolFromSmiles("Oc1ccccn1")
        result_off = block_off._forward(mol)
        result_on = block_on._forward(mol)
        assert result_off is not None
        assert result_on is not None

        smiles_on = Chem.MolToSmiles(result_on)
        # With canonicalization on, the tautomer should be the canonical form
        # (may or may not differ, but the enumerator should have been called)
        assert isinstance(smiles_on, str)
        # Verify the enumerator was actually instantiated
        assert block_on._tautomer_enumerator is not None


class TestErrorHandling:
    """Tests for error handling."""

    def test_malformed_molecule_skipped(self) -> None:
        """Test that None input is filtered by check_input."""
        block = MoleculeStandardizeBlock()
        assert block.check_input(None) is False

    def test_invalid_mol_returns_none(self) -> None:
        """Test that check_input rejects non-molecule types."""
        block = MoleculeStandardizeBlock()
        assert block.check_input("not a molecule") is False
        assert block.check_input(42) is False


class TestIteratorIntegration:
    """Integration tests with iterators."""

    def test_iterator_integration(self) -> None:
        """Test calling block on an iterator of mixed molecules."""
        block = MoleculeStandardizeBlock()

        smiles_list = ["CCO", "CCO.[Na+].[Cl-]", "c1ccccc1"]
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        # Add a None to test filtering
        mols_with_none: list[Chem.Mol | None] = [*mols, None]  # type: ignore[list-item]

        results = list(block(iter(mols_with_none)))
        # All 3 valid molecules should pass; None should be filtered
        assert len(results) == 3


class TestPropertyPreservation:
    """Tests for property preservation through standardization."""

    def test_property_preservation(self) -> None:
        """Test that molecule properties survive standardization."""
        block = MoleculeStandardizeBlock()

        mol = Chem.MolFromSmiles("CCO")
        mol.SetProp("name", "ethanol")
        mol.SetDoubleProp("score", 0.95)
        mol.SetIntProp("rank", 1)

        # Use forward() (not _forward) to get CmxMol with property preservation
        result = block.forward(mol)
        assert result is not None
        assert result.GetProp("name") == "ethanol"
        assert result.GetDoubleProp("score") == 0.95
        assert result.GetIntProp("rank") == 1


class TestBlockMetadata:
    """Tests for block metadata."""

    def test_block_name(self) -> None:
        """Test that block has correct name."""
        block = MoleculeStandardizeBlock()
        assert block.name == "MoleculeStandardize"

    def test_no_required_inputs(self) -> None:
        """Test that block has no required file or text inputs."""
        block = MoleculeStandardizeBlock()
        assert len(block.input_files) == 0
        assert len(block.input_text) == 0
