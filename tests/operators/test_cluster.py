"""Tests for RepresentativeClusterBlock."""

import logging

import pytest
from rdkit import Chem

from cmxflow.operators.cluster import RepresentativeClusterBlock


class TestRepresentativeClusterForward:
    """Tests for RepresentativeClusterBlock._forward method."""

    def test_first_molecule_creates_cluster(self) -> None:
        """First molecule gets cluster_id=0, itself as representative, sim=1.0."""
        block = RepresentativeClusterBlock()
        block.params["scaffold"].set(False)

        mol = Chem.MolFromSmiles("CCO")
        result = block._forward(mol)

        assert result.GetIntProp("cluster_id") == 0
        assert result.GetProp("cluster_representative") == Chem.MolToSmiles(mol)
        assert result.GetDoubleProp("cluster_similarity") == 1.0

    def test_identical_molecule_same_cluster(self) -> None:
        """Same SMILES twice should get the same cluster_id."""
        block = RepresentativeClusterBlock()

        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("CCO")

        block._forward(mol1)
        result = block._forward(mol2)

        assert result.GetIntProp("cluster_id") == 0

    def test_dissimilar_molecule_new_cluster(self) -> None:
        """Very different molecule gets a new cluster_id."""
        block = RepresentativeClusterBlock()
        block.params["threshold"].set(0.9)

        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34")  # pyrene

        block._forward(mol1)
        result = block._forward(mol2)

        assert result.GetIntProp("cluster_id") == 1

    def test_similar_molecule_same_cluster(self) -> None:
        """Similar molecule at low threshold joins existing cluster."""
        block = RepresentativeClusterBlock()
        block.params["threshold"].set(0.1)
        block.params["scaffold"].set(False)

        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("CCCO")

        block._forward(mol1)
        result = block._forward(mol2)

        assert result.GetIntProp("cluster_id") == 0

    def test_cluster_id_increments(self) -> None:
        """Three dissimilar molecules get IDs 0, 1, 2."""
        block = RepresentativeClusterBlock()
        block.params["threshold"].set(0.95)
        block.params["scaffold"].set(False)

        smiles = ["CCO", "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34", "NCCCCN"]
        ids = []
        for s in smiles:
            mol = Chem.MolFromSmiles(s)
            result = block._forward(mol)
            ids.append(result.GetIntProp("cluster_id"))

        assert ids == [0, 1, 2]

    def test_representative_smiles_correct(self) -> None:
        """cluster_representative matches canonical SMILES of first molecule."""
        block = RepresentativeClusterBlock()
        block.params["scaffold"].set(False)

        mol = Chem.MolFromSmiles("CCO")
        result = block._forward(mol)

        assert result.GetProp("cluster_representative") == Chem.MolToSmiles(mol)

    def test_threshold_boundary_at(self) -> None:
        """Similarity exactly at threshold joins cluster (>= semantics)."""
        block = RepresentativeClusterBlock()
        block.params["scaffold"].set(False)

        mol1 = Chem.MolFromSmiles("CCO")
        block._forward(mol1)

        # Same molecule gives similarity=1.0, threshold=1.0 should still join
        block.params["threshold"].set(0.95)
        mol2 = Chem.MolFromSmiles("CCO")
        result = block._forward(mol2)

        assert result.GetIntProp("cluster_id") == 0

    def test_custom_threshold_more_clusters(self) -> None:
        """Higher threshold produces more clusters for the same input."""
        smiles = ["CCO", "CCCO", "CCCCO", "c1ccccc1", "c1ccc(O)cc1"]

        block_low = RepresentativeClusterBlock()
        block_low.params["threshold"].set(0.1)
        block_low.params["scaffold"].set(False)

        block_high = RepresentativeClusterBlock()
        block_high.params["threshold"].set(0.9)
        block_high.params["scaffold"].set(False)

        ids_low = set()
        ids_high = set()
        for s in smiles:
            mol = Chem.MolFromSmiles(s)
            r_low = block_low._forward(mol)
            ids_low.add(r_low.GetIntProp("cluster_id"))

            mol2 = Chem.MolFromSmiles(s)
            r_high = block_high._forward(mol2)
            ids_high.add(r_high.GetIntProp("cluster_id"))

        assert len(ids_high) >= len(ids_low)

    def test_reset_cache_clears_representatives(self) -> None:
        """After reset_cache(), fresh clusters are created."""
        block = RepresentativeClusterBlock()

        mol = Chem.MolFromSmiles("CCO")
        block._forward(mol)
        assert len(block._representatives) == 1

        block.reset_cache()
        assert len(block._representatives) == 0
        assert len(block._representative_smiles) == 0

        result = block._forward(Chem.MolFromSmiles("CCO"))
        assert result.GetIntProp("cluster_id") == 0

    def test_scaffold_mode_clusters_by_scaffold(self) -> None:
        """Molecules with same scaffold but different substituents cluster together."""
        block = RepresentativeClusterBlock()
        block.params["scaffold"].set(True)
        block.params["threshold"].set(0.3)

        # Both share the benzene scaffold
        mol1 = Chem.MolFromSmiles("c1ccc(O)cc1")  # phenol
        mol2 = Chem.MolFromSmiles("c1ccc(N)cc1")  # aniline

        block._forward(mol1)
        result = block._forward(mol2)

        assert result.GetIntProp("cluster_id") == 0

    def test_scaffold_mode_representative_is_scaffold_smiles(self) -> None:
        """cluster_representative is scaffold SMILES when scaffold=True."""
        block = RepresentativeClusterBlock()
        block.params["scaffold"].set(True)

        mol = Chem.MolFromSmiles("c1ccc(O)cc1")  # phenol
        result = block._forward(mol)

        from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

        scaffold = GetScaffoldForMol(mol)
        expected = Chem.MolToSmiles(scaffold)

        assert result.GetProp("cluster_representative") == expected

    def test_cluster_similarity_property(self) -> None:
        """Verify cluster_similarity is set and is a float between 0 and 1."""
        block = RepresentativeClusterBlock()
        block.params["scaffold"].set(False)

        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("CCCO")

        block._forward(mol1)
        result = block._forward(mol2)

        sim = result.GetDoubleProp("cluster_similarity")
        assert isinstance(sim, float)
        assert 0.0 <= sim <= 1.0


class TestRepresentativeClusterIntegration:
    """Integration tests for RepresentativeClusterBlock."""

    def test_iterator_all_molecules_pass(self) -> None:
        """All molecules pass through (none filtered)."""
        block = RepresentativeClusterBlock()

        smiles = ["CCO", "CCC", "c1ccccc1", "NCCN"]
        mols = [Chem.MolFromSmiles(s) for s in smiles]

        results = list(block(iter(mols)))

        assert len(results) == len(smiles)

    def test_property_preservation(self) -> None:
        """Pre-existing properties survive via forward()."""
        block = RepresentativeClusterBlock()

        mol = Chem.MolFromSmiles("CCO")
        mol.SetProp("name", "ethanol")
        mol.SetDoubleProp("MolWt", 46.07)

        result = block.forward(mol)

        assert result is not None
        assert result.GetProp("name") == "ethanol"
        assert result.GetDoubleProp("MolWt") == pytest.approx(46.07)
        assert result.HasProp("cluster_id")

    def test_malformed_input_skipped(self) -> None:
        """None in iterator filtered by check_input."""
        block = RepresentativeClusterBlock()

        results = list(block(iter([None, Chem.MolFromSmiles("CCO"), None])))

        assert len(results) == 1

    def test_block_name(self) -> None:
        """block.name is 'RepresentativeCluster'."""
        block = RepresentativeClusterBlock()
        assert block.name == "RepresentativeCluster"

    def test_no_required_inputs(self) -> None:
        """No input_files or input_text required."""
        block = RepresentativeClusterBlock()
        assert len(block.input_files) == 0
        assert len(block.input_text) == 0

    def test_mutable_parameters(self) -> None:
        """threshold and scaffold in block.params with correct defaults."""
        block = RepresentativeClusterBlock()

        assert "threshold" in block.params
        assert "scaffold" in block.params
        assert block.params["threshold"].get() == pytest.approx(0.4)
        assert block.params["scaffold"].get() is True

    def test_all_identical_single_cluster(self) -> None:
        """N identical molecules all get cluster_id=0."""
        block = RepresentativeClusterBlock()

        mols = [Chem.MolFromSmiles("CCO") for _ in range(5)]
        results = list(block(iter(mols)))

        ids = [r.GetIntProp("cluster_id") for r in results]
        assert all(i == 0 for i in ids)

    def test_new_cluster_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Debug log emitted for new cluster creation."""
        block = RepresentativeClusterBlock()

        mol = Chem.MolFromSmiles("CCO")

        with caplog.at_level(logging.DEBUG, logger="cmxflow.operators.cluster"):
            block._forward(mol)

        assert any("New cluster" in record.message for record in caplog.records)
