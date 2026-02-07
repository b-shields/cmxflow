"""Tests for cluster quality scoring block."""

import pytest
from rdkit import Chem

from cmxflow.scores.cluster import ClusterScoreBlock


def _create_clustered_mol(smiles: str, cluster_id: int, similarity: float) -> Chem.Mol:
    """Create a molecule with cluster annotations."""
    mol = Chem.MolFromSmiles(smiles)
    mol.SetIntProp("cluster_id", cluster_id)
    mol.SetDoubleProp("cluster_similarity", similarity)
    return mol


class TestClusterScoreObjective:
    """Tests for ClusterScoreBlock.objective method."""

    def test_single_cluster_all_identical(self) -> None:
        """All molecules in one cluster with high similarity scores ~1.0."""
        mols = [
            _create_clustered_mol("CCO", 0, 1.0),
            _create_clustered_mol("CCO", 0, 1.0),
            _create_clustered_mol("CCO", 0, 1.0),
        ]
        block = ClusterScoreBlock()
        score = block.objective(iter(mols))
        assert score == pytest.approx(1.0)

    def test_all_singletons_returns_zero(self) -> None:
        """Every molecule in its own cluster returns 0.0."""
        mols = [
            _create_clustered_mol("CCO", 0, 1.0),
            _create_clustered_mol("CO", 1, 1.0),
            _create_clustered_mol("C", 2, 1.0),
        ]
        block = ClusterScoreBlock()
        score = block.objective(iter(mols))
        assert score == 0.0

    def test_empty_input_returns_zero(self) -> None:
        """No molecules returns 0.0."""
        block = ClusterScoreBlock()
        score = block.objective(iter([]))
        assert score == 0.0

    def test_mixed_clusters_and_singletons(self) -> None:
        """Verify formula: mean_similarity - n_single/n_molecules."""
        # Cluster 0: 3 molecules with similarities 0.8, 0.9, 1.0
        # Cluster 1: 1 molecule (singleton)
        # Cluster 2: 1 molecule (singleton)
        mols = [
            _create_clustered_mol("CCO", 0, 0.8),
            _create_clustered_mol("CCCO", 0, 0.9),
            _create_clustered_mol("CCCCO", 0, 1.0),
            _create_clustered_mol("CO", 1, 1.0),
            _create_clustered_mol("C", 2, 1.0),
        ]
        block = ClusterScoreBlock()
        score = block.objective(iter(mols))

        # mean_similarity = (0.8 + 0.9 + 1.0) / 3 = 0.9
        # singleton_penalty = 2 / 5 = 0.4
        # score = 0.9 - 0.4 = 0.5
        assert score == pytest.approx(0.5)

    def test_no_singletons_score_equals_mean_similarity(self) -> None:
        """Without singletons, score equals mean similarity."""
        mols = [
            _create_clustered_mol("CCO", 0, 0.8),
            _create_clustered_mol("CCCO", 0, 0.9),
            _create_clustered_mol("CO", 1, 0.7),
            _create_clustered_mol("COC", 1, 0.6),
        ]
        block = ClusterScoreBlock()
        score = block.objective(iter(mols))

        mean_sim = (0.8 + 0.9 + 0.7 + 0.6) / 4
        assert score == pytest.approx(mean_sim)

    def test_score_range(self) -> None:
        """Score is in [-1, 1] for various inputs."""
        scenarios = [
            # All high similarity, no singletons
            [
                _create_clustered_mol("CCO", 0, 0.95),
                _create_clustered_mol("CO", 0, 0.90),
            ],
            # Low similarity, many singletons
            [
                _create_clustered_mol("CCO", 0, 0.3),
                _create_clustered_mol("CO", 0, 0.2),
                _create_clustered_mol("C", 1, 1.0),
                _create_clustered_mol("CC", 2, 1.0),
                _create_clustered_mol("CCC", 3, 1.0),
            ],
            # All singletons (score = 0.0)
            [_create_clustered_mol("CCO", 0, 1.0)],
        ]
        block = ClusterScoreBlock()
        for mols in scenarios:
            score = block.objective(iter(mols))
            assert -1.0 <= score <= 1.0

    def test_higher_similarity_gives_higher_score(self) -> None:
        """Clusters with higher similarity produce higher scores."""
        mols_high = [
            _create_clustered_mol("CCO", 0, 0.95),
            _create_clustered_mol("CO", 0, 0.90),
        ]
        mols_low = [
            _create_clustered_mol("CCO", 0, 0.4),
            _create_clustered_mol("CO", 0, 0.3),
        ]
        block = ClusterScoreBlock()
        score_high = block.objective(iter(mols_high))
        score_low = block.objective(iter(mols_low))
        assert score_high > score_low


class TestClusterScoreIntegration:
    """Integration tests for ClusterScoreBlock."""

    def test_forward_passthrough(self) -> None:
        """forward() returns molecule unchanged."""
        mol = _create_clustered_mol("CCO", 0, 0.9)
        block = ClusterScoreBlock()
        result = block.forward(mol)
        assert result is mol

    def test_block_name(self) -> None:
        """Block name is ClusterScore."""
        block = ClusterScoreBlock()
        assert block.name == "ClusterScore"

    def test_no_required_inputs(self) -> None:
        """No input_files or input_text required."""
        block = ClusterScoreBlock()
        assert not block.input_files
        assert not block.input_text

    def test_no_mutable_parameters(self) -> None:
        """Block has no mutable parameters."""
        block = ClusterScoreBlock()
        assert not block.params
