"""Tests for automatic scoring blocks."""

import numpy as np
import pytest
from rdkit import Chem

from cmxflow.scores.automatic import (
    AverageScoreBlock,
    EnrichmentScoreBlock,
    enrichment_auc,
    mol_to_dataframe,
)


def _create_mol_with_props(smiles: str, props: dict[str, float]) -> Chem.Mol:
    """Create a molecule with specified properties."""
    mol = Chem.MolFromSmiles(smiles)
    for key, value in props.items():
        mol.SetDoubleProp(key, value)
    return mol


class TestMolToDataframe:
    """Tests for mol_to_dataframe utility function."""

    def test_converts_molecules_to_dataframe(self) -> None:
        """Test basic conversion of molecules with properties."""
        mol1 = _create_mol_with_props("CCO", {"score": 0.5, "mw": 46.07})
        mol2 = _create_mol_with_props("CO", {"score": 0.8, "mw": 32.04})

        df = mol_to_dataframe(iter([mol1, mol2]))

        assert len(df) == 2
        assert "score" in df.columns
        assert "mw" in df.columns
        assert df["score"].tolist() == [0.5, 0.8]

    def test_returns_empty_dataframe_for_empty_input(self) -> None:
        """Test that empty input returns empty DataFrame."""
        df = mol_to_dataframe(iter([]))
        assert df.empty

    def test_skips_none_molecules(self) -> None:
        """Test that None molecules are skipped."""
        mol = _create_mol_with_props("CCO", {"score": 0.5})
        df = mol_to_dataframe(iter([None, mol, None]))
        assert len(df) == 1


class TestAverageScoreBlockInit:
    """Tests for AverageScoreBlock initialization."""

    def test_init_creates_property_input(self) -> None:
        """Test that initialization creates a property input text requirement."""
        block = AverageScoreBlock()
        assert "property" in block.input_text


class TestAverageScoreBlockObjective:
    """Tests for AverageScoreBlock.objective method."""

    def test_computes_average_of_property(self) -> None:
        """Test that objective computes the mean of the specified property."""
        mol1 = _create_mol_with_props("CCO", {"docking_score": 10.0})
        mol2 = _create_mol_with_props("CO", {"docking_score": 20.0})
        mol3 = _create_mol_with_props("C", {"docking_score": 30.0})

        block = AverageScoreBlock()
        block.input_text["property"] = "docking_score"

        score = block.objective(iter([mol1, mol2, mol3]))

        assert score == 20.0

    def test_returns_zero_for_empty_input(self) -> None:
        """Test that empty input returns 0.0."""
        block = AverageScoreBlock()
        block.input_text["property"] = "score"

        score = block.objective(iter([]))

        assert score == 0.0

    def test_raises_error_for_missing_property(self) -> None:
        """Test that missing property column raises ValueError."""
        mol = _create_mol_with_props("CCO", {"other_prop": 1.0})

        block = AverageScoreBlock()
        block.input_text["property"] = "missing_property"

        with pytest.raises(
            ValueError, match="Property column 'missing_property' not found"
        ):
            block.objective(iter([mol]))

    def test_handles_single_molecule(self) -> None:
        """Test average with a single molecule."""
        mol = _create_mol_with_props("CCO", {"value": 42.0})

        block = AverageScoreBlock()
        block.input_text["property"] = "value"

        score = block.objective(iter([mol]))

        assert score == 42.0


class TestAverageScoreBlockForward:
    """Tests for AverageScoreBlock.forward method."""

    def test_forward_returns_molecule_unchanged(self) -> None:
        """Test that forward returns the input molecule unchanged."""
        mol = _create_mol_with_props("CCO", {"score": 0.5})

        block = AverageScoreBlock()
        result = block.forward(mol)

        assert result is mol
        assert result.GetDoubleProp("score") == 0.5


class TestEnrichmentAuc:
    """Tests for enrichment_auc utility function."""

    def test_perfect_ranking(self) -> None:
        """Test high AUC for perfect ranking (all hits ranked first)."""
        scores = np.array([1.0, 0.9, 0.8, 0.2, 0.1])
        labels = np.array([1, 1, 1, 0, 0])

        auc = enrichment_auc(scores, labels)

        # Perfect ranking should give high AUC (trapezoidal integration)
        assert auc > 0.6

    def test_random_ranking(self) -> None:
        """Test AUC close to 0.5 for random ranking."""
        scores = np.array([0.5, 0.4, 0.6, 0.3, 0.7])
        labels = np.array([1, 0, 1, 0, 0])

        auc = enrichment_auc(scores, labels)

        # Random ranking should be around 0.5
        assert 0.3 < auc < 0.7

    def test_worst_ranking(self) -> None:
        """Test low AUC for worst ranking (all hits ranked last)."""
        scores = np.array([0.1, 0.2, 0.8, 0.9, 1.0])
        labels = np.array([1, 1, 0, 0, 0])

        auc = enrichment_auc(scores, labels)

        # Worst ranking should be close to 0
        assert auc < 0.3

    def test_empty_input(self) -> None:
        """Test that empty input returns 0.0."""
        auc = enrichment_auc(np.array([]), np.array([]))
        assert auc == 0.0

    def test_no_hits(self) -> None:
        """Test that no hits returns 0.0."""
        scores = np.array([1.0, 0.5, 0.2])
        labels = np.array([0, 0, 0])

        auc = enrichment_auc(scores, labels)

        assert auc == 0.0


class TestEnrichmentScoreBlockInit:
    """Tests for EnrichmentScoreBlock initialization."""

    def test_init_creates_target_input(self) -> None:
        """Test that initialization creates a target input text requirement."""
        block = EnrichmentScoreBlock()
        assert "target" in block.input_text


class TestEnrichmentScoreBlockObjective:
    """Tests for EnrichmentScoreBlock.objective method."""

    def test_computes_enrichment_auc(self) -> None:
        """Test that objective computes enrichment AUC correctly."""
        # Create molecules with scores and target labels
        mol1 = _create_mol_with_props("CCO", {"score": 1.0, "active": 1.0})
        mol2 = _create_mol_with_props("CO", {"score": 0.8, "active": 1.0})
        mol3 = _create_mol_with_props("C", {"score": 0.2, "active": 0.0})

        block = EnrichmentScoreBlock()
        block.input_text["target"] = "active"
        block._score_properties = {"score": False}

        auc = block.objective(iter([mol1, mol2, mol3]))

        # Perfect ranking should give high AUC
        assert auc > 0.5

    def test_returns_zero_for_empty_input(self) -> None:
        """Test that empty input returns 0.0."""
        block = EnrichmentScoreBlock()
        block.input_text["target"] = "active"
        block._score_properties = {"score": False}

        auc = block.objective(iter([]))

        assert auc == 0.0

    def test_raises_error_for_missing_target_column(self) -> None:
        """Test that missing target column raises an error."""
        mol = _create_mol_with_props("CCO", {"score": 1.0})

        block = EnrichmentScoreBlock()
        block.input_text["target"] = "missing_target"

        # KeyError raised when pandas tries to drop the missing column
        with pytest.raises(KeyError):
            block.objective(iter([mol]))

    def test_dynamically_sets_best_score_name(self) -> None:
        """Test that best score name is set after objective call."""
        mol1 = _create_mol_with_props("CCO", {"docking": 1.0, "active": 1.0})
        mol2 = _create_mol_with_props("CO", {"docking": 0.5, "active": 0.0})

        block = EnrichmentScoreBlock()
        block.input_text["target"] = "active"
        block._score_properties = {"docking": False}

        # Before objective call, no best score
        assert block._best_score_name is None

        block.objective(iter([mol1, mol2]))

        # After objective call, best score name should be set
        assert block._best_score_name == "docking"


class TestEnrichmentScoreBlockForward:
    """Tests for EnrichmentScoreBlock.forward method."""

    def test_forward_returns_molecule_when_no_params(self) -> None:
        """Test that forward returns molecule unchanged when no params set."""
        mol = _create_mol_with_props("CCO", {"score": 0.5})

        block = EnrichmentScoreBlock()
        result = block.forward(mol)

        assert result is mol

    def test_forward_annotates_workflow_score(self) -> None:
        """Test that forward annotates molecule with workflow_score."""
        mol1 = _create_mol_with_props("CCO", {"docking": 0.8, "active": 1.0})
        mol2 = _create_mol_with_props("CO", {"docking": 0.2, "active": 0.0})

        block = EnrichmentScoreBlock()
        block.input_text["target"] = "active"
        block._score_properties = {"docking": False}

        # Trigger best score name setup
        block.objective(iter([mol1, mol2]))

        # Now test forward
        test_mol = _create_mol_with_props("C", {"docking": 0.6})
        result = block.forward(test_mol)

        assert result.HasProp("workflow_score")
        assert result.GetDoubleProp("workflow_score") == 0.6
