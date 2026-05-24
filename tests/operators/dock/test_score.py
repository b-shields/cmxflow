"""Unit tests for Vinardo score components.

TDD: written against the intended API *before* implementation.
All tests fail until ScoreComponents and return_components are added to score.py.
"""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cmxflow.operators.dock.score import (
    AtomTyping,
    ScoreComponents,
    VinardoParams,
    vinardo_score_cached,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_mol_3d(smiles: str, seed: int = 42) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    return mol


def _make_protein_typing(
    n_atoms: int,
    hydrophobic: bool = False,
    donor: bool = False,
    acceptor: bool = False,
) -> AtomTyping:
    return AtomTyping(
        radii=np.full(n_atoms, 1.7),
        is_hydrophobic=np.full(n_atoms, hydrophobic),
        is_hbond_donor=np.full(n_atoms, donor),
        is_hbond_acceptor=np.full(n_atoms, acceptor),
    )


def _benzene_system():
    """Benzene ligand + 3 hydrophobic protein atoms ~3 Å away (surface dist ≈ 0)."""
    ligand = Chem.RemoveHs(_make_mol_3d("c1ccccc1"))
    protein_coords = np.array([[4.0, 0.0, 0.0], [4.0, 1.4, 0.0], [4.0, -1.4, 0.0]])
    protein_typing = _make_protein_typing(3, hydrophobic=True)
    return ligand, protein_coords, protein_typing


# =============================================================================
# TestVinardoScoreCachedComponents
# =============================================================================


class TestVinardoScoreCachedComponents:
    """Tests for the return_components kwarg on vinardo_score_cached."""

    def test_default_returns_float(self) -> None:
        """Unchanged call signature still returns a plain float."""
        ligand, pc, pt = _benzene_system()
        assert isinstance(vinardo_score_cached(ligand, pc, pt), float)

    def test_return_components_true_returns_tuple(self) -> None:
        """return_components=True returns (float, ScoreComponents)."""
        ligand, pc, pt = _benzene_system()
        result = vinardo_score_cached(ligand, pc, pt, return_components=True)
        assert isinstance(result, tuple) and len(result) == 2
        score, comps = result
        assert isinstance(score, float)
        assert isinstance(comps, ScoreComponents)

    def test_score_identical_with_and_without_components(self) -> None:
        """Returned score is identical with and without components."""
        ligand, pc, pt = _benzene_system()
        plain = vinardo_score_cached(ligand, pc, pt)
        with_comps, _ = vinardo_score_cached(ligand, pc, pt, return_components=True)
        assert plain == pytest.approx(with_comps)

    def test_components_total_matches_score(self) -> None:
        """comps.total must equal the returned score — the key consistency invariant."""
        ligand, pc, pt = _benzene_system()
        score, comps = vinardo_score_cached(ligand, pc, pt, return_components=True)
        assert comps.total == pytest.approx(score)

    def test_hydrophobic_raw_zero_when_no_hydrophobic_pairs(self) -> None:
        """hydrophobic_raw == 0 when protein has no hydrophobic atoms."""
        ligand, pc, _ = _benzene_system()
        _, comps = vinardo_score_cached(
            ligand,
            pc,
            _make_protein_typing(3, hydrophobic=False),
            return_components=True,
        )
        assert comps.hydrophobic_raw == 0.0

    def test_hbond_raw_zero_when_no_donor_acceptor_pairs(self) -> None:
        """hbond_raw == 0 when neither molecule has donors or acceptors."""
        ligand, pc, _ = _benzene_system()
        _, comps = vinardo_score_cached(
            ligand, pc, _make_protein_typing(3), return_components=True
        )
        assert comps.hbond_raw == 0.0

    def test_hydrophobic_raw_nonzero_when_close_hydrophobic_pairs(self) -> None:
        """Sanity: hydrophobic_raw > 0 when atoms are within the good cutoff."""
        ligand, pc, pt = _benzene_system()
        _, comps = vinardo_score_cached(ligand, pc, pt, return_components=True)
        assert comps.hydrophobic_raw > 0.0

    def test_weights_in_components_match_params(self) -> None:
        """Weights stored in ScoreComponents must match the VinardoParams passed in."""
        ligand, pc, pt = _benzene_system()
        params = VinardoParams(
            w_gauss1=-0.050, w_repulsion=0.900, w_hydrophobic=-0.040, w_hbond=-0.700
        )
        _, comps = vinardo_score_cached(
            ligand, pc, pt, params=params, return_components=True
        )
        assert comps.w_gauss1 == pytest.approx(-0.050)
        assert comps.w_repulsion == pytest.approx(0.900)
        assert comps.w_hydrophobic == pytest.approx(-0.040)
        assert comps.w_hbond == pytest.approx(-0.700)
