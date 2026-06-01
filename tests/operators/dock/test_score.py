"""Unit tests for empirical score components."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cmxflow.operators.dock.score import (
    AtomTyping,
    EmpiricalParams,
    ScoreComponents,
    empirical_score_cached,
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
    """Heavy-atom benzene ligand + 3 hydrophobic protein atoms ~3 Å away."""
    ligand = Chem.RemoveAllHs(_make_mol_3d("c1ccccc1"))
    protein_coords = np.array([[4.0, 0.0, 0.0], [4.0, 1.4, 0.0], [4.0, -1.4, 0.0]])
    protein_typing = _make_protein_typing(3, hydrophobic=True)
    return ligand, protein_coords, protein_typing


# =============================================================================
# TestEmpiricalScoreCachedComponents
# =============================================================================


class TestEmpiricalScoreCachedComponents:
    """Tests for empirical_score_cached which always returns ScoreComponents."""

    def test_returns_score_components(self) -> None:
        """empirical_score_cached always returns ScoreComponents."""
        ligand, pc, pt = _benzene_system()
        result = empirical_score_cached(ligand, pc, pt)
        assert isinstance(result, ScoreComponents)

    def test_total_is_float(self) -> None:
        """comps.total is a float."""
        ligand, pc, pt = _benzene_system()
        comps = empirical_score_cached(ligand, pc, pt)
        assert isinstance(comps.total, float)

    def test_components_sum_to_total(self) -> None:
        """comps.total must equal the sum of component terms."""
        ligand, pc, pt = _benzene_system()
        comps = empirical_score_cached(ligand, pc, pt)
        reconstructed = comps.gauss1 + comps.repulsion + comps.hydrophobic + comps.hbond
        assert reconstructed == pytest.approx(comps.total)

    def test_hydrophobic_raw_zero_when_no_hydrophobic_pairs(self) -> None:
        """hydrophobic_raw == 0 when protein has no hydrophobic atoms."""
        ligand, pc, _ = _benzene_system()
        comps = empirical_score_cached(
            ligand, pc, _make_protein_typing(3, hydrophobic=False)
        )
        assert comps.hydrophobic_raw == 0.0

    def test_hbond_raw_zero_when_no_donor_acceptor_pairs(self) -> None:
        """hbond_raw == 0 when neither molecule has donors or acceptors."""
        ligand, pc, _ = _benzene_system()
        comps = empirical_score_cached(ligand, pc, _make_protein_typing(3))
        assert comps.hbond_raw == 0.0

    def test_hydrophobic_raw_nonzero_when_close_hydrophobic_pairs(self) -> None:
        """Sanity: hydrophobic_raw > 0 when atoms are within the good cutoff."""
        ligand, pc, pt = _benzene_system()
        comps = empirical_score_cached(ligand, pc, pt)
        assert comps.hydrophobic_raw > 0.0

    def test_weights_in_components_match_params(self) -> None:
        """Weights in ScoreComponents must match the EmpiricalParams passed in."""
        ligand, pc, pt = _benzene_system()
        params = EmpiricalParams(
            w_gauss1=-0.050, w_repulsion=0.900, w_hydrophobic=-0.040, w_hbond=-0.700
        )
        comps = empirical_score_cached(ligand, pc, pt, params=params)
        assert comps.w_gauss1 == pytest.approx(-0.050)
        assert comps.w_repulsion == pytest.approx(0.900)
        assert comps.w_hydrophobic == pytest.approx(-0.040)
        assert comps.w_hbond == pytest.approx(-0.700)
