"""Integration tests for score_components kwarg on MoleculeDockBlock.

TDD: written against the intended API *before* implementation.
All tests fail until score_components is wired into dock.py.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cmxflow.operators.dock.score import AtomTyping

COMPONENT_TAGS = [
    "docking_gauss1_raw",
    "docking_repulsion_raw",
    "docking_hydrophobic_raw",
    "docking_hbond_raw",
    "docking_gauss1",
    "docking_repulsion",
    "docking_hydrophobic",
    "docking_hbond",
]


def _make_block(score_components: bool = True):
    from cmxflow.operators.dock import MoleculeDockBlock

    block = MoleculeDockBlock(score_components=score_components)
    block._protein_coords = np.zeros((3, 3))
    block._protein_typing = AtomTyping(
        radii=np.full(3, 1.7),
        is_hydrophobic=np.zeros(3, dtype=bool),
        is_hbond_donor=np.zeros(3, dtype=bool),
        is_hbond_acceptor=np.zeros(3, dtype=bool),
    )
    block._protein_ec_coords = np.zeros((3, 3))
    block._protein_ec_charges = np.zeros(3)
    return block


def _make_mol() -> Chem.Mol:
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    return mol


def _mock_result(mol: Chem.Mol) -> MagicMock:
    r = MagicMock()
    r.mol = mol
    r.score = -5.0
    r.initial_score = -3.0
    r.converged = True
    r.ec = 0.0
    return r


class TestMoleculeDockBlockScoreComponents:

    def test_default_is_true(self) -> None:
        from cmxflow.operators.dock import MoleculeDockBlock

        assert MoleculeDockBlock()._score_components is True

    def test_false_stored(self) -> None:
        from cmxflow.operators.dock import MoleculeDockBlock

        assert MoleculeDockBlock(score_components=False)._score_components is False

    def test_true_writes_all_component_tags(self) -> None:
        block = _make_block(score_components=True)
        mol = _make_mol()
        with patch(
            "cmxflow.operators.dock.dock.optimize_pose_cached",
            return_value=_mock_result(mol),
        ):
            result = block._forward(mol)
        assert result is not None
        for tag in COMPONENT_TAGS:
            assert result.HasProp(tag), f"Missing tag: {tag}"

    def test_false_writes_no_component_tags(self) -> None:
        block = _make_block(score_components=False)
        mol = _make_mol()
        with patch(
            "cmxflow.operators.dock.dock.optimize_pose_cached",
            return_value=_mock_result(mol),
        ):
            result = block._forward(mol)
        assert result is not None
        for tag in COMPONENT_TAGS:
            assert not result.HasProp(tag), f"Unexpected tag: {tag}"

    def test_weighted_equals_w_times_raw(self) -> None:
        """docking_gauss1 == w_gauss1 * docking_gauss1_raw (likewise for all terms)."""
        block = _make_block(score_components=True)
        mol = _make_mol()
        with patch(
            "cmxflow.operators.dock.dock.optimize_pose_cached",
            return_value=_mock_result(mol),
        ):
            result = block._forward(mol)
        assert result is not None
        for raw_tag, wtd_tag, param in [
            ("docking_gauss1_raw", "docking_gauss1", "w_gauss1"),
            ("docking_repulsion_raw", "docking_repulsion", "w_repulsion"),
            ("docking_hydrophobic_raw", "docking_hydrophobic", "w_hydrophobic"),
            ("docking_hbond_raw", "docking_hbond", "w_hbond"),
        ]:
            w = block.get_param(param)
            raw = result.GetDoubleProp(raw_tag)
            wtd = result.GetDoubleProp(wtd_tag)
            assert wtd == pytest.approx(w * raw), f"{wtd_tag} != {param} * {raw_tag}"

    def test_check_output_does_not_require_component_tags(self) -> None:
        """check_output must pass without component tags present."""
        from cmxflow.operators.dock import MoleculeDockBlock

        block = MoleculeDockBlock()
        mol = _make_mol()
        mol.SetDoubleProp("docking_initial_pose_score", -3.0)
        mol.SetDoubleProp("docking_score", -5.0)
        mol.SetDoubleProp("docking_ec", 0.0)
        mol.SetBoolProp("docking_converged", True)
        assert block.check_output(mol) is True
