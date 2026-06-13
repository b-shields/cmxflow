"""Tests for the template docking primitives.

Covers template-pose transfer (substructure overlay + core-held relax) and the
composed constrained template dock against a synthetic protein. Cores are built
the way the real pipeline does -- the Bemis-Murcko scaffold pose -- and matched
with a fast ``GetSubstructMatch`` (no MCS search).
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from cmxflow.operators.dock.scaffold_index import scaffold_pose
from cmxflow.operators.dock.score import EmpiricalParams, get_atom_typing
from cmxflow.operators.dock.template import template_dock, transfer_template_pose

# A congeneric pair sharing a biphenyl scaffold, differing only in the tail.
REF_SMILES = "c1ccc(-c2ccccc2)cc1CCN"
QUERY_SMILES = "c1ccc(-c2ccccc2)cc1CCO"


def _embed(smiles: str, seed: int = 7) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    return mol


def _biphenyl_core() -> Chem.Mol:
    """The shared scaffold pose (biphenyl) taken from the reference ligand."""
    core = scaffold_pose(_embed(REF_SMILES))
    assert core is not None and core.GetNumAtoms() == 12
    return core


class TestTransferTemplatePose:

    def test_overlays_core_on_template(self) -> None:
        core = _biphenyl_core()
        result = transfer_template_pose(_embed(QUERY_SMILES), core)
        assert result is not None
        prepared, idx = result
        # Matched atoms land on the template core (held fixed during relax).
        prepared_heavy = Chem.RemoveAllHs(prepared)
        match = prepared_heavy.GetSubstructMatch(core)
        pos = np.array(prepared_heavy.GetConformer().GetPositions())[list(match)]
        core_pos = np.array(core.GetConformer().GetPositions())
        assert np.linalg.norm(pos - core_pos, axis=1).max() < 1e-3
        # Returned indices are exactly the matched core atoms.
        assert set(idx) == set(match)
        assert len(idx) == core.GetNumAtoms()

    def test_no_match_returns_none(self) -> None:
        query = _embed(QUERY_SMILES)
        unrelated_core = Chem.RemoveAllHs(_embed("C1CCNCC1"))  # piperidine, absent
        assert transfer_template_pose(query, unrelated_core) is None

    def test_falls_back_without_ff_params(self, monkeypatch) -> None:
        """No MMFF/UFF params -> returns the snapped pose without crashing."""
        core = _biphenyl_core()
        monkeypatch.setattr(AllChem, "MMFFHasAllMoleculeParams", lambda _m: False)
        monkeypatch.setattr(AllChem, "UFFHasAllMoleculeParams", lambda _m: False)
        result = transfer_template_pose(_embed(QUERY_SMILES), core)
        assert result is not None
        prepared, _ = result
        prepared_heavy = Chem.RemoveAllHs(prepared)
        match = prepared_heavy.GetSubstructMatch(core)
        pos = np.array(prepared_heavy.GetConformer().GetPositions())[list(match)]
        core_pos = np.array(core.GetConformer().GetPositions())
        np.testing.assert_allclose(pos, core_pos, atol=1e-6)


class TestTemplateDock:

    def _synthetic_protein(self, ligand: Chem.Mol):
        """A protein shell offset from the ligand so it doesn't overlap."""
        heavy = Chem.RemoveAllHs(ligand)
        coords = np.array(heavy.GetConformer().GetPositions()) + np.array(
            [6.0, 0.0, 0.0]
        )
        return coords, get_atom_typing(heavy)

    def test_constrained_dock_holds_core(self) -> None:
        core = _biphenyl_core()
        query = _embed(QUERY_SMILES)
        protein_coords, protein_typing = self._synthetic_protein(query)
        result = template_dock(
            query,
            core,
            protein_coords,
            protein_typing,
            constraint_weight=50.0,
            constraint_tol=0.5,
            score_params=EmpiricalParams(),
        )
        assert result is not None
        out_heavy = Chem.RemoveAllHs(result.mol)
        match = out_heavy.GetSubstructMatch(core)
        pos = np.array(out_heavy.GetConformer().GetPositions())[list(match)]
        core_pos = np.array(core.GetConformer().GetPositions())
        assert np.linalg.norm(pos - core_pos, axis=1).max() < 1.0
        assert np.isfinite(result.score)

    def test_no_match_returns_none(self) -> None:
        query = _embed(QUERY_SMILES)
        protein_coords, protein_typing = self._synthetic_protein(query)
        unrelated_core = Chem.RemoveAllHs(_embed("C1CCNCC1"))
        assert (
            template_dock(
                query,
                unrelated_core,
                protein_coords,
                protein_typing,
                constraint_weight=50.0,
            )
            is None
        )
