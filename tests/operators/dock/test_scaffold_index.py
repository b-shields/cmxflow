"""Tests for scaffold keying and the SQLite scaffold pose store."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from cmxflow.operators.dock.scaffold_index import (
    ScaffoldPoseStore,
    scaffold_key,
    scaffold_pose,
)


def _embed(smiles: str, seed: int = 3) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    return Chem.RemoveAllHs(mol)


class TestScaffoldKey:

    def test_congeners_share_key(self) -> None:
        """Different substituents on the same ring system give the same key."""
        a = scaffold_key(Chem.MolFromSmiles("c1ccc(-c2ccccc2)cc1CCN"))
        b = scaffold_key(Chem.MolFromSmiles("c1ccc(-c2ccccc2)cc1CCO"))
        assert a is not None and a == b

    def test_acyclic_is_none(self) -> None:
        assert scaffold_key(Chem.MolFromSmiles("CCCCO")) is None

    def test_stereo_distinguishes_key(self) -> None:
        """E/Z on a scaffold linker (stilbene) yields distinct keys."""
        trans = scaffold_key(Chem.MolFromSmiles(r"c1ccccc1/C=C/c1ccccc1"))
        cis = scaffold_key(Chem.MolFromSmiles(r"c1ccccc1/C=C\c1ccccc1"))
        assert trans is not None and cis is not None
        assert trans != cis


class TestScaffoldPose:

    def test_posed_scaffold_is_substructure_with_coords(self) -> None:
        mol = _embed("c1ccc(-c2ccccc2)cc1CCN")  # biphenyl + tail
        posed = scaffold_pose(mol)
        assert posed is not None
        assert posed.GetNumConformers() == 1
        # The scaffold (biphenyl) is a substructure of the molecule.
        assert mol.HasSubstructMatch(posed)
        # Posed coords equal the molecule's coords on the matched scaffold atoms.
        match = mol.GetSubstructMatch(posed)
        mol_pos = np.array(mol.GetConformer().GetPositions())[list(match)]
        np.testing.assert_allclose(
            np.array(posed.GetConformer().GetPositions()), mol_pos, atol=1e-6
        )

    def test_acyclic_is_none(self) -> None:
        assert scaffold_pose(_embed("CCCCO")) is None


class TestScaffoldPoseStore:

    def test_put_get_roundtrip(self, tmp_path) -> None:
        store = ScaffoldPoseStore(tmp_path / "idx.db")
        scaffold = _embed("c1ccccc1-c1ccccc1")
        store.put("biphenyl", scaffold)
        got = store.get("biphenyl")
        assert got is not None
        assert Chem.MolToSmiles(got) == Chem.MolToSmiles(scaffold)
        np.testing.assert_allclose(
            np.array(got.GetConformer().GetPositions()),
            np.array(scaffold.GetConformer().GetPositions()),
            atol=1e-3,  # molblock writes 4-decimal coordinates
        )

    def test_get_missing_returns_none(self, tmp_path) -> None:
        store = ScaffoldPoseStore(tmp_path / "idx.db")
        assert store.get("absent") is None

    def test_first_writer_wins(self, tmp_path) -> None:
        store = ScaffoldPoseStore(tmp_path / "idx.db")
        first = _embed("c1ccccc1-c1ccccc1", seed=1)
        second = _embed("c1ccccc1-c1ccccc1", seed=99)  # different conformer
        store.put("k", first)
        store.put("k", second)  # ignored
        got = store.get("k")
        assert got is not None
        np.testing.assert_allclose(
            np.array(got.GetConformer().GetPositions()),
            np.array(first.GetConformer().GetPositions()),
            atol=1e-3,
        )

    def test_reuse_across_instances(self, tmp_path) -> None:
        """A second store instance on the same file sees prior writes (persistence)."""
        path = tmp_path / "idx.db"
        scaffold = _embed("c1ccccc1-c1ccccc1")
        ScaffoldPoseStore(path).put("biphenyl", scaffold)
        assert ScaffoldPoseStore(path).get("biphenyl") is not None

    def test_creates_parent_dir(self, tmp_path) -> None:
        """The store creates a missing parent dir (e.g. ./.cmxflow/) on first write."""
        path = tmp_path / ".cmxflow" / "scaffold_index.db"
        assert not path.parent.exists()
        ScaffoldPoseStore(path).put("k", _embed("c1ccccc1-c1ccccc1"))
        assert path.exists()
