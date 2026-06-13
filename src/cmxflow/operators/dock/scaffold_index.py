"""Scaffold pose index for template-accelerated docking.

Keys molecules by their (stereo-aware) Bemis-Murcko scaffold and persists one
docked pose per unique scaffold in a single-file SQLite store. When a later
molecule shares a stored scaffold, its pose can be transferred from the stored
template and refined with a single constrained local search (see
:mod:`cmxflow.operators.dock.template`) instead of a full search.

The store is a single SQLite file (one row per scaffold), safe for concurrent
worker processes: each process opens its own connection, and writes use
``INSERT OR IGNORE`` so the first writer of a scaffold wins without locking.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)


def scaffold_key(mol: Chem.Mol) -> str | None:
    """Return the canonical isomeric Bemis-Murcko scaffold SMILES for ``mol``.

    Stereochemistry on scaffold atoms is preserved (isomeric SMILES), so e.g. E/Z
    linkers and scaffold stereocentres map to distinct keys.

    Args:
        mol: Input molecule.

    Returns:
        Canonical isomeric scaffold SMILES, or ``None`` for acyclic molecules
        (empty Murcko scaffold) which have no ring scaffold to template on.
    """
    scaffold = MurckoScaffold.GetScaffoldForMol(Chem.RemoveAllHs(mol))
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(scaffold)


def scaffold_pose(mol: Chem.Mol) -> Chem.Mol | None:
    """Return the Bemis-Murcko scaffold of ``mol`` carrying ``mol``'s 3D coords.

    This is the posed scaffold stored as a template: a substructure of any
    molecule sharing the same :func:`scaffold_key`, with the donor molecule's
    coordinates.

    Args:
        mol: Molecule with a 3D conformer.

    Returns:
        The scaffold Mol with coordinates, or ``None`` if acyclic / unmatched.
    """
    heavy = Chem.RemoveAllHs(mol)
    scaffold = MurckoScaffold.GetScaffoldForMol(heavy)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return None
    # scaffold atom i <-> heavy atom match[i]. GetScaffoldForMol returns a sanitized
    # mol (kekulizable); attaching the donor's coordinates to it avoids raw atom
    # surgery, which can leave an aromatic fragment that won't round-trip a molblock.
    match = heavy.GetSubstructMatch(scaffold)
    if not match:
        return None
    heavy_conf = heavy.GetConformer()
    conf = Chem.Conformer(scaffold.GetNumAtoms())
    for i, h in enumerate(match):
        conf.SetAtomPosition(i, heavy_conf.GetAtomPosition(int(h)))
    scaffold = Chem.Mol(scaffold)
    scaffold.RemoveAllConformers()
    scaffold.AddConformer(conf, assignId=True)
    return scaffold


class ScaffoldPoseStore:
    """Single-file SQLite store mapping scaffold key -> a posed scaffold molblock.

    The connection is opened lazily on first use and held per instance (and thus
    per process). Never share an instance across processes; create one per worker.
    """

    def __init__(self, path: str | Path) -> None:
        """Initialize the store.

        Args:
            path: Path to the SQLite database file. Parent directories are
                created on first write.
        """
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.path), timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS scaffolds "
                "(key TEXT PRIMARY KEY, molblock TEXT NOT NULL)"
            )
            conn.commit()
            self._conn = conn
        return self._conn

    def get(self, key: str) -> Chem.Mol | None:
        """Return the stored posed scaffold for ``key``, or ``None`` if absent."""
        cur = self._connect().execute(
            "SELECT molblock FROM scaffolds WHERE key = ?", (key,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        return Chem.MolFromMolBlock(row[0])

    def put(self, key: str, scaffold_pose: Chem.Mol) -> None:
        """Store ``scaffold_pose`` (a scaffold Mol with a 3D conformer) under ``key``.

        First-writer-wins: an existing key is left unchanged (``INSERT OR IGNORE``).
        """
        molblock = Chem.MolToMolBlock(scaffold_pose)
        conn = self._connect()
        conn.execute(
            "INSERT OR IGNORE INTO scaffolds (key, molblock) VALUES (?, ?)",
            (key, molblock),
        )
        conn.commit()

    def close(self) -> None:
        """Close the underlying connection if open."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
