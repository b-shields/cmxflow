"""cmxflow: Automated cheminformatics workflow optimization."""

from cmxflow.block import Block, BlockBase, SinkBlock, SourceBlock
from cmxflow.cmxmol import Mol, unwrap_mol, wrap_mol
from cmxflow.workflow import Workflow

__version__ = "0.1.0"
__all__ = [
    "Block",
    "BlockBase",
    "Mol",
    "SinkBlock",
    "SourceBlock",
    "Workflow",
    "unwrap_mol",
    "wrap_mol",
]
