"""cmxflow: Automated cheminformatics workflow optimization."""

from cmxflow.block import Block, BlockBase, SinkBlock, SourceBlock
from cmxflow.cmxmol import Mol, unwrap_mol, wrap_mol
from cmxflow.workflow import (
    Workflow,
    WorkflowValidationError,
    load_workflow,
    save_workflow,
)

__version__ = "0.1.0"
__all__ = [
    "Block",
    "BlockBase",
    "Mol",
    "SinkBlock",
    "SourceBlock",
    "Workflow",
    "WorkflowValidationError",
    "load_workflow",
    "save_workflow",
    "unwrap_mol",
    "wrap_mol",
]
