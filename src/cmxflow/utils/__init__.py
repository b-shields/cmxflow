"""Utility functions for cmxflow."""

from cmxflow.utils.parallel import ParallelBlock, make_parallel, parallel
from cmxflow.utils.serial import WorkflowRegistry, load_workflow, save_workflow

__all__ = [
    "ParallelBlock",
    "WorkflowRegistry",
    "load_workflow",
    "make_parallel",
    "parallel",
    "save_workflow",
]

try:
    from cmxflow.utils.pymol import open_pymol_session  # noqa: F401

    __all__.append("open_pymol_session")
except ImportError:
    pass  # pymol not installed
