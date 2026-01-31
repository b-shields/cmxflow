"""Utility functions for cmxflow."""

from cmxflow.utils.parallel import ParallelBlock, make_parallel, parallel

__all__ = ["parallel", "make_parallel", "ParallelBlock"]

try:
    from cmxflow.utils.pymol import open_pymol_session  # noqa: F401

    __all__.append("open_pymol_session")
except ImportError:
    pass  # pymol not installed
