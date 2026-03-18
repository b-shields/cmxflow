"""PyMOL session utilities for molecular visualization."""

from __future__ import annotations

import subprocess
from pathlib import Path


def open_pymol_session(
    *files: str | Path,
    show_backbone: bool = True,
    show_sidechains: bool = True,
    background_color: str = "black",
) -> subprocess.Popen[bytes]:
    """Open molecule files in a non-blocking PyMOL session.

    Launches PyMOL as a subprocess with the specified visualization
    settings. The function returns immediately, allowing the caller to
    continue working while PyMOL is open.

    Args:
        *files: One or more paths to molecule files (PDB, SDF, MOL2, etc.).
        show_backbone: If True, display protein backbone as cartoon.
        show_sidechains: If True, display side chains as sticks.
        background_color: PyMOL background color (e.g., "white", "black").

    Returns:
        The Popen object running PyMOL. Can be used to wait or terminate
        the session.

    Raises:
        FileNotFoundError: If any of the specified files do not exist.
        ValueError: If no files are provided.

    Example:
        >>> proc = open_pymol_session("protein.pdb", "ligand.sdf")
        >>> # Continue working while PyMOL is open
        >>> proc.wait()  # Optionally wait for PyMOL to close
    """
    if not files:
        raise ValueError("At least one file must be provided")

    validated_files: list[str] = []
    for file in files:
        path = Path(file)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file}")
        validated_files.append(str(path.resolve()))

    # Build PyMOL commands
    commands: list[str] = []
    if show_backbone:
        commands.append("show cartoon, all")
    if show_sidechains:
        commands.append("show lines, all")
    commands.append(f"bg_color {background_color}")
    commands.append("zoom all")

    # Launch PyMOL with files and -d flag for commands
    process = subprocess.Popen(
        ["pymol", *validated_files, "-d", "; ".join(commands)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return process
