"""Workflow serialization and registry utilities."""

import gzip
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from cmxflow.workflow import Workflow


def save_workflow(workflow: "Workflow", path: Path | str) -> None:
    """Save a workflow to a gzip-compressed pickle file.

    Args:
        workflow: The workflow to save.
        path: Path to save the workflow to.

    Raises:
        WorkflowValidationError: If the workflow fails validation.
    """
    from cmxflow.workflow import WorkflowValidationError

    try:
        workflow.check()
    except (IndexError, ValueError) as e:
        raise WorkflowValidationError(f"Cannot save invalid workflow: {e}") from e

    if isinstance(path, str):
        path = Path(path)

    with gzip.open(path, "wb") as f:
        pickle.dump(workflow, f)


def load_workflow(path: Path | str) -> "Workflow":
    """Load a workflow from a file.

    Supports gzip-compressed pickle files and legacy uncompressed pickle files.

    Args:
        path: Path to load the workflow from.

    Returns:
        The loaded workflow.

    Raises:
        WorkflowValidationError: If the loaded workflow fails validation.
        FileNotFoundError: If the file does not exist.
    """
    from cmxflow.workflow import Workflow, WorkflowValidationError

    if isinstance(path, str):
        path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"No such file: '{path}'")

    # Try gzip first, fall back to plain pickle for legacy files
    try:
        with gzip.open(path, "rb") as f:
            workflow: Workflow = pickle.load(f)
    except gzip.BadGzipFile:
        with open(path, "rb") as f:
            workflow = pickle.load(f)

    try:
        workflow.check()
    except (IndexError, ValueError) as e:
        raise WorkflowValidationError(f"Loaded workflow is invalid: {e}") from e

    return workflow


class WorkflowRegistry:
    """A registry for saving, listing, and loading named workflows.

    Workflows are stored as gzip-compressed pickle files in a directory,
    with a JSON metadata file tracking names, dates, and representations.

    Attributes:
        path: Directory where registry files are stored.
    """

    _METADATA_FILE = "registry.json"
    _COLUMNS = ["name", "date", "representation"]

    def __init__(self, path: Path | str = "~/.cmxflow/registry") -> None:
        """Initialize the workflow registry.

        Args:
            path: Directory to store registry files. Created if it doesn't exist.
        """
        self.path = Path(path).expanduser()
        self.path.mkdir(parents=True, exist_ok=True)

    @property
    def _metadata_path(self) -> Path:
        return self.path / self._METADATA_FILE

    def _read_metadata(self) -> list[dict[str, str]]:
        """Read the registry metadata file."""
        if not self._metadata_path.is_file():
            return []
        with open(self._metadata_path) as f:
            data = json.load(f)
            assert isinstance(data, list)
            if len(data) > 0:
                assert isinstance(data[0], dict)
            return data

    def _write_metadata(self, entries: list[dict[str, str]]) -> None:
        """Write the registry metadata file."""
        with open(self._metadata_path, "w") as f:
            json.dump(entries, f, indent=2)

    def register(
        self,
        name: str,
        workflow: "Workflow",
        overwrite: bool = False,
    ) -> None:
        """Register a workflow under a given name.

        Args:
            name: Name to register the workflow under.
            workflow: The workflow to register.
            overwrite: If True, overwrite an existing entry with the same name.

        Raises:
            ValueError: If name already exists and overwrite is False.
            WorkflowValidationError: If the workflow fails validation.
        """
        entries = self._read_metadata()
        existing = [e for e in entries if e["name"] == name]

        if existing and not overwrite:
            raise ValueError(
                f"Workflow '{name}' already exists. Use overwrite=True to replace."
            )

        # Save the workflow (validates internally)
        save_workflow(workflow, self.path / f"{name}.pkl.gz")

        # Build representation
        representation = " → ".join(b.name for b in workflow.blocks)

        entry = {
            "name": name,
            "date": datetime.now(timezone.utc).isoformat(),
            "representation": representation,
        }

        if existing:
            entries = [e if e["name"] != name else entry for e in entries]
        else:
            entries.append(entry)

        self._write_metadata(entries)

    def list(self) -> pd.DataFrame:
        """List all registered workflows.

        Returns:
            DataFrame with columns: name, date, representation.
        """
        entries = self._read_metadata()
        if not entries:
            return pd.DataFrame(columns=self._COLUMNS)
        return pd.DataFrame(entries, columns=self._COLUMNS)

    def load(self, name: str) -> "Workflow":
        """Load a registered workflow by name.

        Args:
            name: Name of the workflow to load.

        Returns:
            The loaded workflow.

        Raises:
            KeyError: If no workflow with the given name exists.
        """
        entries = self._read_metadata()
        if not any(e["name"] == name for e in entries):
            raise KeyError(f"No workflow registered with name '{name}'")
        return load_workflow(self.path / f"{name}.pkl.gz")

    def remove(self, name: str) -> None:
        """Remove a registered workflow.

        Args:
            name: Name of the workflow to remove.

        Raises:
            KeyError: If no workflow with the given name exists.
        """
        entries = self._read_metadata()
        if not any(e["name"] == name for e in entries):
            raise KeyError(f"No workflow registered with name '{name}'")

        # Remove the file
        pkl_path = self.path / f"{name}.pkl.gz"
        if pkl_path.is_file():
            pkl_path.unlink()

        # Update metadata
        entries = [e for e in entries if e["name"] != name]
        self._write_metadata(entries)
