"""Test pickle serialization for all available blocks."""

import pickle

import pytest

from cmxflow.mcp.state import get_available_blocks


def _instantiate_block(name: str, cls: type) -> object:
    """Instantiate a block with minimal valid arguments."""
    if name == "RDKitBlock":
        return cls("rdkit.Chem.Descriptors.MolWt")
    return cls()


@pytest.mark.parametrize(
    "block_name,block_cls",
    get_available_blocks().items(),
    ids=get_available_blocks().keys(),
)
def test_block_pickleable(block_name: str, block_cls: type) -> None:
    """Every block exposed by get_available_blocks must survive a pickle round-trip."""
    original = _instantiate_block(block_name, block_cls)
    data = pickle.dumps(original)
    loaded = pickle.loads(data)  # noqa: S301
    assert isinstance(loaded, type(original))
