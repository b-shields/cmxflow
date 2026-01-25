"""Workflow class for dynamically building and executing block pipelines."""

from pathlib import Path
from typing import Any

from cmxflow.block import Block, BlockBase, SinkBlock, SourceBlock
from cmxflow.parameter import Parameter
from cmxflow.utils import text


class Workflow:
    """A workflow that chains multiple blocks together.

    Workflows allow dynamic composition of blocks into executable pipelines.
    Blocks are executed sequentially, with the output of each block passed
    as input to the next.

    Attributes:
        name: Human-readable name for the workflow.
        blocks: List of blocks in execution order.
    """

    def __init__(self, name: str = "Workflow") -> None:
        """Initialize an empty workflow.

        Args:
            name: Optional name for the workflow.
        """
        self.name = name
        self.blocks: list[BlockBase] = []

    def add(self, *blocks: BlockBase) -> "Workflow":
        """Add a block to the end of the workflow.

        Args:
            block: The block to add.

        Returns:
            Self for method chaining.
        """
        for block in blocks:
            self.blocks.append(block)
        return self

    def insert(self, index: int, block: BlockBase) -> "Workflow":
        """Insert a block at a specific position.

        Args:
            index: Position to insert the block.
            block: The block to insert.

        Returns:
            Self for method chaining.
        """
        self.blocks.insert(index, block)
        return self

    def clear(self) -> "Workflow":
        """Remove all blocks from the workflow.

        Returns:
            Self for method chaining.
        """
        self.blocks.clear()
        return self

    def get_params(self) -> list[Parameter]:
        """Get all mutable parameters from all blocks.

        Returns:
            List of parameter objects linked to blocks.
        """
        all_params: list[Parameter] = []
        for block in self.blocks:
            all_params += list(block.get_params().values())
        return all_params

    def forward(self, input_path: Path, output_path: Path) -> None:
        if len(self.blocks) < 2:
            raise IndexError("A SourceBlock and SinkBlock are required")
        if not isinstance(self.blocks[0], SourceBlock):
            raise ValueError("The first block must be a SourceBlock")
        if not isinstance(self.blocks[-1], SinkBlock):
            raise ValueError("The final block bust be a SinkBlock")
        for block in self.blocks[1:-1]:
            if not isinstance(block, Block):
                raise ValueError("Operator blocks must be a Block")

        iter = self.blocks[0](input_path)
        for block in self.blocks[1:-1]:
            iter = block(iter)

        self.blocks[-1](iter, output_path)

    def __call__(self, input_path: Path, output_path: Path) -> Any:
        """Execute the workflow.

        Args:
            *args: Input arguments for the first block.

        Returns:
            Output from the final block.
        """
        return self.forward(input_path, output_path)

    def __len__(self) -> int:
        """Return the number of blocks in the workflow."""
        return len(self.blocks)

    def __getitem__(self, index: int) -> BlockBase:
        """Get a block by index."""
        return self.blocks[index]

    def __repr__(self) -> str:
        """Generate a visual representation of the workflow."""
        if not self.blocks:
            return text.generate_framed_block(self.name, {})

        # Build visual representation by merging all blocks
        result = str(self.blocks[0])
        for block in self.blocks[1:]:
            block_repr = str(block)
            result = text.column_merge_framed_block(result, block_repr)

        return result
