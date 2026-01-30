"""Workflow class for dynamically building and executing block pipelines."""

from pathlib import Path
from typing import Any

from cmxflow.block import Block, BlockBase, ScoreBlock, SinkBlock, SourceBlock
from cmxflow.parameter import Parameter
from cmxflow.utils import text
from cmxflow.utils.parallel import ParallelBlock


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

    def add(self, *blocks: Any) -> "Workflow":
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

    def check(self) -> None:
        """Validate the workflow structure.

        Ensures the workflow has a SourceBlock first, SinkBlock last,
        and only Block instances in between.

        Raises:
            IndexError: If fewer than 2 blocks are present.
            ValueError: If block types are in incorrect positions.
        """
        # Check for required structure
        if len(self.blocks) < 2:
            raise IndexError("A SourceBlock and SinkBlock are required")
        if not isinstance(self.blocks[0], SourceBlock):
            raise ValueError("The first block must be a SourceBlock")
        if not isinstance(self.blocks[-1], (SinkBlock, ScoreBlock)):
            raise ValueError("The final block bust be a SinkBlock")
        for block in self.blocks[1:-1]:
            if not isinstance(block, (Block, ParallelBlock)):
                raise ValueError("Operator blocks must be a (Parallel)Block")
        # Compute properties
        self._n = len(self.blocks)
        self._operator_index = list(range(1, self._n - 1))

    def get_required_input(self) -> dict[str, type]:
        """Get all required inputs from blocks in the workflow.

        Returns:
            Dictionary mapping input keys to their expected types.
            Keys are formatted as '{block_index}.{type}@{name}'.
        """
        self.check()
        required: dict[str, type] = {}
        for i in self._operator_index:
            block = self.blocks[i]
            if block.input_files is not None:
                for key in block.input_files.keys():
                    required[f"{i}.file@{key}"] = str
            if block.input_text is not None:
                for key in block.input_text.keys():
                    required[f"{i}.text@{key}"] = str

        if isinstance(self.blocks[-1], ScoreBlock):
            i = len(self.blocks) - 1
            block = self.blocks[-1]
            if block.input_files is not None:
                for key in block.input_files.keys():
                    required[f"{i}.file@{key}"] = str
            if block.input_text is not None:
                for key in block.input_text.keys():
                    required[f"{i}.text@{key}"] = str

        return required

    def set_required_input(self, required_inputs: dict[str, str]) -> None:
        """Set required inputs for blocks in the workflow.

        Args:
            required_inputs: Dictionary mapping input keys to values.
                Keys should match those returned by get_required_input().

        Raises:
            KeyError: If a required input is missing.
            FileNotFoundError: If a required input file does not exist.
        """
        required = self.get_required_input()
        for key in required:
            # Check for required input
            if key not in required_inputs:
                raise KeyError(f"Required inputs missing {key}")
            uid, name = key.split("@")
            bid_str, itype = uid.split(".")
            bid = int(bid_str)
            if itype == "file":
                # Make sure file exists
                path = Path(required_inputs[key])
                if not path.is_file():
                    raise FileNotFoundError(f"Required input file {key} does not exist")
                self.blocks[bid].input_files[name] = path
            elif itype == "text":
                self.blocks[bid].input_text[name] = required_inputs[key]

    def forward(
        self, input_path: Path | str, output_path: Path | str = ""
    ) -> tuple[float, tuple[str, ...] | None] | None:
        """Execute the workflow pipeline.

        Args:
            input_path: Path to the input file for the SourceBlock.
            output_path: Path to the output file for the SinkBlock.
        """
        self.check()

        if isinstance(input_path, str):
            input_path = Path(input_path)
        if isinstance(output_path, str):
            output_path = Path(output_path)

        iter = self.blocks[0](input_path)
        for block in self.blocks[1:-1]:
            iter = block(iter)

        if isinstance(self.blocks[-1], SinkBlock):
            self.blocks[-1](iter, output_path)
            return None
        elif isinstance(self.blocks[-1], ScoreBlock):
            uid = tuple([str(p) for p in self.get_params()])
            return self.blocks[-1](iter, uid)
        return None

    def __call__(self, input_path: Path | str, output_path: Path | str = "") -> Any:
        """Execute the workflow.

        Args:
            input_path: Input path.
            output_path: Output path (optional).

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
