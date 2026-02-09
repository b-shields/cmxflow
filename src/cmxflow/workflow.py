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

    def reset_cache(self) -> None:
        """Reset cached state in all blocks for a new optimization iteration."""
        for block in self.blocks:
            block.reset_cache()

    def add(self, *blocks: Any) -> "Workflow":
        """Add a block to the end of the workflow.

        Args:
            block: The block to add.

        Returns:
            Self for method chaining.
        """
        for block in blocks:
            self.blocks.append(block)
        self.reset_cache()
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
            raise ValueError("The final block bust be a SinkBlock or ScoreBlock")
        for block in self.blocks[1:-1]:
            if not isinstance(block, (Block, ParallelBlock, ScoreBlock)):
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
        missing_files = []
        missing_text = []
        for key in self.get_required_input():
            # Get workflow to block map
            uid, name = key.split("@")
            bid_str, itype = uid.split(".")
            bid = int(bid_str)
            # Check files
            if itype == "file":
                if key in required_inputs:
                    path = Path(required_inputs[key])
                    if path.is_file():
                        self.blocks[bid].input_files[name] = path
                    elif not self.blocks[bid].input_files[name].is_file():
                        raise FileNotFoundError(
                            f"Input file specified for key {key} does not exist."
                        )
                if not self.blocks[bid].input_files[name].is_file():
                    missing_files.append(f"'{key}'")
            # Check text
            elif itype == "text":
                if key in required_inputs:
                    self.blocks[bid].input_text[name] = required_inputs[key]
                if not self.blocks[bid].input_text[name]:
                    missing_text.append(f"'{key}'")
        if missing_files:
            raise KeyError(
                f"Required input files are missing: {", ".join(missing_files)}"
            )
        if missing_text:
            raise KeyError(
                f"Required input text keys are missing: {", ".join(missing_text)}"
            )

    def forward(
        self, input_path: Path | str, output_path: Path | str = ""
    ) -> float | None:
        """Execute the workflow pipeline.

        Args:
            input_path: Path to the input file for the SourceBlock.
            output_path: Path to the output file for the SinkBlock.

        Returns:
            Score value if workflow ends with ScoreBlock, None otherwise.
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
            self.blocks[-1]._set_score_properties(*self.blocks)
            all_but_score_params: list[Parameter] = []
            for block in self.blocks[:-1]:
                all_but_score_params += list(block.get_params().values())
            uid = tuple([str(p) for p in all_but_score_params])
            result = self.blocks[-1](iter, uid)
            if isinstance(result, float):
                return result
            return None
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

        return "\n" + result


class WorkflowValidationError(Exception):
    """Raised when a workflow fails validation during save or load."""

    pass


# Re-export from canonical location for backward compatibility
from cmxflow.utils.serial import load_workflow, save_workflow  # noqa: E402, F401
