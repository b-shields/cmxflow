from typing import Any


def generate_framed_block(
    name: str, parameters: dict[str, Any], max_width: int = 50
) -> str:
    """Generate a framed text block with a name header and parameters."""
    # Limit width of view
    limited_params = {}
    for key, value in parameters.items():
        key = str(key)
        value = str(value)
        width = len(f"{key}: {value}")
        if width > max_width:
            remove = width - max_width
            value = value[:-remove]
        limited_params[key] = value

    param_lines = [f"{key}: {value}" for key, value in limited_params.items()]

    # Calculate frame width based on content
    all_lines = [name] + param_lines
    content_width = max(len(line) for line in all_lines)
    inner_width = content_width + 2  # padding on each side

    # Build the frame
    top_border = "┌" + "─" * inner_width + "┐"
    bottom_border = "└" + "─" * inner_width + "┘"
    separator = "├" + "─" * inner_width + "┤"

    # Center the name
    name_line = "│ " + name.center(content_width) + " │"

    # Left-align parameters
    param_output = [f"│ {line.ljust(content_width)} │" for line in param_lines]

    # Assemble the block
    lines = [top_border, name_line, separator] + param_output + [bottom_border]

    return "\n".join(lines)


def left_merge_framed_block(left_block: str, right_block: str, arrow: str = "←") -> str:
    """Merge two framed blocks with a left-to-right arrow in between.

    The arrow is centered vertically on both blocks by aligning their
    vertical centers.

    Args:
        left_block: The left framed block (output from generate_framed_block).
        right_block: The right framed block (output from generate_framed_block).
        arrow: The arrow character(s) to use between blocks.

    Returns:
        A string with both blocks side by side, connected by an arrow.
    """
    left_lines = left_block.split("\n")
    right_lines = right_block.split("\n")

    left_height = len(left_lines)
    right_height = len(right_lines)

    # Calculate center row for each block (0-indexed within each block)
    left_center = left_height // 2
    right_center = right_height // 2

    # Calculate how far each block extends above and below its center
    left_above = left_center
    left_below = left_height - left_center - 1
    right_above = right_center
    right_below = right_height - right_center - 1

    # The arrow row in the merged output (where both centers align)
    arrow_row = max(left_above, right_above)

    # Total height of the merged output
    total_height = arrow_row + max(left_below, right_below) + 1

    # Get the width of the left block
    left_width = max(len(line) for line in left_lines)

    # Calculate top padding for each block to align centers
    left_top_pad = arrow_row - left_center
    right_top_pad = arrow_row - right_center

    # Create padded line lists with empty strings of correct width
    left_padded: list[str] = (
        [" " * left_width] * left_top_pad
        + left_lines
        + [" " * left_width] * (total_height - left_top_pad - left_height)
    )
    right_padded: list[str] = (
        [""] * right_top_pad
        + right_lines
        + [""] * (total_height - right_top_pad - right_height)
    )

    # Get the width of the right block for consistent output
    right_width = max(len(line) for line in right_lines)

    # Build the merged output
    arrow_spacer = " " * len(arrow)
    result_lines: list[str] = []
    for i in range(total_height):
        left_line = left_padded[i].ljust(left_width)
        right_line = right_padded[i].ljust(right_width)

        if i == arrow_row:
            connector = f" {arrow} "
        else:
            connector = f" {arrow_spacer} "

        result_lines.append(left_line + connector + right_line)

    return "\n".join(result_lines)


def column_merge_framed_block(
    top_block: str, bottom_block: str, arrow: str = "↓"
) -> str:
    """Merge two framed blocks vertically with an arrow in between.

    The arrow is centered horizontally on both blocks. Each block is
    padded as a whole unit to preserve internal alignment.

    Args:
        top_block: The top framed block (output from generate_framed_block).
        bottom_block: The bottom framed block (output from generate_framed_block).
        arrow: The arrow character(s) to use between blocks.

    Returns:
        A string with both blocks stacked vertically, connected by an arrow.
    """
    top_lines = top_block.split("\n")
    bottom_lines = bottom_block.split("\n")

    # Get the width of each block (normalize lines within each block first)
    top_width = max(len(line) for line in top_lines)
    bottom_width = max(len(line) for line in bottom_lines)
    max_width = max(top_width, bottom_width)

    # Normalize each block's lines to consistent width within the block
    top_normalized = [line.ljust(top_width) for line in top_lines]
    bottom_normalized = [line.ljust(bottom_width) for line in bottom_lines]

    # Calculate left padding to center each block as a whole
    top_left_pad = (max_width - top_width) // 2
    bottom_left_pad = (max_width - bottom_width) // 2

    # Pad each block uniformly to center it
    top_padded = [
        " " * top_left_pad + line + " " * (max_width - top_width - top_left_pad)
        for line in top_normalized
    ]
    bottom_padded = [
        " " * bottom_left_pad
        + line
        + " " * (max_width - bottom_width - bottom_left_pad)
        for line in bottom_normalized
    ]

    # Create the centered arrow line
    arrow_line = arrow.center(max_width)

    # Combine all lines
    result_lines = top_padded + [arrow_line] + bottom_padded

    return "\n".join(result_lines)
