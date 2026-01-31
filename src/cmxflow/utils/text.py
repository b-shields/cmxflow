"""Text formatting utilities for workflow visualization."""

import re
from typing import Any

# Regex to match ANSI escape sequences
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from a string.

    Args:
        text: String potentially containing ANSI escape codes.

    Returns:
        String with all ANSI escape codes removed.
    """
    return ANSI_ESCAPE_PATTERN.sub("", text)


def visual_len(text: str) -> int:
    """Return the visual width of a string (excluding ANSI codes).

    Args:
        text: String potentially containing ANSI escape codes.

    Returns:
        The visual width of the string as rendered in a terminal.
    """
    return len(strip_ansi(text))


def visual_ljust(text: str, width: int) -> str:
    """Left-justify a string to visual width, accounting for ANSI codes.

    Args:
        text: String to left-justify.
        width: Target visual width.

    Returns:
        String padded with spaces to reach the target visual width.
    """
    padding = width - visual_len(text)
    return text + " " * max(0, padding)


def visual_center(text: str, width: int) -> str:
    """Center a string to visual width, accounting for ANSI codes.

    Args:
        text: String to center.
        width: Target visual width.

    Returns:
        String padded with spaces on both sides to center it.
    """
    padding = width - visual_len(text)
    left_pad = padding // 2
    right_pad = padding - left_pad
    return " " * left_pad + text + " " * right_pad


def generate_framed_block(
    name: str, parameters: dict[str, Any], max_width: int = 50
) -> str:
    """Generate a framed text block with a name header and parameters.

    Args:
        name: Header name to display centered at the top of the block.
        parameters: Dictionary of key-value pairs to display in the block.
        max_width: Maximum width for parameter lines before truncation.

    Returns:
        A string containing the framed block with box-drawing characters.
    """
    # Limit width of view
    limited_params = {}
    for key, value in parameters.items():
        key = str(key)
        value = str(value)
        # Color text
        if value in ["[FILE]", "[TEXT]"]:
            if key in ["input", "output"]:
                value = "\033[92m" + value + "\033[0m"
            else:
                value = "\033[31m" + value + "\033[0m"
        width = visual_len(f"{key}: {value}")
        if width > max_width:
            remove = width - max_width
            # Strip ANSI, truncate, but we lose color on truncated values
            value = strip_ansi(value)[:-remove]
        limited_params[key] = value

    param_lines = [f"{key}: {value}" for key, value in limited_params.items()]

    # Calculate frame width based on content
    all_lines = [name] + param_lines
    content_width = max(visual_len(line) for line in all_lines)
    inner_width = content_width + 2  # padding on each side

    # Build the frame
    top_border = "┌" + "─" * inner_width + "┐"
    bottom_border = "└" + "─" * inner_width + "┘"
    separator = "├" + "─" * inner_width + "┤"

    # Center the name
    name_line = "│ " + visual_center(name, content_width) + " │"

    # Left-align parameters
    param_output = [f"│ {visual_ljust(line, content_width)} │" for line in param_lines]

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

    # Get the width of the left block (visual width)
    left_width = max(visual_len(line) for line in left_lines)

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

    # Get the width of the right block for consistent output (visual width)
    right_width = max(visual_len(line) for line in right_lines)

    # Build the merged output
    arrow_spacer = " " * len(arrow)
    result_lines: list[str] = []
    for i in range(total_height):
        left_line = visual_ljust(left_padded[i], left_width)
        right_line = visual_ljust(right_padded[i], right_width)

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

    # Get the width of each block (visual width, normalize lines within each block)
    top_width = max(visual_len(line) for line in top_lines)
    bottom_width = max(visual_len(line) for line in bottom_lines)
    max_width = max(top_width, bottom_width)

    # Normalize each block's lines to consistent width within the block
    top_normalized = [visual_ljust(line, top_width) for line in top_lines]
    bottom_normalized = [visual_ljust(line, bottom_width) for line in bottom_lines]

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
    arrow_line = visual_center(arrow, max_width)

    # Combine all lines
    result_lines = top_padded + [arrow_line] + bottom_padded

    return "\n".join(result_lines)
