"""MCP server for cmxflow workflow building and execution."""

from cmxflow.mcp.server import mcp


def run() -> None:
    """Run the MCP server."""
    mcp.run()


__all__ = ["mcp", "run"]
