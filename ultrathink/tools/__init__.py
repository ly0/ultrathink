"""Custom tools for Ultrathink.

This module provides additional tools beyond what deepagent provides by default.
"""

from pathlib import Path
from typing import List, Optional

from langchain_core.tools import BaseTool

from ultrathink.tools.ask_user import create_ask_user_tool, create_ask_user_multi_tool
from ultrathink.tools.filesystem import create_filesystem_tools


def get_custom_tools(
    safe_mode: bool = True,
    cwd: Optional[Path] = None,
    ui_callback=None,
    ui_multi_callback=None,
    include_filesystem: bool = False,
) -> List[BaseTool]:
    """Get all custom tools for Ultrathink.

    Args:
        safe_mode: Whether to enable permission checks
        cwd: Current working directory
        ui_callback: Callback function for single-question user interaction
        ui_multi_callback: Callback function for multi-question user interaction
        include_filesystem: Whether to include filesystem tools

    Returns:
        List of custom tools
    """
    tools = []
    working_dir = cwd or Path.cwd()

    # Add ask_user tool if we have a UI callback
    if ui_callback is not None:
        tools.append(create_ask_user_tool(ui_callback))

    # Add ask_user_multi tool if we have a multi UI callback
    if ui_multi_callback is not None:
        tools.append(create_ask_user_multi_tool(ui_multi_callback))

    # Add filesystem tools if requested
    # (used for DeepSeek Reasoner which doesn't use deepagents' FilesystemMiddleware)
    if include_filesystem:
        tools.extend(create_filesystem_tools(working_dir))

    return tools


__all__ = ["get_custom_tools", "create_filesystem_tools"]
