"""MCP runtime management.

This module handles MCP server lifecycle and tool management.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool

from ultrathink.mcp.config_loader import load_mcp_config, get_mcp_server_names
from ultrathink.mcp.tool_adapter import create_mcp_tool


def _suppress_mcp_logging() -> None:
    """Suppress noisy MCP logging unless ULTRATHINK_DEBUG is set."""
    if os.environ.get("ULTRATHINK_DEBUG"):
        return

    # MCP library loggers that produce verbose output
    mcp_loggers = [
        "mcp",
        "mcp.client",
        "mcp.server",
        "mcp.shared",
        "mcp.shared.session",
        "langchain_mcp_adapters",
    ]

    for name in mcp_loggers:
        logger = logging.getLogger(name)
        logger.setLevel(logging.WARNING)
        logger.handlers = []
        logger.propagate = False


class MCPRuntime:
    """Manages MCP server connections and tools.

    This class handles:
    - Loading MCP configuration
    - Connecting to MCP servers
    - Converting MCP tools to LangChain tools
    - Managing server lifecycle
    """

    def __init__(self, project_path: Optional[Path] = None) -> None:
        """Initialize MCP runtime.

        Args:
            project_path: Project directory for config loading
        """
        self.project_path = project_path or Path.cwd()
        self._config: Optional[Dict[str, Any]] = None
        self._client: Optional[Any] = None
        self._tools: List[BaseTool] = []
        self._initialized = False
        self._init_error: Optional[str] = None

    @property
    def config(self) -> Optional[Dict[str, Any]]:
        """Get MCP configuration."""
        if self._config is None:
            self._config = load_mcp_config(self.project_path)
        return self._config

    @property
    def server_names(self) -> List[str]:
        """Get list of configured server names."""
        if self.config:
            return get_mcp_server_names(self.config)
        return []

    def _get_tool_server(self, tool: BaseTool, server_names: List[str]) -> str:
        """Determine which server a tool belongs to.

        Args:
            tool: The LangChain tool
            server_names: List of configured server names

        Returns:
            Server name for the tool
        """
        # Check if tool has metadata with server info
        if hasattr(tool, "metadata") and isinstance(tool.metadata, dict):
            if "server" in tool.metadata:
                return tool.metadata["server"]
            if "server_name" in tool.metadata:
                return tool.metadata["server_name"]

        # Check description for server hints (e.g., "[MCP: server_name]")
        if tool.description and "[MCP:" in tool.description:
            import re
            match = re.search(r'\[MCP:\s*([^\]]+)\]', tool.description)
            if match:
                return match.group(1).strip()

        # If only one server configured, use that
        if len(server_names) == 1:
            return server_names[0]

        # Default fallback
        return "mcp"

    def _prefix_tool(self, tool: BaseTool, server_name: str) -> BaseTool:
        """Create a copy of a tool with prefixed name.

        Args:
            tool: Original LangChain tool
            server_name: Server name for prefix

        Returns:
            Tool with prefixed name
        """
        from langchain_core.tools import StructuredTool

        # Create new name with prefix
        new_name = f"mcp__{server_name}__{tool.name}"

        # Update description to include server info if not already present
        description = tool.description
        if not description.startswith("[MCP:"):
            description = f"[MCP: {server_name}] {description}"

        # Create a wrapper that calls the original tool
        if hasattr(tool, "coroutine") and tool.coroutine:
            async def async_wrapper(**kwargs):
                return await tool.ainvoke(kwargs)

            def sync_wrapper(**kwargs):
                return tool.invoke(kwargs)

            return StructuredTool.from_function(
                name=new_name,
                description=description,
                func=sync_wrapper,
                coroutine=async_wrapper,
                args_schema=tool.args_schema if hasattr(tool, "args_schema") else None,
            )
        else:
            def wrapper(**kwargs):
                return tool.invoke(kwargs)

            return StructuredTool.from_function(
                name=new_name,
                description=description,
                func=wrapper,
                args_schema=tool.args_schema if hasattr(tool, "args_schema") else None,
            )

    async def initialize(self) -> None:
        """Initialize MCP client and load tools."""
        if self._initialized:
            return

        config = self.config
        if not config:
            self._initialized = True
            return

        try:
            # Suppress MCP logging before importing/using the client
            _suppress_mcp_logging()

            # Monkeypatch the MCP client to suppress subprocess stderr
            # unless debug mode is enabled
            if not os.environ.get("ULTRATHINK_DEBUG"):
                from contextlib import asynccontextmanager
                import mcp.client.stdio as stdio_module

                # Check if we've already patched
                if not hasattr(stdio_module, "_ultrathink_patched"):
                    _original_stdio_client = stdio_module.stdio_client

                    @asynccontextmanager
                    async def _quiet_stdio_client(server, errlog=None):
                        """Wrapper that redirects errlog to /dev/null."""
                        # Use os.devnull for proper file descriptor support
                        with open(os.devnull, 'w') as null_file:
                            async with _original_stdio_client(server, errlog=null_file) as result:
                                yield result

                    stdio_module.stdio_client = _quiet_stdio_client
                    stdio_module._ultrathink_patched = True

            from langchain_mcp_adapters.client import MultiServerMCPClient

            # Create client and get tools (new API - no context manager)
            self._client = MultiServerMCPClient(config)

            # Get tools from all servers
            # The new API returns LangChain tools directly
            mcp_tools = await self._client.get_tools()

            # Get server names from config for prefixing
            server_names = list(config.keys())

            # Process tools and ensure proper naming with server prefix
            for tool in mcp_tools:
                if isinstance(tool, BaseTool):
                    # Check if tool already has mcp__ prefix
                    if not tool.name.startswith("mcp__"):
                        # Add server prefix - try to determine server from tool metadata
                        # or use first server if only one configured
                        server_name = self._get_tool_server(tool, server_names)
                        prefixed_tool = self._prefix_tool(tool, server_name)
                        self._tools.append(prefixed_tool)
                    else:
                        self._tools.append(tool)
                elif isinstance(tool, dict):
                    # Need to convert from dict format
                    server_name = tool.get("server", server_names[0] if server_names else "unknown")
                    lc_tool = create_mcp_tool(
                        tool,
                        self._call_tool,
                        server_name,
                    )
                    self._tools.append(lc_tool)
                else:
                    # Assume it's a tool-like object, add directly
                    self._tools.append(tool)

            self._initialized = True

        except ImportError as e:
            # langchain-mcp-adapters not installed - only warn in debug mode
            if os.environ.get("ULTRATHINK_DEBUG"):
                print(f"Warning: langchain-mcp-adapters not available: {e}")
            self._initialized = True
        except Exception as e:
            # Log error but continue without MCP
            # Always show a brief warning, but only show traceback in debug mode
            self._init_error = str(e)
            if os.environ.get("ULTRATHINK_DEBUG"):
                import traceback
                print(f"Warning: Failed to initialize MCP: {e}")
                traceback.print_exc()
            self._initialized = True

    async def _call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Any:
        """Call an MCP tool.

        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Tool result
        """
        if not self._client:
            raise RuntimeError("MCP client not initialized")

        return await self._client.call_tool(server_name, tool_name, arguments)

    @property
    def tools(self) -> List[BaseTool]:
        """Get list of MCP tools as LangChain tools."""
        return self._tools

    async def cleanup(self) -> None:
        """Clean up MCP client and connections."""
        if self._client:
            try:
                # Try to close the client if it has a close method
                if hasattr(self._client, "close"):
                    await self._client.close()
                elif hasattr(self._client, "aclose"):
                    await self._client.aclose()
            except Exception:
                pass
            self._client = None
        self._tools = []
        self._initialized = False

    async def __aenter__(self) -> "MCPRuntime":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.cleanup()


# Global runtime instance
_runtime: Optional[MCPRuntime] = None


async def get_mcp_runtime(project_path: Optional[Path] = None) -> MCPRuntime:
    """Get or create the MCP runtime.

    Args:
        project_path: Project directory for config loading

    Returns:
        MCPRuntime instance
    """
    global _runtime

    # Resolve paths for consistent comparison
    if project_path:
        project_path = project_path.resolve()

    # Check if we need to create a new runtime
    need_new_runtime = _runtime is None
    if not need_new_runtime and project_path:
        current_path = _runtime.project_path.resolve() if _runtime.project_path else None
        need_new_runtime = current_path != project_path

    if need_new_runtime:
        if _runtime:
            await _runtime.cleanup()
        _runtime = MCPRuntime(project_path)
        await _runtime.initialize()

    return _runtime


async def get_mcp_tools(project_path: Optional[Path] = None) -> List[BaseTool]:
    """Get MCP tools for the project.

    Args:
        project_path: Project directory for config loading

    Returns:
        List of MCP tools as LangChain tools
    """
    runtime = await get_mcp_runtime(project_path)
    return runtime.tools


def format_mcp_instructions(runtime: MCPRuntime) -> str:
    """Build MCP instruction block for the system prompt.

    This creates a concise summary of available MCP servers and their tools
    that helps the model understand what MCP capabilities are available.

    Args:
        runtime: The MCP runtime instance

    Returns:
        Formatted instruction string for the system prompt
    """
    if not runtime or not runtime._initialized:
        return ""

    config = runtime.config
    if not config:
        return ""

    lines: List[str] = [
        "MCP servers are available. Tools from these servers are prefixed with mcp__<server>__<tool>.",
    ]

    for server_name, server_config in config.items():
        # Get server status
        status = "connected" if runtime._initialized and runtime._client else "not started"

        # Build server line
        transport = server_config.get("transport", "stdio")
        cmd = server_config.get("command", "")
        lines.append(f"- {server_name} [{status}] ({transport})")

        if cmd:
            args = " ".join(server_config.get("args", []))[:50]
            lines.append(f"  Command: {cmd} {args}")

        # Get tools for this server
        server_tools = [t for t in runtime.tools if t.name.startswith(f"mcp__{server_name}__")]
        if server_tools:
            tool_names = [t.name.split("__")[-1] for t in server_tools[:6]]
            tool_summary = ", ".join(tool_names)
            if len(server_tools) > 6:
                tool_summary += f", and {len(server_tools) - 6} more"
            lines.append(f"  Tools: {tool_summary}")

    return "\n".join(lines)


def get_mcp_instructions_sync(project_path: Optional[Path] = None) -> str:
    """Get MCP instructions synchronously (for use when runtime is already initialized).

    Args:
        project_path: Project directory

    Returns:
        MCP instructions string or empty string
    """
    global _runtime

    if _runtime and _runtime._initialized:
        return format_mcp_instructions(_runtime)
    return ""
