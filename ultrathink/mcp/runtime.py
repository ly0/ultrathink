"""MCP runtime management.

This module handles MCP server lifecycle and tool management.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool

from ultrathink.mcp.config_loader import load_mcp_config, get_mcp_server_names
from ultrathink.mcp.tool_adapter import create_mcp_tool


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

    async def initialize(self) -> None:
        """Initialize MCP client and load tools."""
        if self._initialized:
            return

        config = self.config
        if not config:
            self._initialized = True
            return

        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            self._client = MultiServerMCPClient(config)
            await self._client.__aenter__()

            # Get tools from all servers
            mcp_tools = await self._client.get_tools()

            # Convert to LangChain tools
            for tool_info in mcp_tools:
                server_name = tool_info.get("server", "unknown")
                lc_tool = create_mcp_tool(
                    tool_info,
                    self._call_tool,
                    server_name,
                )
                self._tools.append(lc_tool)

            self._initialized = True

        except ImportError:
            # langchain-mcp-adapters not installed
            self._initialized = True
        except Exception as e:
            # Log error but continue without MCP
            print(f"Warning: Failed to initialize MCP: {e}")
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
                await self._client.__aexit__(None, None, None)
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

    if _runtime is None or (project_path and _runtime.project_path != project_path):
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
