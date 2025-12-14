"""MCP integration middleware for deepagent.

This middleware loads MCP tools and injects them into the agent.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langchain_core.tools import BaseTool


class MCPMiddleware:
    """Middleware that integrates MCP tools into the agent.

    This middleware:
    - Loads MCP server configuration
    - Initializes MCP client connections
    - Converts MCP tools to LangChain format
    - Provides tools to the agent
    """

    def __init__(
        self,
        project_path: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
        system_prompt_addon: Optional[str] = None,
    ) -> None:
        """Initialize MCP middleware.

        Args:
            project_path: Project directory for config loading
            config: Pre-loaded MCP configuration
            system_prompt_addon: Additional system prompt text
        """
        self.project_path = project_path or Path.cwd()
        self._config = config
        self._system_prompt_addon = system_prompt_addon
        self._tools: List[BaseTool] = []
        self._initialized = False

    @property
    def tools(self) -> List[BaseTool]:
        """Get MCP tools."""
        return self._tools

    @property
    def system_prompt(self) -> str:
        """Get system prompt addition for MCP tools."""
        if not self._tools:
            return ""

        lines = ["# MCP Tools Available"]
        for tool in self._tools:
            desc = tool.description[:100] + "..." if len(tool.description) > 100 else tool.description
            lines.append(f"- {tool.name}: {desc}")

        if self._system_prompt_addon:
            lines.append("")
            lines.append(self._system_prompt_addon)

        return "\n".join(lines)

    async def initialize(self) -> None:
        """Initialize MCP connections and load tools."""
        if self._initialized:
            return

        try:
            from ultrathink.mcp.runtime import get_mcp_tools

            self._tools = await get_mcp_tools(self.project_path)
            self._initialized = True

        except ImportError:
            # MCP dependencies not installed
            self._initialized = True
        except Exception as e:
            print(f"Warning: Failed to initialize MCP middleware: {e}")
            self._initialized = True

    async def cleanup(self) -> None:
        """Clean up MCP connections."""
        from ultrathink.mcp.runtime import _runtime

        if _runtime:
            await _runtime.cleanup()

        self._tools = []
        self._initialized = False
