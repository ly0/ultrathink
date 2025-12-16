"""MCP configuration loader.

This module handles loading MCP server configurations from project files.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""

    command: str
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    cwd: Optional[str] = None


class MCPConfig(BaseModel):
    """Full MCP configuration."""

    mcpServers: Dict[str, MCPServerConfig] = Field(default_factory=dict)


def load_mcp_config(project_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load MCP configuration from project and user directories.

    Searches for MCP config files in the following locations (in order):
    1. User-level: ~/.ultrathink/mcp.json, ~/.mcp.json
    2. Project-level: .mcp.json, mcp_config.json, .ultrathink/mcp.json

    Multiple configs are merged, with project-level configs taking precedence.

    Args:
        project_path: Project directory path (defaults to cwd)

    Returns:
        MCP configuration dict or None if not found
    """
    path = project_path or Path.cwd()
    home = Path.home()

    # Config locations to check (in order of increasing precedence)
    # Later configs override earlier ones
    config_paths = [
        # User-level configs (lowest precedence)
        home / ".ultrathink" / "mcp.json",
        home / ".mcp.json",
        # Project-level configs (highest precedence)
        path / ".mcp.json",
        path / "mcp_config.json",
        path / ".ultrathink" / "mcp.json",
        path / ".ultrathink" / "mcp_config.json",
    ]

    merged_config: Dict[str, Any] = {}

    for config_path in config_paths:
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text())
                normalized = _normalize_config(data)
                if normalized:
                    # Merge configs (later configs override earlier ones)
                    merged_config.update(normalized)
            except Exception:
                continue

    return merged_config if merged_config else None


def _normalize_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize MCP configuration to langchain-mcp-adapters format.

    Converts Claude Desktop / Anthropic MCP config format to the format
    expected by langchain-mcp-adapters, which requires explicit 'transport'.

    Args:
        data: Raw configuration data

    Returns:
        Normalized configuration dict for MultiServerMCPClient
    """
    # Handle different config formats
    if "mcpServers" in data:
        servers = data["mcpServers"]
    elif "servers" in data:
        servers = data["servers"]
    else:
        servers = data

    # Convert to MultiServerMCPClient format
    result = {}
    for name, config in servers.items():
        if not isinstance(config, dict):
            continue

        # Skip disabled servers
        if config.get("disabled", False):
            continue

        # Determine transport type
        # If 'transport' is already specified, use it
        # Otherwise, infer from config:
        #   - command + args -> stdio
        #   - url with sse -> sse
        #   - url with ws -> websocket
        #   - url -> http
        transport = config.get("transport")

        if not transport:
            if "command" in config:
                transport = "stdio"
            elif "url" in config:
                url = config["url"].lower()
                if "sse" in url or config.get("sse"):
                    transport = "sse"
                elif url.startswith("ws://") or url.startswith("wss://"):
                    transport = "websocket"
                else:
                    transport = "http"
            else:
                # Skip invalid config
                continue

        server_config: Dict[str, Any] = {"transport": transport}

        if transport == "stdio":
            server_config["command"] = config.get("command", "")
            server_config["args"] = config.get("args", [])

            # Build env with log suppression unless debug mode is enabled
            import os
            env = dict(config.get("env", {}))
            if not os.environ.get("ULTRATHINK_DEBUG"):
                # Suppress MCP server logging via various common env vars
                env.setdefault("FASTMCP_LOG_LEVEL", "WARNING")
                env.setdefault("LOG_LEVEL", "WARNING")
                env.setdefault("LOGLEVEL", "WARNING")
                env.setdefault("MCP_LOG_LEVEL", "WARNING")
                env.setdefault("PYTHONWARNINGS", "ignore")
                # Disable verbose output in some tools
                env.setdefault("QUIET", "1")
            if env:
                server_config["env"] = env

            if config.get("cwd"):
                server_config["cwd"] = config["cwd"]
        else:
            # For http/sse/websocket transports
            if "url" in config:
                server_config["url"] = config["url"]
            if "headers" in config:
                server_config["headers"] = config["headers"]

        result[name] = server_config

    return result


def get_mcp_server_names(config: Dict[str, Any]) -> List[str]:
    """Get list of configured MCP server names.

    Args:
        config: MCP configuration dict

    Returns:
        List of server names
    """
    return list(config.keys())
