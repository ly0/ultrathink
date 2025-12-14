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
    """Load MCP configuration from project directory.

    Looks for mcp_config.json or .ultrathink/mcp.json in the project directory.

    Args:
        project_path: Project directory path (defaults to cwd)

    Returns:
        MCP configuration dict or None if not found
    """
    path = project_path or Path.cwd()

    # Check possible config locations
    config_paths = [
        path / "mcp_config.json",
        path / ".ultrathink" / "mcp.json",
        path / ".ultrathink" / "mcp_config.json",
    ]

    for config_path in config_paths:
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text())
                return _normalize_config(data)
            except Exception:
                continue

    return None


def _normalize_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize MCP configuration to standard format.

    Args:
        data: Raw configuration data

    Returns:
        Normalized configuration dict
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
        if isinstance(config, dict):
            server_config = {
                "command": config.get("command", ""),
                "args": config.get("args", []),
                "env": config.get("env", {}),
            }
            if "cwd" in config:
                server_config["cwd"] = config["cwd"]
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
