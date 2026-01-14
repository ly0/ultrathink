"""Ultrathink - AI-powered coding assistant built on deepagent."""

import logging
import os
import warnings

__version__ = "0.1.0"
__author__ = "Ultrathink Team"

# Suppress noisy third-party warnings
if not os.environ.get("ULTRATHINK_DEBUG"):
    # Suppress transformers/tokenizers warnings about missing ML frameworks
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    warnings.filterwarnings("ignore", message="None of PyTorch")
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

    # Suppress MCP server logging (FastMCP uses FASTMCP_ prefix for env vars)
    os.environ.setdefault("FASTMCP_LOG_LEVEL", "WARNING")

# Suppress MCP debug logging by default (uses rich handler)
# Set ULTRATHINK_DEBUG=1 to see all logs
if not os.environ.get("ULTRATHINK_DEBUG"):
    # Configure default logging to suppress noisy third-party loggers
    # This can be overridden by calling configure_logging() with verbose=True
    _noisy_loggers = [
        "langchain_mcp_adapters",
        "mcp",
        "mcp.client",
        "mcp.server",
        "mcp.shared",
        "httpx",
        "httpcore",
        "urllib3",
        "asyncio",
        "anyio",
        "langchain",
        "langchain_core",
        "langchain_anthropic",
        "langchain_openai",
        "openai",
        "anthropic",
    ]

    for _logger_name in _noisy_loggers:
        logging.getLogger(_logger_name).setLevel(logging.WARNING)

    # Also disable any handlers that might have been added
    for _logger_name in _noisy_loggers:
        _logger = logging.getLogger(_logger_name)
        _logger.handlers = []
        _logger.propagate = False

# Lazy imports to avoid breaking the API when there are dependency conflicts
# These will be loaded on first access
_create_ultrathink_agent = None
_UltrathinkClient = None


def __getattr__(name):
    """Lazy loading of optional components that may have dependency issues."""
    global _create_ultrathink_agent, _UltrathinkClient

    if name == "create_ultrathink_agent":
        if _create_ultrathink_agent is None:
            try:
                from ultrathink.core.agent_factory import create_ultrathink_agent as _func
                _create_ultrathink_agent = _func
            except ImportError as e:
                raise ImportError(
                    f"Cannot import create_ultrathink_agent due to dependency conflict: {e}. "
                    "Please check your langchain and pydantic versions."
                ) from e
        return _create_ultrathink_agent

    if name == "UltrathinkClient":
        if _UltrathinkClient is None:
            try:
                from ultrathink.sdk.client import UltrathinkClient as _cls
                _UltrathinkClient = _cls
            except ImportError as e:
                raise ImportError(
                    f"Cannot import UltrathinkClient due to dependency conflict: {e}. "
                    "Please check your langchain and pydantic versions."
                ) from e
        return _UltrathinkClient

    raise AttributeError(f"module 'ultrathink' has no attribute '{name}'")


__all__ = [
    "__version__",
    "create_ultrathink_agent",
    "UltrathinkClient",
]
