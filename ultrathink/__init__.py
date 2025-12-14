"""Ultrathink - AI-powered coding assistant built on deepagent."""

__version__ = "0.1.0"
__author__ = "Ultrathink Team"

from ultrathink.core.agent_factory import create_ultrathink_agent
from ultrathink.sdk.client import UltrathinkClient

__all__ = [
    "__version__",
    "create_ultrathink_agent",
    "UltrathinkClient",
]
