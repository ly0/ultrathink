"""Assistant registry service.

Manages available assistants (agents) for the API.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from ultrathink.api.models.assistant import Assistant, AssistantSearchParams


class AssistantRegistry:
    """Registry of available assistants.

    For ultrathink, we provide a single default assistant that uses
    the configured model and tools.
    """

    def __init__(self):
        """Initialize the registry with the default assistant."""
        self._assistants: Dict[str, Assistant] = {}
        self._init_default_assistant()

    def _init_default_assistant(self) -> None:
        """Create the default ultrathink assistant."""
        now = datetime.utcnow()
        default = Assistant(
            assistant_id="default",
            graph_id="ultrathink",
            name="Ultrathink Assistant",
            created_at=now,
            updated_at=now,
            config={
                "recursion_limit": 100,
            },
            metadata={
                "description": "AI-powered coding assistant",
            },
        )
        self._assistants["default"] = default

    async def get(self, assistant_id: str) -> Optional[Assistant]:
        """Get an assistant by ID.

        Args:
            assistant_id: The assistant ID or graph name.

        Returns:
            The assistant, or None if not found.
        """
        # Support both UUID and graph name lookup
        if assistant_id in self._assistants:
            return self._assistants[assistant_id]

        # Check by graph_id
        for assistant in self._assistants.values():
            if assistant.graph_id == assistant_id:
                return assistant

        # For local development, treat any ID as the default
        return self._assistants.get("default")

    async def search(self, params: AssistantSearchParams) -> List[Assistant]:
        """Search for assistants.

        Args:
            params: Search parameters.

        Returns:
            List of matching assistants.
        """
        assistants = list(self._assistants.values())

        # Filter by graph_id
        if params.graph_id:
            assistants = [a for a in assistants if a.graph_id == params.graph_id]

        # Filter by metadata
        if params.metadata:
            def matches_metadata(assistant: Assistant) -> bool:
                for key, value in params.metadata.items():  # type: ignore
                    if assistant.metadata.get(key) != value:
                        return False
                return True

            assistants = [a for a in assistants if matches_metadata(a)]

        # Paginate
        start = params.offset
        end = start + params.limit
        return assistants[start:end]

    async def register(self, assistant: Assistant) -> None:
        """Register a new assistant.

        Args:
            assistant: The assistant to register.
        """
        self._assistants[assistant.assistant_id] = assistant

    async def unregister(self, assistant_id: str) -> bool:
        """Unregister an assistant.

        Args:
            assistant_id: The assistant ID.

        Returns:
            True if removed, False if not found.
        """
        if assistant_id in self._assistants:
            del self._assistants[assistant_id]
            return True
        return False


# Global instance
_registry: Optional[AssistantRegistry] = None


def get_assistant_registry() -> AssistantRegistry:
    """Get the global assistant registry instance."""
    global _registry
    if _registry is None:
        _registry = AssistantRegistry()
    return _registry
