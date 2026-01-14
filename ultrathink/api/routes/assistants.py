"""Assistant management endpoints.

These endpoints match the LangGraph SDK assistant API.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from ultrathink.api.models.assistant import Assistant, AssistantSearchParams
from ultrathink.api.services.assistant_registry import get_assistant_registry

router = APIRouter()


@router.get("/{assistant_id}")
async def get_assistant(assistant_id: str) -> Assistant:
    """Get an assistant by ID.

    Args:
        assistant_id: The assistant ID or graph name.

    Returns:
        The assistant.

    Raises:
        HTTPException: If assistant not found.
    """
    registry = get_assistant_registry()
    assistant = await registry.get(assistant_id)
    if assistant is None:
        raise HTTPException(status_code=404, detail="Assistant not found")
    return assistant


@router.post("/search")
async def search_assistants(
    graph_id: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
) -> List[Assistant]:
    """Search for assistants.

    Args:
        graph_id: Optional graph ID to filter by.
        limit: Maximum number of results.

    Returns:
        List of matching assistants.
    """
    registry = get_assistant_registry()
    params = AssistantSearchParams(graph_id=graph_id, limit=limit)
    return await registry.search(params)
