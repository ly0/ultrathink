"""Assistant models for the API.

Matches the LangGraph SDK assistant format expected by deep-agents-ui.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class Assistant(BaseModel):
    """An assistant (agent) configuration.

    Matches the LangGraph SDK Assistant type.
    """

    assistant_id: str
    graph_id: str
    name: str
    created_at: datetime
    updated_at: datetime
    config: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    version: int = 1


class AssistantSearchParams(BaseModel):
    """Parameters for searching assistants."""

    graph_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
