"""Thread models for the API.

Matches the LangGraph SDK thread format expected by deep-agents-ui.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from ultrathink.api.models.message import Message, TodoItem


ThreadStatus = Literal["idle", "busy", "interrupted", "error"]


class ThreadState(BaseModel):
    """The state of a thread, matching frontend StateType."""

    messages: List[Message] = Field(default_factory=list)
    todos: List[TodoItem] = Field(default_factory=list)
    files: Dict[str, str] = Field(default_factory=dict)
    email: Optional[Dict[str, Any]] = None
    ui: Optional[List[Any]] = None
    interrupt: Optional[Dict[str, Any]] = None


class Thread(BaseModel):
    """A conversation thread.

    Matches the LangGraph SDK Thread type.
    """

    thread_id: str
    created_at: datetime
    updated_at: datetime
    status: ThreadStatus = "idle"
    values: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)

    def get_state(self) -> ThreadState:
        """Get the thread state as a ThreadState object."""
        return ThreadState(
            messages=[
                Message(**m) if isinstance(m, dict) else m
                for m in self.values.get("messages", [])
            ],
            todos=[
                TodoItem(**t) if isinstance(t, dict) else t
                for t in self.values.get("todos", [])
            ],
            files=self.values.get("files", {}),
            email=self.values.get("email"),
            ui=self.values.get("ui"),
            interrupt=self.values.get("interrupt"),
        )

    def update_state(self, updates: Dict[str, Any]) -> None:
        """Update the thread state with new values."""
        for key, value in updates.items():
            self.values[key] = value
        self.updated_at = datetime.utcnow()


class ThreadSearchParams(BaseModel):
    """Parameters for searching threads."""

    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    status: Optional[ThreadStatus] = None
    metadata: Optional[Dict[str, Any]] = None
    sort_by: str = "updated_at"
    sort_order: Literal["asc", "desc"] = "desc"


class StateSnapshot(BaseModel):
    """A snapshot of thread state at a point in time."""

    checkpoint_id: str
    thread_id: str
    timestamp: datetime
    values: Dict[str, Any] = Field(default_factory=dict)
    parent_checkpoint_id: Optional[str] = None
