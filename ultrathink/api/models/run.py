"""Run and streaming models for the API.

Matches the LangGraph SDK run format expected by deep-agents-ui.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    """Configuration for a run."""

    recursion_limit: int = Field(default=100)
    configurable: Dict[str, Any] = Field(default_factory=dict)


class Checkpoint(BaseModel):
    """A checkpoint for resuming execution."""

    checkpoint_id: str
    thread_id: str
    ts: datetime
    channel_values: Dict[str, Any] = Field(default_factory=dict)
    parent_checkpoint_id: Optional[str] = None


class Command(BaseModel):
    """A command to control execution flow."""

    resume: Optional[Any] = None
    goto: Optional[str] = None
    update: Optional[Any] = None


class StreamInput(BaseModel):
    """Input for streaming execution."""

    input: Optional[Dict[str, Any]] = None
    config: Optional[RunConfig] = None
    checkpoint: Optional[Checkpoint] = None
    interrupt_before: Optional[List[str]] = None
    interrupt_after: Optional[List[str]] = None
    command: Optional[Command] = None


class ActionRequest(BaseModel):
    """A request for an action (tool call) that needs approval."""

    name: str
    args: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None


class ReviewConfig(BaseModel):
    """Configuration for reviewing an action."""

    actionName: str
    allowedDecisions: List[str] = Field(default_factory=lambda: ["approve", "reject", "edit"])


class Interrupt(BaseModel):
    """An interrupt that requires user action."""

    value: Any
    ns: Optional[List[str]] = None
    scope: Optional[str] = None

    @classmethod
    def for_tool_approval(
        cls, action_requests: List[ActionRequest], review_configs: Optional[List[ReviewConfig]] = None
    ) -> "Interrupt":
        """Create an interrupt for tool approval."""
        return cls(
            value={
                "action_requests": [ar.model_dump() for ar in action_requests],
                "review_configs": [rc.model_dump() for rc in review_configs]
                if review_configs
                else None,
            }
        )


class StreamEventType(str, Enum):
    """Types of events in the stream."""

    # Thread lifecycle
    THREAD_CREATED = "thread_created"

    # State updates
    VALUES = "values"
    MESSAGES = "messages"

    # Execution events
    METADATA = "metadata"
    ERROR = "error"
    END = "end"

    # Interrupts
    INTERRUPT = "interrupt"


class StreamEvent(BaseModel):
    """An event in the execution stream."""

    event: StreamEventType
    data: Any = None
    run_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def to_sse(self) -> str:
        """Convert to SSE format."""
        import json

        data_str = json.dumps(self.data) if self.data is not None else "{}"
        return f"event: {self.event.value}\ndata: {data_str}\n\n"
