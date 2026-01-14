"""Message models for the API.

Matches the LangGraph SDK message format expected by deep-agents-ui.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Represents a tool call within a message."""

    id: str
    name: str
    args: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[str] = None
    status: Literal["pending", "completed", "error", "interrupted"] = "pending"


class TodoItem(BaseModel):
    """A todo item tracked by the agent."""

    id: str
    content: str
    status: Literal["pending", "in_progress", "completed"] = "pending"
    activeForm: Optional[str] = None
    updated_at: Optional[datetime] = None


class Message(BaseModel):
    """A message in the conversation.

    Matches the LangGraph SDK Message type.
    """

    id: str
    type: Literal["human", "ai", "tool", "system"]
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    additional_kwargs: Dict[str, Any] = Field(default_factory=dict)
    response_metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_langchain_format(self) -> Dict[str, Any]:
        """Convert to LangChain message format."""
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "name": self.name,
            "tool_calls": [tc.model_dump() for tc in self.tool_calls]
            if self.tool_calls
            else None,
            "tool_call_id": self.tool_call_id,
            "additional_kwargs": self.additional_kwargs,
            "response_metadata": self.response_metadata,
        }

    @classmethod
    def from_langchain(cls, msg: Any) -> "Message":
        """Create from a LangChain message object."""
        tool_calls = None
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.get("id", ""),
                    name=tc.get("name", ""),
                    args=tc.get("args", {}),
                )
                for tc in msg.tool_calls
            ]

        msg_type = "ai"
        if hasattr(msg, "type"):
            msg_type = msg.type
        elif msg.__class__.__name__ == "HumanMessage":
            msg_type = "human"
        elif msg.__class__.__name__ == "AIMessage":
            msg_type = "ai"
        elif msg.__class__.__name__ == "ToolMessage":
            msg_type = "tool"
        elif msg.__class__.__name__ == "SystemMessage":
            msg_type = "system"

        return cls(
            id=getattr(msg, "id", "") or "",
            type=msg_type,
            content=msg.content if hasattr(msg, "content") else str(msg),
            name=getattr(msg, "name", None),
            tool_calls=tool_calls,
            tool_call_id=getattr(msg, "tool_call_id", None),
            additional_kwargs=getattr(msg, "additional_kwargs", {}),
            response_metadata=getattr(msg, "response_metadata", {}),
        )
