"""Session management for Ultrathink.

This module handles conversation state and message history.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel


MessageRole = Literal["user", "assistant", "system", "tool"]


@dataclass
class Message:
    """A single message in the conversation."""

    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LangChain."""
        return {"role": self.role, "content": self.content}

    def to_tuple(self) -> tuple[str, str]:
        """Convert to tuple format for deepagent."""
        return (self.role, self.content)


@dataclass
class ToolCall:
    """Record of a tool call."""

    tool_name: str
    tool_input: Dict[str, Any]
    tool_output: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: Optional[float] = None
    error: Optional[str] = None


class SessionStats(BaseModel):
    """Statistics for a session."""

    total_messages: int = 0
    user_messages: int = 0
    assistant_messages: int = 0
    tool_calls: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost_usd: float = 0.0
    start_time: datetime = datetime.now()
    last_activity: datetime = datetime.now()


class ConversationSession:
    """Manages a conversation session."""

    def __init__(self, session_id: Optional[str] = None) -> None:
        self.session_id = session_id or str(uuid4())
        self.messages: List[Message] = []
        self.tool_calls: List[ToolCall] = []
        self.stats = SessionStats()
        self._context: Dict[str, Any] = {}
        self.summary: Optional[str] = None  # Summarized conversation history

    def add_message(
        self,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Add a message to the conversation."""
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self.messages.append(message)

        # Update stats
        self.stats.total_messages += 1
        if role == "user":
            self.stats.user_messages += 1
        elif role == "assistant":
            self.stats.assistant_messages += 1
        self.stats.last_activity = datetime.now()

        return message

    def add_tool_call(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Optional[str] = None,
        duration_ms: Optional[float] = None,
        error: Optional[str] = None,
    ) -> ToolCall:
        """Record a tool call."""
        tool_call = ToolCall(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            duration_ms=duration_ms,
            error=error,
        )
        self.tool_calls.append(tool_call)
        self.stats.tool_calls += 1
        self.stats.last_activity = datetime.now()
        return tool_call

    def update_token_usage(
        self,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Update token usage statistics."""
        self.stats.total_tokens_in += tokens_in
        self.stats.total_tokens_out += tokens_out
        self.stats.total_cost_usd += cost_usd

    def get_messages_for_agent(self) -> List[tuple[str, str]]:
        """Get messages in format suitable for deepagent."""
        return [msg.to_tuple() for msg in self.messages]

    def get_messages_as_dicts(self) -> List[Dict[str, str]]:
        """Get messages as list of dicts."""
        return [msg.to_dict() for msg in self.messages]

    def set_summary(self, summary: str) -> None:
        """Set the summarized conversation context.

        Args:
            summary: The summarized content of previous conversation.
        """
        self.summary = summary

    def clear_summary(self) -> None:
        """Clear the summarized conversation context."""
        self.summary = None

    def get_messages_with_summary(self) -> List[Dict[str, str]]:
        """Get messages with summary prepended as system context.

        If a summary exists from previous compaction, it will be prepended
        as a system message to provide context to the model.

        Returns:
            List of message dictionaries with summary prepended if exists.
        """
        messages = []
        if self.summary:
            messages.append({
                "role": "system",
                "content": f"[Previous conversation summary]\n{self.summary}"
            })
        for m in self.messages:
            messages.append({"role": m.role, "content": m.content})
        return messages

    def get_last_user_message(self) -> Optional[Message]:
        """Get the most recent user message."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg
        return None

    def get_last_assistant_message(self) -> Optional[Message]:
        """Get the most recent assistant message."""
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg
        return None

    def clear(self) -> None:
        """Clear all messages, tool calls, and summary."""
        self.messages.clear()
        self.tool_calls.clear()
        self.stats = SessionStats()
        self.summary = None

    def set_context(self, key: str, value: Any) -> None:
        """Set a context value."""
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context value."""
        return self._context.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session to dictionary."""
        return {
            "session_id": self.session_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "stats": self.stats.model_dump(),
            "context": self._context,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationSession":
        """Deserialize session from dictionary."""
        session = cls(session_id=data.get("session_id"))
        for msg_data in data.get("messages", []):
            session.add_message(
                role=msg_data["role"],
                content=msg_data["content"],
            )
        if "stats" in data:
            session.stats = SessionStats(**data["stats"])
        session._context = data.get("context", {})
        return session

    def __len__(self) -> int:
        return len(self.messages)

    def __repr__(self) -> str:
        return f"ConversationSession(id={self.session_id[:8]}..., messages={len(self.messages)})"
