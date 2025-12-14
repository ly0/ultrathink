"""Tests for session module."""

import pytest

from ultrathink.core.session import ConversationSession, Message


class TestMessage:
    """Tests for Message dataclass."""

    def test_create_message(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_to_dict(self):
        msg = Message(role="user", content="Hello")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "Hello"}

    def test_to_tuple(self):
        msg = Message(role="assistant", content="Hi there")
        t = msg.to_tuple()
        assert t == ("assistant", "Hi there")


class TestConversationSession:
    """Tests for ConversationSession."""

    def test_create_session(self):
        session = ConversationSession()
        assert len(session) == 0
        assert session.session_id is not None

    def test_create_session_with_id(self):
        session = ConversationSession(session_id="test-123")
        assert session.session_id == "test-123"

    def test_add_message(self):
        session = ConversationSession()
        msg = session.add_message("user", "Hello")
        assert len(session) == 1
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_add_multiple_messages(self):
        session = ConversationSession()
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi!")
        session.add_message("user", "How are you?")
        assert len(session) == 3

    def test_stats_updated(self):
        session = ConversationSession()
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi!")
        assert session.stats.total_messages == 2
        assert session.stats.user_messages == 1
        assert session.stats.assistant_messages == 1

    def test_get_last_user_message(self):
        session = ConversationSession()
        session.add_message("user", "First")
        session.add_message("assistant", "Response")
        session.add_message("user", "Second")
        last = session.get_last_user_message()
        assert last is not None
        assert last.content == "Second"

    def test_get_last_assistant_message(self):
        session = ConversationSession()
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi!")
        session.add_message("user", "Another")
        last = session.get_last_assistant_message()
        assert last is not None
        assert last.content == "Hi!"

    def test_clear(self):
        session = ConversationSession()
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi!")
        session.clear()
        assert len(session) == 0
        assert session.stats.total_messages == 0

    def test_context(self):
        session = ConversationSession()
        session.set_context("key", "value")
        assert session.get_context("key") == "value"
        assert session.get_context("missing") is None
        assert session.get_context("missing", "default") == "default"

    def test_get_messages_for_agent(self):
        session = ConversationSession()
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi!")
        messages = session.get_messages_for_agent()
        assert messages == [("user", "Hello"), ("assistant", "Hi!")]

    def test_to_dict(self):
        session = ConversationSession(session_id="test-123")
        session.add_message("user", "Hello")
        d = session.to_dict()
        assert d["session_id"] == "test-123"
        assert len(d["messages"]) == 1

    def test_from_dict(self):
        data = {
            "session_id": "test-456",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ],
        }
        session = ConversationSession.from_dict(data)
        assert session.session_id == "test-456"
        assert len(session) == 2

    def test_add_tool_call(self):
        session = ConversationSession()
        tool_call = session.add_tool_call(
            tool_name="read_file",
            tool_input={"path": "/test.py"},
            tool_output="file contents",
        )
        assert tool_call.tool_name == "read_file"
        assert session.stats.tool_calls == 1
