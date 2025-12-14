"""SDK client for programmatic use of Ultrathink.

This module provides a Python API for using Ultrathink in scripts and applications.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from ultrathink.core.session import ConversationSession


@dataclass
class UltrathinkOptions:
    """Options for configuring the Ultrathink client.

    Attributes:
        model: Model identifier (e.g., 'anthropic:claude-sonnet-4-20250514')
        base_url: Custom API base URL
        safe_mode: Whether to enable permission checks
        verbose: Whether to enable verbose output
        cwd: Working directory
        system_prompt: Custom system prompt
        tools: Additional tools to provide
        subagents: Custom subagent definitions
    """

    model: Optional[str] = None
    base_url: Optional[str] = None
    safe_mode: bool = True
    verbose: bool = False
    cwd: Optional[Path] = None
    system_prompt: Optional[str] = None
    tools: List[Any] = field(default_factory=list)
    subagents: Optional[List[Dict[str, Any]]] = None


@dataclass
class QueryResult:
    """Result of a query.

    Attributes:
        content: The assistant's response content
        messages: Full message history
        tool_calls: List of tool calls made
        tokens_in: Input tokens used
        tokens_out: Output tokens generated
    """

    content: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0


class UltrathinkClient:
    """Client for programmatic use of Ultrathink.

    Provides a Python API for:
    - Single queries
    - Streaming responses
    - Multi-turn conversations

    Example:
        ```python
        async with UltrathinkClient() as client:
            result = await client.query("Explain this code")
            print(result.content)
        ```
    """

    def __init__(self, options: Optional[UltrathinkOptions] = None) -> None:
        """Initialize the client.

        Args:
            options: Client configuration options
        """
        self.options = options or UltrathinkOptions()
        self.session = ConversationSession()
        self._agent = None
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure the agent is initialized."""
        if self._initialized:
            return

        from ultrathink.core.agent_factory import create_ultrathink_agent

        self._agent = await create_ultrathink_agent(
            model=self.options.model,
            tools=self.options.tools if self.options.tools else None,
            system_prompt=self.options.system_prompt,
            subagents=self.options.subagents,
            safe_mode=self.options.safe_mode,
            verbose=self.options.verbose,
            session=self.session,
            cwd=self.options.cwd,
            base_url=self.options.base_url,
        )
        self._initialized = True

    async def query(self, prompt: str) -> QueryResult:
        """Execute a query and return the result.

        Args:
            prompt: The user's prompt

        Returns:
            QueryResult with the response
        """
        await self._ensure_initialized()

        # Add to session
        self.session.add_message("user", prompt)

        # Get messages for agent
        messages = [{"role": m.role, "content": m.content} for m in self.session.messages]

        # Invoke agent
        result = await self._agent.ainvoke({"messages": messages})

        # Extract response
        content = ""
        if "messages" in result and result["messages"]:
            final_message = result["messages"][-1]
            if hasattr(final_message, "content"):
                content = final_message.content

        # Add to session
        if content:
            self.session.add_message("assistant", content)

        return QueryResult(
            content=content,
            messages=[m.to_dict() for m in self.session.messages],
        )

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Execute a query and stream the response.

        Args:
            prompt: The user's prompt

        Yields:
            Response chunks as they arrive
        """
        await self._ensure_initialized()

        # Add to session
        self.session.add_message("user", prompt)

        # Get messages for agent
        messages = [{"role": m.role, "content": m.content} for m in self.session.messages]

        # Stream response
        full_content = ""

        async for event in self._agent.astream_events(
            {"messages": messages},
            version="v2",
        ):
            event_type = event.get("event", "")

            if event_type == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    full_content += chunk.content
                    yield chunk.content

            elif event_type == "on_chain_end":
                output = event.get("data", {}).get("output", {})
                if "messages" in output and output["messages"]:
                    final_message = output["messages"][-1]
                    if hasattr(final_message, "content"):
                        full_content = final_message.content

        # Add to session
        if full_content:
            self.session.add_message("assistant", full_content)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.session.clear()

    async def close(self) -> None:
        """Close the client and clean up resources."""
        # Clean up MCP if used
        try:
            from ultrathink.mcp.runtime import _runtime

            if _runtime:
                await _runtime.cleanup()
        except Exception:
            pass

        self._agent = None
        self._initialized = False

    async def __aenter__(self) -> "UltrathinkClient":
        """Async context manager entry."""
        await self._ensure_initialized()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


async def query(
    prompt: str,
    model: Optional[str] = None,
    cwd: Optional[Path] = None,
    **kwargs: Any,
) -> str:
    """Execute a single query (convenience function).

    Args:
        prompt: The user's prompt
        model: Model to use
        cwd: Working directory
        **kwargs: Additional options

    Returns:
        The assistant's response
    """
    options = UltrathinkOptions(model=model, cwd=cwd, **kwargs)

    async with UltrathinkClient(options) as client:
        result = await client.query(prompt)
        return result.content
