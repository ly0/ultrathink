"""Message rendering utilities for the CLI.

This module handles formatting and displaying messages, tool calls,
and other output in the terminal.
"""

from typing import Any, Dict, Optional, Union

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text


def render_message(
    console: Console,
    message: Any,
    role: Optional[str] = None,
) -> None:
    """Render a message to the console.

    Args:
        console: Rich console for output
        message: Message object or string to render
        role: Override role (user/assistant/system)
    """
    # Extract content and role from message object
    if hasattr(message, "content"):
        content = message.content
    else:
        content = str(message)

    if hasattr(message, "role"):
        role = message.role
    elif role is None:
        role = "assistant"

    # Handle different content types
    if isinstance(content, str):
        _render_text_content(console, content, role)
    elif isinstance(content, list):
        # Handle Anthropic-style content blocks
        for block in content:
            if isinstance(block, dict):
                _render_content_block(console, block, role)
            elif isinstance(block, str):
                _render_text_content(console, block, role)
    else:
        _render_text_content(console, str(content), role)


def _render_text_content(console: Console, content: str, role: str) -> None:
    """Render text content with appropriate formatting.

    Args:
        console: Rich console
        content: Text content to render
        role: Message role for styling
    """
    if not content.strip():
        return

    # Style based on role
    if role == "user":
        console.print(Panel(content, title="You", border_style="blue"))
    elif role == "system":
        console.print(Panel(content, title="System", border_style="yellow"))
    else:
        # Assistant messages rendered as markdown
        try:
            md = Markdown(content)
            console.print(md)
        except Exception:
            console.print(content)


def _render_content_block(console: Console, block: Dict[str, Any], role: str) -> None:
    """Render a content block (Anthropic format).

    Args:
        console: Rich console
        block: Content block dictionary
        role: Message role
    """
    block_type = block.get("type", "text")

    if block_type == "text":
        text = block.get("text", "")
        _render_text_content(console, text, role)
    elif block_type == "thinking":
        thinking = block.get("thinking", "")
        if thinking:
            console.print(Panel(
                Text(thinking, style="dim italic"),
                title="[dim]Thinking[/dim]",
                border_style="dim",
            ))
    elif block_type == "tool_use":
        name = block.get("name", "unknown")
        input_data = block.get("input", {})
        render_tool_use(console, name, input_data, "call")
    elif block_type == "tool_result":
        content = block.get("content", "")
        render_tool_use(console, "result", content, "result")


def render_tool_use(
    console: Console,
    tool_name: str,
    data: Any,
    event_type: str = "call",
) -> None:
    """Render a tool call or result.

    Args:
        console: Rich console
        tool_name: Name of the tool
        data: Tool input or output data
        event_type: "call" or "result"
    """
    if event_type == "call":
        # Format tool call
        title = f"[bold cyan]Tool:[/bold cyan] {tool_name}"

        # Format input data
        if isinstance(data, dict):
            lines = []
            for key, value in data.items():
                # Truncate long values
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                lines.append(f"  {key}: {value_str}")
            content = "\n".join(lines) if lines else "(no input)"
        else:
            content = str(data) if data else "(no input)"

        console.print(Panel(
            content,
            title=title,
            border_style="cyan",
            padding=(0, 1),
        ))

    elif event_type == "result":
        # Format tool result
        content = str(data) if data else "(no output)"

        # Truncate very long results
        if len(content) > 500:
            content = content[:500] + f"\n... ({len(content) - 500} more characters)"

        console.print(Panel(
            Text(content, style="dim"),
            title=f"[dim]Result: {tool_name}[/dim]",
            border_style="dim",
            padding=(0, 1),
        ))


def render_error(console: Console, error: Union[str, Exception]) -> None:
    """Render an error message.

    Args:
        console: Rich console
        error: Error message or exception
    """
    if isinstance(error, Exception):
        error_text = f"{type(error).__name__}: {error}"
    else:
        error_text = str(error)

    console.print(Panel(
        Text(error_text, style="bold red"),
        title="[red]Error[/red]",
        border_style="red",
    ))


def render_code(
    console: Console,
    code: str,
    language: str = "python",
    title: Optional[str] = None,
) -> None:
    """Render a code block with syntax highlighting.

    Args:
        console: Rich console
        code: Code to display
        language: Programming language for highlighting
        title: Optional title for the panel
    """
    syntax = Syntax(code, language, theme="monokai", line_numbers=True)
    if title:
        console.print(Panel(syntax, title=title, border_style="green"))
    else:
        console.print(syntax)


def render_progress(
    console: Console,
    message: str,
    completed: int,
    total: int,
) -> None:
    """Render a progress message.

    Args:
        console: Rich console
        message: Progress message
        completed: Number of completed items
        total: Total number of items
    """
    percentage = (completed / total * 100) if total > 0 else 0
    console.print(f"[dim]{message}: {completed}/{total} ({percentage:.0f}%)[/dim]")
