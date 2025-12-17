"""Message rendering utilities for the CLI.

This module handles formatting and displaying messages, tool calls,
and other output in the terminal.
"""

from typing import Any, Dict, List, Optional, Union

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

# Todo tool names for special rendering
TODO_TOOLS = {"write_todos", "read_todos", "update_todo", "complete_task"}

# Subagent color mapping
AGENT_COLORS = {
    "explore": "magenta",
    "research": "blue",
    "code-review": "yellow",
    "refactor": "green",
    "test": "cyan",
}


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


def _get_indent_level(todo_id: str, todos: List[Dict[str, Any]], todo_map: Dict[str, Dict]) -> int:
    """Calculate indent level for a todo based on parent chain."""
    level = 0
    current = todo_map.get(todo_id)
    while current and current.get("parent_id"):
        level += 1
        current = todo_map.get(current["parent_id"])
    return level


def _get_hierarchical_order(todos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reorder todos to show parent-child relationships.

    Returns todos in hierarchical order: parent followed by its children.
    """
    todo_map = {str(t.get("id", "")): t for t in todos}
    result = []
    added = set()

    def add_with_children(todo: Dict[str, Any]) -> None:
        todo_id = str(todo.get("id", ""))
        if todo_id in added:
            return
        added.add(todo_id)
        result.append(todo)
        # Add children
        for t in todos:
            if t.get("parent_id") == todo_id:
                add_with_children(t)

    # Start with root todos (no parent_id)
    for todo in todos:
        if not todo.get("parent_id"):
            add_with_children(todo)

    # Add any orphaned todos (parent not in list)
    for todo in todos:
        todo_id = str(todo.get("id", ""))
        if todo_id not in added:
            add_with_children(todo)

    return result


def _render_todo_table(
    console: Console,
    todos: List[Dict[str, Any]],
    title: str,
    highlight_id: Optional[str] = None,
) -> None:
    """Render a todo list as a beautiful table with hierarchical display.

    Args:
        console: Rich console
        todos: List of todo items
        title: Title for the panel
        highlight_id: Optional task ID to highlight (for update/complete operations)
    """
    if not todos:
        console.print(Panel("[dim]No todos[/dim]", title=title, border_style="cyan"))
        return

    # Status icons and colors
    status_style = {
        "pending": ("[ ]", "white"),
        "in_progress": ("[>]", "yellow bold"),
        "completed": ("[X]", "green"),
    }

    # Priority indicators
    priority_style = {
        "high": ("!", "red bold"),
        "medium": ("", ""),
        "low": ("~", "dim"),
    }

    # Build todo map for indent calculation
    todo_map = {str(t.get("id", "")): t for t in todos}

    # Reorder todos hierarchically
    ordered_todos = _get_hierarchical_order(todos)

    table = Table(
        show_header=False,
        box=None,
        padding=(0, 1),
        expand=False,
    )
    table.add_column("Status", width=3, no_wrap=True)
    table.add_column("Priority", width=1, no_wrap=True)
    table.add_column("Content", no_wrap=True)
    table.add_column("ID", style="dim", no_wrap=True)

    for todo in ordered_todos:
        status = todo.get("status", "pending")
        priority = todo.get("priority", "medium")
        content = todo.get("content", "")
        task_id = str(todo.get("id", ""))
        is_highlighted = highlight_id and task_id == highlight_id

        status_icon, status_color = status_style.get(status, ("[ ]", "white"))
        priority_icon, priority_color = priority_style.get(priority, ("", ""))

        # Calculate indent level
        indent_level = _get_indent_level(task_id, todos, todo_map)
        indent = "  " * indent_level

        # Build content text with appropriate styling
        content_text = Text()
        if indent:
            content_text.append(indent, style="dim")
        if is_highlighted:
            content_text.append("→ ", style="cyan bold")

        if status == "completed":
            content_text.append(content, style="dim strike")
        elif status == "in_progress":
            content_text.append(content, style="yellow bold")
        elif is_highlighted:
            content_text.append(content, style="cyan bold")
        else:
            content_text.append(content)

        table.add_row(
            Text(status_icon, style=status_color),
            Text(priority_icon, style=priority_color),
            content_text,
            task_id[:8] if len(task_id) > 8 else task_id,
        )

    console.print(Panel(table, title=title, border_style="cyan", padding=(0, 0)))


def _get_current_todos_as_dicts() -> List[Dict[str, Any]]:
    """Load current todos and convert to dicts for rendering."""
    try:
        from ultrathink.utils.todo import load_todos

        todos = load_todos()
        return [
            {
                "id": todo.id,
                "content": todo.content,
                "status": todo.status,
                "priority": todo.priority,
                "parent_id": todo.parent_id,
            }
            for todo in todos
        ]
    except Exception:
        return []


def _render_todo_call(console: Console, tool_name: str, data: Any) -> None:
    """Render a todo tool call with beautiful formatting.

    Args:
        console: Rich console
        tool_name: Name of the todo tool
        data: Tool input data
    """
    if tool_name == "write_todos":
        todos = data.get("todos", []) if isinstance(data, dict) else []
        _render_todo_table(console, todos, "[bold cyan]📝 Write Todos[/bold cyan]")

    elif tool_name == "update_todo":
        # Show update details and current todo list
        if isinstance(data, dict):
            task_id = data.get("task_id", "?")
            updates = []
            if "status" in data:
                updates.append(f"status → {data['status']}")
            if "content" in data:
                updates.append(f"content → {data['content']}")
            if "priority" in data:
                updates.append(f"priority → {data['priority']}")
            update_str = ", ".join(updates) if updates else "no changes"

            # Show operation info
            console.print(Panel(
                f"Task [bold]{task_id}[/bold]: {update_str}",
                title="[bold cyan]✏️  Update Todo[/bold cyan]",
                border_style="cyan",
                padding=(0, 1),
            ))

            # Show current todo list with highlighted task
            todos = _get_current_todos_as_dicts()
            if todos:
                _render_todo_table(console, todos, "[dim]Current Todos[/dim]", highlight_id=str(task_id))
        else:
            console.print(Panel(str(data), title="[bold cyan]✏️  Update Todo[/bold cyan]", border_style="cyan"))

    elif tool_name == "complete_task":
        if isinstance(data, dict):
            task_id = data.get("task_id", "?")
            summary = data.get("result_summary", "")
            content = f"Task [bold]{task_id}[/bold]"
            if summary:
                content += f"\nResult: {summary}"

            # Show operation info
            console.print(Panel(
                content,
                title="[bold cyan]✅ Complete Task[/bold cyan]",
                border_style="cyan",
                padding=(0, 1),
            ))

            # Show current todo list with highlighted task
            todos = _get_current_todos_as_dicts()
            if todos:
                _render_todo_table(console, todos, "[dim]Current Todos[/dim]", highlight_id=str(task_id))
        else:
            console.print(Panel(str(data), title="[bold cyan]✅ Complete Task[/bold cyan]", border_style="cyan"))

    elif tool_name == "read_todos":
        # Show filter parameters
        if isinstance(data, dict):
            filters = []
            if data.get("status"):
                filters.append(f"status={data['status']}")
            if data.get("next_only"):
                filters.append("next_only=True")
            if data.get("limit"):
                filters.append(f"limit={data['limit']}")
            filter_str = ", ".join(filters) if filters else "all"
            console.print(Panel(
                f"Filter: {filter_str}",
                title="[bold cyan]📋 Read Todos[/bold cyan]",
                border_style="cyan",
                padding=(0, 1),
            ))

            # Show current todo list
            todos = _get_current_todos_as_dicts()
            if todos:
                _render_todo_table(console, todos, "[dim]Current Todos[/dim]")
        else:
            console.print(Panel("(no filters)", title="[bold cyan]📋 Read Todos[/bold cyan]", border_style="cyan"))


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
    # Special rendering for todo tools
    if tool_name in TODO_TOOLS:
        if event_type == "call":
            _render_todo_call(console, tool_name, data)
            return
        elif event_type == "result":
            # For update_todo and complete_task, show the updated todo list after operation
            if tool_name in ("update_todo", "complete_task"):
                todos = _get_current_todos_as_dicts()
                if todos:
                    _render_todo_table(console, todos, "[bold green]📋 Updated Todos[/bold green]")
            return

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


def render_subagent_thinking(
    console: Console,
    content: str,
    agent_name: str,
    index: Optional[int] = None,
) -> None:
    """Render subagent thinking output with agent name prefix.

    Args:
        console: Rich console
        content: Thinking content
        agent_name: Name of the subagent
        index: Optional index for parallel task identification
    """
    if not content.strip():
        return

    color = AGENT_COLORS.get(agent_name, "magenta")
    # Use \[ to escape brackets so Rich doesn't interpret [agent_name] as a tag
    if index is not None:
        title = f"[{color}]\\[{agent_name}-{index}][/{color}] Thinking"
    else:
        title = f"[{color}]\\[{agent_name}][/{color}] Thinking"
    console.print(Panel(
        Text(content, style="dim italic"),
        title=title,
        border_style=color,
    ))


def render_subagent_tool_use(
    console: Console,
    tool_name: str,
    data: Any,
    event_type: str,
    agent_name: str,
    index: Optional[int] = None,
) -> None:
    """Render subagent tool call with agent name prefix.

    Args:
        console: Rich console
        tool_name: Name of the tool
        data: Tool input or output data
        event_type: "call" or "result"
        agent_name: Name of the subagent
        index: Optional index for parallel task identification
    """
    color = AGENT_COLORS.get(agent_name, "magenta")

    # Build agent label with optional index
    if index is not None:
        agent_label = f"{agent_name}-{index}"
    else:
        agent_label = agent_name

    if event_type == "call":
        # Use \[ to escape brackets so Rich doesn't interpret [agent_name] as a tag
        title = f"[{color}]\\[{agent_label}][/{color}] Tool: {tool_name}"

        # Format input data
        if isinstance(data, dict):
            lines = []
            for key, value in data.items():
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
            border_style=color,
            padding=(0, 1),
        ))

    elif event_type == "result":
        content = str(data) if data else "(no output)"

        if len(content) > 500:
            content = content[:500] + f"\n... ({len(content) - 500} more characters)"

        # Use \[ to escape brackets so Rich doesn't interpret [agent_name] as a tag
        title = f"[dim]\\[{agent_label}] Result: {tool_name}[/dim]"
        console.print(Panel(
            Text(content, style="dim"),
            title=title,
            border_style="dim",
            padding=(0, 1),
        ))
