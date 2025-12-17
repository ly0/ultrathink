"""Todo storage and utilities for Ultrathink.

This module provides file-based todo management for tracking tasks
between turns. Todos are stored at `~/.ultrathink/todos/<project>/todos.json`.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError


TodoStatus = Literal["pending", "in_progress", "completed"]
TodoPriority = Literal["high", "medium", "low"]


class TodoItem(BaseModel):
    """Represents a single todo entry."""

    id: str = Field(description="Unique identifier for the todo item")
    content: str = Field(description="Task description")
    status: TodoStatus = Field(
        default="pending", description="Current state: pending, in_progress, completed"
    )
    priority: TodoPriority = Field(default="medium", description="Priority: high|medium|low")
    created_at: Optional[float] = Field(default=None, description="Unix timestamp when created")
    updated_at: Optional[float] = Field(default=None, description="Unix timestamp when updated")

    # 层级和反思相关字段
    parent_id: Optional[str] = Field(default=None, description="Parent task ID for subtasks")
    result_summary: Optional[str] = Field(default=None, description="Brief summary of execution result")
    is_complex: bool = Field(default=False, description="Mark as complex task (requires reflection)")

    model_config = ConfigDict(extra="ignore")


MAX_TODOS = 200


def _project_hash(project_root: Path) -> str:
    """Create a short hash for the project path."""
    path_str = str(project_root.resolve())
    return hashlib.sha256(path_str.encode()).hexdigest()[:12]


def _storage_path(project_root: Optional[Path], ensure_dir: bool) -> Path:
    """Return the todo storage path, optionally ensuring the directory exists."""
    root = project_root or Path.cwd()
    base_dir = Path.home() / ".ultrathink" / "todos"
    project_dir = base_dir / _project_hash(root)

    if ensure_dir:
        project_dir.mkdir(parents=True, exist_ok=True)

    return project_dir / "todos.json"


def validate_todos(
    todos: Sequence[TodoItem], max_items: int = MAX_TODOS
) -> Tuple[bool, Optional[str]]:
    """Basic validation for a todo list."""
    if len(todos) > max_items:
        return False, f"Too many todos; limit is {max_items}."

    ids = [todo.id for todo in todos]
    duplicate_ids = {id_ for id_ in ids if ids.count(id_) > 1}
    if duplicate_ids:
        return False, f"Duplicate todo IDs found: {sorted(duplicate_ids)}"

    in_progress = [todo for todo in todos if todo.status == "in_progress"]
    if len(in_progress) > 1:
        return False, "Only one todo can be marked in_progress at a time."

    empty_contents = [todo.id for todo in todos if not todo.content.strip()]
    if empty_contents:
        return False, f"Todos require content. Empty content for IDs: {sorted(empty_contents)}"

    return True, None


def load_todos(project_root: Optional[Path] = None) -> List[TodoItem]:
    """Load todos from disk."""
    path = _storage_path(project_root, ensure_dir=False)
    if not path.exists():
        return []

    try:
        raw = json.loads(path.read_text())
    except Exception:
        return []

    todos: List[TodoItem] = []
    for item in raw:
        try:
            todos.append(TodoItem(**item))
        except ValidationError:
            continue

    return todos


def save_todos(todos: Sequence[TodoItem], project_root: Optional[Path] = None) -> None:
    """Persist todos to disk."""
    path = _storage_path(project_root, ensure_dir=True)
    path.write_text(json.dumps([todo.model_dump() for todo in todos], indent=2))


def set_todos(
    todos: Sequence[TodoItem],
    project_root: Optional[Path] = None,
) -> List[TodoItem]:
    """Validate, normalize, and persist the provided todos."""
    ok, message = validate_todos(todos)
    if not ok:
        raise ValueError(message or "Invalid todos.")

    existing = {todo.id: todo for todo in load_todos(project_root)}
    now = time.time()

    normalized: List[TodoItem] = []
    for todo in todos:
        previous = existing.get(todo.id)
        normalized.append(
            todo.model_copy(
                update={
                    "created_at": previous.created_at if previous else todo.created_at or now,
                    "updated_at": now,
                }
            )
        )

    save_todos(normalized, project_root)
    return list(normalized)


def clear_todos(project_root: Optional[Path] = None) -> None:
    """Remove all todos."""
    save_todos([], project_root)


PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def get_next_actionable(todos: Sequence[TodoItem]) -> Optional[TodoItem]:
    """Return the next todo to work on, considering priority.

    Priority order: high > medium > low.
    Within same priority, prefers in_progress over pending.
    For hierarchical todos, prefers leaf tasks (those without pending children).
    """
    for status in ("in_progress", "pending"):
        candidates = [t for t in todos if t.status == status]
        if not candidates:
            continue

        # Sort by priority (high > medium > low)
        candidates.sort(key=lambda t: PRIORITY_ORDER.get(t.priority, 1))

        for todo in candidates:
            # If this task has pending children, skip it and work on children first
            children = get_children(todo.id, todos)
            pending_children = [c for c in children if c.status in ("pending", "in_progress")]
            if pending_children:
                # Recursively find actionable child (considering priority)
                child_actionable = get_next_actionable(pending_children)
                if child_actionable:
                    return child_actionable
            return todo
    return None


def get_children(parent_id: str, todos: Sequence[TodoItem]) -> List[TodoItem]:
    """Get all direct children of a task."""
    return [todo for todo in todos if todo.parent_id == parent_id]


def get_root_todos(todos: Sequence[TodoItem]) -> List[TodoItem]:
    """Get all top-level tasks (those without a parent)."""
    return [todo for todo in todos if todo.parent_id is None]


def has_children(task_id: str, todos: Sequence[TodoItem]) -> bool:
    """Check if a task has any children."""
    return any(todo.parent_id == task_id for todo in todos)


def is_parent_completable(parent_id: str, todos: Sequence[TodoItem]) -> bool:
    """Check if all children of a parent are completed."""
    children = get_children(parent_id, todos)
    if not children:
        return True
    return all(child.status == "completed" for child in children)


def should_reflect(task: TodoItem, todos: Sequence[TodoItem]) -> bool:
    """Determine if reflection is needed after completing a task.

    Reflection is triggered for:
    - Tasks marked as complex
    - Tasks that have children (parent tasks)
    """
    if task.is_complex:
        return True
    if has_children(task.id, todos):
        return True
    return False


def insert_todo(
    todo: TodoItem,
    todos: List[TodoItem],
    position: str = "last",
    project_root: Optional[Path] = None,
) -> Tuple[bool, str, List[TodoItem]]:
    """Insert a single todo at the specified position.

    Args:
        todo: The TodoItem to insert
        todos: Current list of todos
        position: Where to insert - "first", "last", "next", or "after:<task_id>"
        project_root: Project root for storage

    Returns:
        Tuple of (success, message, updated_todos)
    """
    # Check for duplicate ID
    existing_ids = {t.id for t in todos}
    if todo.id in existing_ids:
        return False, f"Task ID '{todo.id}' already exists.", todos

    # Set timestamps
    now = time.time()
    todo = todo.model_copy(update={"created_at": now, "updated_at": now})

    # Determine insert position
    if position == "first":
        insert_index = 0
    elif position == "last":
        insert_index = len(todos)
    elif position == "next":
        # Insert after the current in_progress task, or at the end
        in_progress_idx = None
        for i, t in enumerate(todos):
            if t.status == "in_progress":
                in_progress_idx = i
                break
        insert_index = (in_progress_idx + 1) if in_progress_idx is not None else len(todos)
    elif position.startswith("after:"):
        target_id = position[6:]
        target_idx = None
        for i, t in enumerate(todos):
            if t.id == target_id:
                target_idx = i
                break
        if target_idx is None:
            return False, f"Target task '{target_id}' not found.", todos
        insert_index = target_idx + 1
    else:
        return False, f"Invalid position '{position}'. Use first, last, next, or after:<task_id>.", todos

    # Insert the todo
    new_todos = list(todos)
    new_todos.insert(insert_index, todo)

    # Validate
    ok, message = validate_todos(new_todos)
    if not ok:
        return False, message or "Validation failed.", todos

    # Save
    save_todos(new_todos, project_root)
    return True, f"Task '{todo.id}' inserted successfully.", new_todos


def delete_todo(
    task_id: str,
    todos: List[TodoItem],
    project_root: Optional[Path] = None,
) -> Tuple[bool, str, List[TodoItem]]:
    """Delete a single todo by ID.

    Args:
        task_id: ID of the task to delete
        todos: Current list of todos
        project_root: Project root for storage

    Returns:
        Tuple of (success, message, updated_todos)

    Note:
        Cannot delete a task that has children. Must delete children first.
    """
    # Find the task
    task_index = None
    for i, t in enumerate(todos):
        if t.id == task_id:
            task_index = i
            break

    if task_index is None:
        return False, f"Task '{task_id}' not found.", todos

    # Check for children
    if has_children(task_id, todos):
        children = get_children(task_id, todos)
        child_ids = [c.id for c in children]
        return False, f"Cannot delete task '{task_id}' because it has children: {child_ids}. Delete children first.", todos

    # Remove the task
    new_todos = [t for t in todos if t.id != task_id]

    # Save
    save_todos(new_todos, project_root)
    return True, f"Task '{task_id}' deleted successfully.", new_todos


def summarize_todos(todos: Sequence[TodoItem]) -> dict:
    """Return simple statistics for a todo collection."""
    return {
        "total": len(todos),
        "by_status": {
            "pending": len([t for t in todos if t.status == "pending"]),
            "in_progress": len([t for t in todos if t.status == "in_progress"]),
            "completed": len([t for t in todos if t.status == "completed"]),
        },
        "by_priority": {
            "high": len([t for t in todos if t.priority == "high"]),
            "medium": len([t for t in todos if t.priority == "medium"]),
            "low": len([t for t in todos if t.priority == "low"]),
        },
    }


def format_todo_summary(todos: Sequence[TodoItem]) -> str:
    """Create a concise summary string for use in tool outputs."""
    stats = summarize_todos(todos)
    summary = (
        f"Todos updated (total {stats['total']}; "
        f"{stats['by_status']['pending']} pending, "
        f"{stats['by_status']['in_progress']} in progress, "
        f"{stats['by_status']['completed']} completed)."
    )

    next_item = get_next_actionable(todos)
    if next_item:
        summary += f" Next: {next_item.content} (id: {next_item.id}, status: {next_item.status})."
    elif stats["total"] == 0:
        summary += " No todos stored yet."

    return summary


def format_todo_lines(todos: Sequence[TodoItem], hierarchical: bool = True) -> List[str]:
    """Return human-readable todo lines with optional hierarchical display.

    Args:
        todos: List of todo items
        hierarchical: If True, display with indentation for parent-child relationships

    Returns:
        List of formatted strings
    """
    status_marker = {
        "completed": "[x]",
        "in_progress": "[>]",
        "pending": "[ ]",
    }
    priority_marker = {
        "high": "!",
        "medium": "",
        "low": "~",
    }

    def format_single(todo: TodoItem, indent: int = 0) -> str:
        marker = status_marker.get(todo.status, "[ ]")
        pri = priority_marker.get(todo.priority, "")
        prefix = "  " * indent
        line = f"{prefix}{marker}{pri} {todo.content} (id: {todo.id})"
        # Add result summary for completed tasks
        if todo.status == "completed" and todo.result_summary:
            line += f" → {todo.result_summary}"
        return line

    if not hierarchical:
        return [format_single(todo) for todo in todos]

    # Build hierarchical output
    lines = []
    todo_map = {todo.id: todo for todo in todos}

    def format_with_children(todo: TodoItem, indent: int = 0) -> None:
        lines.append(format_single(todo, indent))
        children = get_children(todo.id, todos)
        for child in children:
            format_with_children(child, indent + 1)

    # Start with root todos
    for todo in todos:
        if todo.parent_id is None:
            format_with_children(todo)

    return lines
