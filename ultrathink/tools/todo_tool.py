"""Todo management tools for Ultrathink.

These tools allow the model to create and manage a structured task list
for tracking progress on complex, multi-step tasks.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from ultrathink.utils.todo import (
    TodoItem,
    analyze_todos,
    delete_todo,
    format_todo_lines,
    format_todo_summary,
    get_children,
    get_next_actionable,
    has_children,
    insert_todo,
    load_todos,
    save_todos,
    set_todos,
    should_reflect,
    summarize_todos,
    validate_todos,
)


def create_todo_tools(cwd: Optional[Path] = None) -> List:
    """Create todo management tools.

    Args:
        cwd: Current working directory (project root)

    Returns:
        List of todo tools
    """
    working_dir = cwd or Path.cwd()

    @tool
    def write_todos(todos: List[Dict[str, Any]]) -> str:
        """Create or replace the entire todo list for the current session.

        IMPORTANT: This tool REPLACES the entire todo list. Any tasks not included
        in the list will be deleted. Use update_todo to modify individual tasks
        without affecting others.

        When to use:
        - Creating a new todo list at the start of a complex task
        - Completely reorganizing or resetting the task list
        - User provides a new set of tasks that replaces existing ones

        When NOT to use:
        - Updating status of a single task (use update_todo instead)
        - Marking a task as completed (use complete_task instead)
        - Single, straightforward tasks
        - Trivial tasks that can be done in 1-2 steps

        Args:
            todos: List of todo items (REPLACES existing list). Each item should have:
                - id (str): Unique identifier for the task
                - content (str): Task description
                - status (str): "pending", "in_progress", or "completed"
                - priority (str, optional): "high", "medium", or "low" (default: "medium")
                - parent_id (str, optional): Parent task ID for subtasks
                - result_summary (str, optional): Brief summary of execution result
                - is_complex (bool, optional): Mark as complex task requiring reflection

        Returns:
            Summary of the updated todo list with hierarchical display
        """
        try:
            # Convert dicts to TodoItem objects
            todo_items = []
            for item in todos:
                todo_items.append(
                    TodoItem(
                        id=str(item.get("id", "")),
                        content=str(item.get("content", "")),
                        status=item.get("status", "pending"),
                        priority=item.get("priority", "medium"),
                        parent_id=item.get("parent_id"),
                        result_summary=item.get("result_summary"),
                        is_complex=item.get("is_complex", False),
                    )
                )

            # Validate before saving
            ok, message = validate_todos(todo_items)
            if not ok:
                return f"Error: {message}"

            # Save and get normalized list
            updated = set_todos(todo_items, working_dir)

            # Format output with hierarchical display
            summary = format_todo_summary(updated)
            lines = format_todo_lines(updated, hierarchical=True)
            return "\n".join([summary, "", *lines]) if lines else summary

        except Exception as e:
            return f"Error updating todos: {e}"

    @tool
    def read_todos(
        status: Optional[str] = None,
        next_only: bool = False,
        limit: int = 0,
        with_reflection: bool = False,
    ) -> str:
        """Read the todo list with optional filters.

        Use this to check current progress, find the next task to work on,
        or review completed tasks.

        Args:
            status: Filter by status - "pending", "in_progress", "completed",
                    or comma-separated values like "pending,in_progress".
                    Leave empty for all todos.
            next_only: If True, return only the next actionable todo
                       (in_progress first, then pending)
            limit: Maximum number of todos to return (0 = all)
            with_reflection: If True, include reflection prompts for plan review

        Returns:
            Formatted todo list with summary
        """
        try:
            all_todos = load_todos(working_dir)

            if not all_todos:
                return "No todos stored yet. Use write_todos to create tasks."

            # Filter by status if specified
            filtered = all_todos
            if status:
                allowed = {s.strip() for s in status.split(",")}
                filtered = [todo for todo in all_todos if todo.status in allowed]

            # Get next actionable
            next_todo = get_next_actionable(all_todos)

            # If next_only, just return that
            if next_only:
                if next_todo:
                    result = (
                        f"Next actionable: {next_todo.content}\n"
                        f"  id: {next_todo.id}\n"
                        f"  status: {next_todo.status}\n"
                        f"  priority: {next_todo.priority}"
                    )
                    if next_todo.parent_id:
                        result += f"\n  parent: {next_todo.parent_id}"
                    if next_todo.is_complex:
                        result += "\n  [complex task - requires reflection after completion]"
                    return result
                return "No actionable todos (none pending or in_progress)."

            # Apply limit
            display = filtered
            if limit > 0:
                display = display[:limit]

            # Format output with hierarchical display
            summary = format_todo_summary(filtered)
            lines = format_todo_lines(display, hierarchical=True)

            if next_todo:
                next_info = f"Next to work on: {next_todo.content} (id: {next_todo.id})"
            else:
                next_info = "No actionable todos remaining."

            output = [summary, next_info, "", *lines]

            # Add reflection prompt if requested
            if with_reflection:
                output.extend([
                    "",
                    "--- Reflection Prompt ---",
                    "Review your todo list and consider:",
                    "1. Are there new tasks to add based on what you learned?",
                    "2. Do any existing tasks need to be modified or split?",
                    "3. Should any task priorities be adjusted?",
                    "4. Are there tasks that are no longer needed?",
                ])

            return "\n".join(output)

        except Exception as e:
            return f"Error reading todos: {e}"

    @tool
    def complete_task(task_id: str, result_summary: str = "") -> str:
        """Mark a task as completed and record the result.

        Use this tool after finishing a task to:
        - Mark the task as completed
        - Record a brief summary of what was accomplished
        - Get the next task to work on
        - Receive reflection prompts if needed

        Args:
            task_id: The ID of the task to mark as completed
            result_summary: Brief summary of the execution result (1 sentence)

        Returns:
            Confirmation with next task and optional reflection prompt
        """
        try:
            all_todos = load_todos(working_dir)

            if not all_todos:
                return "Error: No todos found."

            # Find the task
            task_index = None
            for i, todo in enumerate(all_todos):
                if todo.id == task_id:
                    task_index = i
                    break

            if task_index is None:
                return f"Error: Task with id '{task_id}' not found."

            task = all_todos[task_index]

            if task.status == "completed":
                return f"Task '{task_id}' is already completed."

            # Update the task
            all_todos[task_index] = task.model_copy(
                update={
                    "status": "completed",
                    "result_summary": result_summary if result_summary else None,
                }
            )

            # Save
            save_todos(all_todos, working_dir)

            # Check if reflection is needed
            needs_reflection = should_reflect(task, all_todos)

            # Get next actionable task
            next_todo = get_next_actionable(all_todos)

            # Build response
            output = [f"Completed: {task.content}"]
            if result_summary:
                output.append(f"Result: {result_summary}")
            output.append("")

            # Summary
            stats = summarize_todos(all_todos)
            output.append(
                f"Progress: {stats['by_status']['completed']}/{stats['total']} tasks completed"
            )

            # Next task
            if next_todo:
                output.append(f"\nNext task: {next_todo.content} (id: {next_todo.id})")
            else:
                output.append("\nAll tasks completed!")

            # Reflection prompt for complex tasks
            if needs_reflection:
                output.extend([
                    "",
                    "--- Reflection Required ---",
                    "This was a complex task. Please review:",
                    "1. Did the execution result match expectations?",
                    "2. Were any new subtasks discovered that need to be added?",
                    "3. Do remaining tasks need adjustment based on what you learned?",
                    "",
                    "Use write_todos to update the plan if needed.",
                ])

            # Periodic review reminder (every 3 completed tasks)
            completed_count = stats["by_status"]["completed"]
            if (
                completed_count > 0
                and completed_count % 3 == 0
                and not needs_reflection  # Don't stack with complex task reflection
            ):
                output.extend([
                    "",
                    "--- Periodic Review ---",
                    f"You've completed {completed_count} tasks. Take a moment to review:",
                    "1. Is the remaining plan still valid?",
                    "2. Have priorities changed based on what you've learned?",
                    "3. Are there tasks that should be added or removed?",
                    "",
                    "Use read_todos(with_reflection=True) to review the full list.",
                ])

            return "\n".join(output)

        except Exception as e:
            return f"Error completing task: {e}"

    @tool
    def update_todo(
        task_id: str,
        status: Optional[str] = None,
        content: Optional[str] = None,
        priority: Optional[str] = None,
        result_summary: Optional[str] = None,
    ) -> str:
        """Update a single todo item without affecting other tasks.

        Use this tool to change the status or other properties of a specific task
        without having to rewrite the entire todo list.

        Args:
            task_id: The ID of the task to update
            status: New status - "pending", "in_progress", or "completed"
            content: New task description (optional)
            priority: New priority - "high", "medium", or "low" (optional)
            result_summary: Brief summary of execution result (optional)

        Returns:
            Confirmation with updated task details
        """
        try:
            all_todos = load_todos(working_dir)

            if not all_todos:
                return f"Error: No todos found. Task '{task_id}' does not exist."

            # Find the task
            task_index = None
            for i, todo in enumerate(all_todos):
                if todo.id == task_id:
                    task_index = i
                    break

            if task_index is None:
                return f"Error: Task with id '{task_id}' not found."

            task = all_todos[task_index]

            # Build update dict with only provided fields
            updates = {}
            if status is not None:
                if status not in ("pending", "in_progress", "completed"):
                    return f"Error: Invalid status '{status}'. Must be pending, in_progress, or completed."
                updates["status"] = status
            if content is not None:
                updates["content"] = content
            if priority is not None:
                if priority not in ("high", "medium", "low"):
                    return f"Error: Invalid priority '{priority}'. Must be high, medium, or low."
                updates["priority"] = priority
            if result_summary is not None:
                updates["result_summary"] = result_summary

            if not updates:
                return f"No changes specified for task '{task_id}'."

            # Update the task
            all_todos[task_index] = task.model_copy(update=updates)

            # Validate before saving (check only one in_progress)
            ok, message = validate_todos(all_todos)
            if not ok:
                return f"Error: {message}"

            # Save
            save_todos(all_todos, working_dir)

            updated_task = all_todos[task_index]
            output = [
                f"Updated task '{task_id}':",
                f"  content: {updated_task.content}",
                f"  status: {updated_task.status}",
                f"  priority: {updated_task.priority}",
            ]

            # Add start-time refinement prompt when transitioning to in_progress
            if (
                status == "in_progress"
                and task.status == "pending"
                and not has_children(task_id, all_todos)
            ):
                output.extend([
                    "",
                    "--- Before Starting ---",
                    "Consider whether this task needs to be broken down:",
                    "1. Does this task have 3+ distinct steps?",
                    "2. Would subtasks help track progress better?",
                    "3. Are there dependencies that should be separate tasks?",
                    "",
                    f"If yes, use insert_task with parent_id=\"{task_id}\" to add subtasks.",
                    "DO NOT use write_todos - it will replace your entire plan!",
                    "",
                    "Example:",
                    f"  insert_task(\"{task_id}-a\", \"First step\", parent_id=\"{task_id}\")",
                    f"  insert_task(\"{task_id}-b\", \"Second step\", parent_id=\"{task_id}\")",
                ])

            return "\n".join(output)

        except Exception as e:
            return f"Error updating todo: {e}"

    @tool
    def insert_task(
        task_id: str,
        content: str,
        priority: str = "medium",
        parent_id: Optional[str] = None,
        insert_position: str = "last",
        is_complex: bool = False,
    ) -> str:
        """Insert a single task without rewriting the entire todo list.

        Use this for dynamically adding tasks discovered during execution,
        or when receiving new requests while working on other tasks.

        Args:
            task_id: Unique identifier for the task
            content: Task description
            priority: "high", "medium", or "low" (default: "medium")
                      High priority tasks will be worked on first.
            parent_id: Parent task ID if this is a subtask (optional)
            insert_position: Where to insert the task:
                - "first": At the beginning of the list
                - "last": At the end of the list (default)
                - "next": After the current in_progress task
                - "after:<task_id>": After a specific task
            is_complex: Mark as complex task requiring reflection (default: False)

        Returns:
            Confirmation with updated task list
        """
        try:
            # Validate priority
            if priority not in ("high", "medium", "low"):
                return f"Error: Invalid priority '{priority}'. Must be high, medium, or low."

            # Validate parent_id if provided
            all_todos = load_todos(working_dir)
            if parent_id:
                parent_exists = any(t.id == parent_id for t in all_todos)
                if not parent_exists:
                    return f"Error: Parent task '{parent_id}' not found."

            # Create the todo item
            new_todo = TodoItem(
                id=task_id,
                content=content,
                status="pending",
                priority=priority,
                parent_id=parent_id,
                is_complex=is_complex,
            )

            # Insert
            success, message, updated_todos = insert_todo(
                new_todo, all_todos, insert_position, working_dir
            )

            if not success:
                return f"Error: {message}"

            # Format output
            summary = format_todo_summary(updated_todos)
            next_todo = get_next_actionable(updated_todos)

            output = [
                f"Inserted: {content} (id: {task_id}, priority: {priority})",
                "",
                summary,
            ]

            if next_todo:
                output.append(f"Next task: {next_todo.content} (id: {next_todo.id})")

            return "\n".join(output)

        except Exception as e:
            return f"Error inserting task: {e}"

    @tool
    def delete_task(task_id: str) -> str:
        """Delete a single task from the todo list.

        Use this to remove tasks that are no longer needed.

        IMPORTANT: Cannot delete a task that has children (subtasks).
        You must delete all children first before deleting the parent.

        Args:
            task_id: ID of the task to delete

        Returns:
            Confirmation with updated task list
        """
        try:
            all_todos = load_todos(working_dir)

            if not all_todos:
                return f"Error: No todos found. Task '{task_id}' does not exist."

            # Delete
            success, message, updated_todos = delete_todo(
                task_id, all_todos, working_dir
            )

            if not success:
                return f"Error: {message}"

            # Format output
            if updated_todos:
                summary = format_todo_summary(updated_todos)
                next_todo = get_next_actionable(updated_todos)

                output = [
                    f"Deleted: {task_id}",
                    "",
                    summary,
                ]

                if next_todo:
                    output.append(f"Next task: {next_todo.content} (id: {next_todo.id})")

                return "\n".join(output)
            else:
                return f"Deleted: {task_id}\n\nNo todos remaining."

        except Exception as e:
            return f"Error deleting task: {e}"

    @tool
    def review_todos() -> str:
        """Actively review and analyze the current todo list.

        Use this tool periodically to:
        - Check if the plan is still on track
        - Identify tasks that may need expansion into subtasks
        - Spot potential issues or blockers

        This is useful when:
        - You've completed several tasks and want to reassess the plan
        - You're unsure if remaining tasks are still relevant
        - The system prompts you to review after completing 3 tasks

        Returns:
            Analysis with current status and suggestions for plan adjustments
        """
        try:
            all_todos = load_todos(working_dir)

            if not all_todos:
                return "No todos stored yet. Use write_todos to create tasks."

            # Get statistics
            stats = summarize_todos(all_todos)
            analysis = analyze_todos(all_todos)

            output = [
                "=== Todo List Review ===",
                "",
                f"Progress: {stats['by_status']['completed']}/{stats['total']} completed",
                f"Status: {stats['by_status']['pending']} pending, {stats['by_status']['in_progress']} in progress",
                "",
            ]

            # Current task
            in_progress = [t for t in all_todos if t.status == "in_progress"]
            if in_progress:
                output.append(f"Currently working on: {in_progress[0].content}")
                output.append("")

            # Tasks that may need breakdown
            possibly_complex = analysis.get("possibly_complex", [])
            if possibly_complex:
                output.append("Tasks that may need breakdown:")
                for task in possibly_complex:
                    output.append(f"  - {task.content} (id: {task.id})")
                output.append("")

            # High priority pending
            high_priority = analysis.get("high_priority_pending", [])
            if high_priority:
                output.append(f"High priority tasks remaining: {len(high_priority)}")
                output.append("")

            # Review questions
            output.extend([
                "--- Review Questions ---",
                "1. Are the remaining tasks still relevant to the goal?",
                "2. Do any pending tasks need to be expanded into subtasks?",
                "3. Should any task priorities be adjusted?",
                "4. Have you discovered tasks that should be added?",
                "",
                "Use insert_task, delete_task, or update_todo to adjust the plan.",
            ])

            return "\n".join(output)

        except Exception as e:
            return f"Error reviewing todos: {e}"

    return [write_todos, read_todos, complete_task, update_todo, insert_task, delete_task, review_todos]
