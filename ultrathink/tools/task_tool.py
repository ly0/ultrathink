"""Task tool for delegating work to subagents."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from rich.console import Console


class TaskInput(BaseModel):
    """Input schema for task tool."""

    name: str = Field(description="The subagent name to use (e.g., 'explore', 'research')")
    task: str = Field(description="Detailed description of the work to delegate")


# Module-level console for subagent output
_console: Optional[Console] = None

# Module-level registry for subagent read-only status
# This is used by the parallel executor to determine if task calls can be parallelized
_subagent_readonly_registry: Dict[str, bool] = {}


def get_subagent_readonly_map() -> Dict[str, bool]:
    """Get the subagent read-only registry for parallel execution."""
    return _subagent_readonly_registry


# Subagent context management constants
SUBAGENT_RECURSION_LIMIT = 50  # Lower recursion limit for subagents (vs 10000 for main)
TASK_REMINDER_INTERVAL = 5  # Remind subagent of original task every N tool calls
SUBAGENT_MESSAGE_WINDOW = 20  # Keep only last N messages to prevent context overflow


def set_task_console(console: Console) -> None:
    """Set the console instance for task tool output."""
    global _console
    _console = console


def get_task_console() -> Optional[Console]:
    """Get the console instance for task tool output."""
    return _console


async def execute_single_subagent(
    name: str,
    task: str,
    subagent_def: Dict[str, Any],
    model: Any,
    working_dir: Path,
    verbose: bool = False,
    index: Optional[int] = None,
    console: Optional[Console] = None,
) -> str:
    """Execute a single subagent task with streaming output.

    This is the core execution function for subagent task delegation.

    Args:
        name: Subagent name
        task: Task description
        subagent_def: Subagent definition dict
        model: The LLM model to use
        working_dir: Current working directory
        verbose: Whether to print debug info
        index: Optional index for parallel task identification (e.g., 0, 1, 2)
        console: Console for rendering output (uses module-level console if None)

    Returns:
        The subagent's final response content
    """
    if verbose:
        index_str = f"-{index}" if index is not None else ""
        print(f"Delegating to subagent '{name}{index_str}': {task[:100]}...")

    try:
        # Import here to avoid circular imports
        from ultrathink.core.agent_factory import create_ultrathink_agent
        from ultrathink.tools.filesystem import create_filesystem_tools

        # Get tools for this subagent (filesystem tools only, NO task tool)
        subagent_tools = create_filesystem_tools(working_dir)

        # IMPORTANT: Verify subagent cannot recursively call task tool
        # This prevents infinite recursion where subagents spawn more subagents
        assert not any(t.name == "task" for t in subagent_tools), \
            "task tool should not be available to subagents - this would cause infinite recursion"

        # Enhance system prompt with the original task embedded at the end
        # This helps the subagent stay focused on the task even after many tool calls
        base_prompt = subagent_def.get("system_prompt", "")
        task_preview = task[:500] + "..." if len(task) > 500 else task
        enhanced_prompt = f"""{base_prompt}

=== CURRENT TASK (IMPORTANT - STAY FOCUSED) ===
{task_preview}

IMPORTANT REMINDERS:
- Complete THIS SPECIFIC TASK and return a concise report
- Do not deviate from the task or explore unrelated areas
- If you've gathered enough information, summarize and finish
- Maximum {SUBAGENT_RECURSION_LIMIT} tool calls allowed - be efficient
"""

        # Create a temporary agent for this subagent (async)
        # Use lower recursion limit and message window to prevent context overflow
        subagent = await create_ultrathink_agent(
            model=model,
            tools=subagent_tools,
            system_prompt=enhanced_prompt,
            subagents=None,  # Don't pass subagents to avoid recursion
            recursion_limit=SUBAGENT_RECURSION_LIMIT,  # Lower limit for subagents
            max_message_window=SUBAGENT_MESSAGE_WINDOW,  # Trim old messages to prevent overflow
        )

        # Get console for rendering (use provided or module-level)
        render_console = console or _console

        # If no console, fall back to simple invoke without rendering
        if render_console is None:
            result = await subagent.ainvoke({"messages": [{"role": "user", "content": task}]})
            if "messages" in result and result["messages"]:
                last_message = result["messages"][-1]
                if hasattr(last_message, "content"):
                    return last_message.content
                return str(last_message)
            return str(result)

        # Import rendering functions
        from ultrathink.cli.ui.message_renderer import (
            render_subagent_thinking,
            render_subagent_tool_use,
        )

        # Stream execution with rendering
        final_content = ""
        thinking_content = ""
        tool_call_count = 0  # Track tool calls for task reminder

        async for event in subagent.astream_events(
            {"messages": [{"role": "user", "content": task}]},
            version="v2",
        ):
            event_type = event.get("event", "")

            if event_type == "on_chat_model_stream":
                # Accumulate thinking content
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "additional_kwargs"):
                    reasoning = chunk.additional_kwargs.get("reasoning_chunk", "")
                    if reasoning:
                        thinking_content += reasoning

            elif event_type == "on_tool_start":
                # Render accumulated thinking before tool call
                if thinking_content:
                    render_subagent_thinking(render_console, thinking_content, name, index)
                    thinking_content = ""

                # Render tool call with subagent prefix
                tool_name = event.get("name", "unknown")
                tool_input = event.get("data", {}).get("input", {})
                render_subagent_tool_use(render_console, tool_name, tool_input, "call", name, index)

            elif event_type == "on_tool_end":
                tool_call_count += 1

                # Optionally render tool result (only in verbose mode)
                if verbose:
                    tool_name = event.get("name", "unknown")
                    output = event.get("data", {}).get("output", "")
                    render_subagent_tool_use(render_console, tool_name, output, "result", name, index)

                # Show task reminder every N tool calls
                if tool_call_count % TASK_REMINDER_INTERVAL == 0:
                    from rich.panel import Panel
                    task_preview = task[:100] + "..." if len(task) > 100 else task
                    index_str = f"-{index}" if index is not None else ""
                    render_console.print(Panel(
                        f"[dim]Progress: {tool_call_count} tool calls completed[/dim]\n"
                        f"[bold]Original task:[/bold] {task_preview}",
                        title=f"[yellow]\\[{name}{index_str}] Task Reminder[/yellow]",
                        border_style="yellow",
                    ))

            elif event_type == "on_chain_end":
                # Extract final response
                output = event.get("data", {}).get("output", {})
                if "messages" in output and output["messages"]:
                    final_message = output["messages"][-1]
                    if hasattr(final_message, "content") and final_message.content:
                        final_content = final_message.content
                    # Also check for thinking in final message
                    if not thinking_content and hasattr(final_message, "additional_kwargs"):
                        thinking_content = final_message.additional_kwargs.get(
                            "reasoning_content", ""
                        ) or final_message.additional_kwargs.get(
                            "reasoning_chunk", ""
                        )

        # Render any remaining thinking content
        if thinking_content:
            render_subagent_thinking(render_console, thinking_content, name, index)

        return final_content or "(no response from subagent)"

    except Exception as e:
        import traceback

        if verbose:
            traceback.print_exc()
        index_str = f"-{index}" if index is not None else ""
        return f"Error executing subagent '{name}{index_str}': {e}"


def create_task_tool(
    subagents: List[Dict[str, Any]],
    model: Any,
    cwd: Optional[Path] = None,
    verbose: bool = False,
) -> StructuredTool:
    """Create a task tool for delegating work to subagents.

    Args:
        subagents: List of subagent definitions
        model: The LLM model to use for subagents
        cwd: Current working directory
        verbose: Whether to print debug info

    Returns:
        A StructuredTool that can delegate tasks to subagents
    """
    working_dir = cwd or Path.cwd()

    # Build subagent lookup map
    subagent_map = {sa["name"]: sa for sa in subagents}

    async def execute_task(name: str, task: str) -> str:
        """Execute a task using the specified subagent with streaming output."""
        if name not in subagent_map:
            available = ", ".join(subagent_map.keys())
            return f"Error: Unknown subagent '{name}'. Available: {available}"

        subagent_def = subagent_map[name]

        return await execute_single_subagent(
            name=name,
            task=task,
            subagent_def=subagent_def,
            model=model,
            working_dir=working_dir,
            verbose=verbose,
            index=None,  # Single task, no index
            console=None,  # Use module-level console
        )

    def sync_execute_task(name: str, task: str) -> str:
        """Synchronous wrapper for execute_task."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, execute_task(name, task))
                    return future.result()
            else:
                return loop.run_until_complete(execute_task(name, task))
        except RuntimeError:
            return asyncio.run(execute_task(name, task))

    # Build description with available subagents
    agent_descriptions = []
    for sa in subagents:
        agent_descriptions.append(f"- {sa['name']}: {sa.get('description', 'No description')}")
    agents_list = "\n".join(agent_descriptions)

    description = f"""Delegate a task to a specialized subagent.

Available subagents:
{agents_list}

Use this tool when you need to:
- Explore the codebase structure
- Research complex topics
- Perform specialized tasks

The subagent will work autonomously and return a concise report."""

    # Update module-level registry with read-only mapping for parallel execution
    global _subagent_readonly_registry
    _subagent_readonly_registry = {
        sa["name"]: sa.get("read_only", False)
        for sa in subagents
    }

    return StructuredTool.from_function(
        name="task",
        description=description,
        func=sync_execute_task,
        coroutine=execute_task,
        args_schema=TaskInput,
    )
