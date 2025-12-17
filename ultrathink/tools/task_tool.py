"""Task tool for delegating work to subagents."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class TaskInput(BaseModel):
    """Input schema for task tool."""

    name: str = Field(description="The subagent name to use (e.g., 'explore', 'research')")
    task: str = Field(description="Detailed description of the work to delegate")


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
        """Execute a task using the specified subagent."""
        if name not in subagent_map:
            available = ", ".join(subagent_map.keys())
            return f"Error: Unknown subagent '{name}'. Available: {available}"

        subagent_def = subagent_map[name]

        if verbose:
            print(f"Delegating to subagent '{name}': {task[:100]}...")

        try:
            # Import here to avoid circular imports
            from ultrathink.core.agent_factory import create_ultrathink_agent
            from ultrathink.tools.filesystem import create_filesystem_tools

            # Get tools for this subagent
            subagent_tools = create_filesystem_tools(working_dir)

            # Create a temporary agent for this subagent (async)
            subagent = await create_ultrathink_agent(
                model=model,
                tools=subagent_tools,
                system_prompt=subagent_def.get("system_prompt", ""),
                subagents=None,  # Don't pass subagents to avoid recursion
            )

            # Execute the task
            result = await subagent.ainvoke({"messages": [{"role": "user", "content": task}]})

            # Extract the response
            if "messages" in result and result["messages"]:
                last_message = result["messages"][-1]
                if hasattr(last_message, "content"):
                    return last_message.content
                return str(last_message)
            return str(result)

        except Exception as e:
            import traceback

            if verbose:
                traceback.print_exc()
            return f"Error executing subagent '{name}': {e}"

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

    return StructuredTool.from_function(
        name="task",
        description=description,
        func=sync_execute_task,
        coroutine=execute_task,
        args_schema=TaskInput,
    )
