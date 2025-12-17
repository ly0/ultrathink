"""Parallel tool executor for read-only operations.

This module provides a custom ToolNode that executes read-only tool calls
in parallel while ensuring write operations remain sequential.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode


def is_tool_call_readonly(
    tool: Optional[BaseTool],
    tool_call: Dict[str, Any],
) -> bool:
    """Determine if a tool call is read-only based on tool metadata.

    For the 'task' tool, read-only status depends on the subagent being called.
    The subagent read-only mapping is stored in a module-level registry.

    Args:
        tool: The tool being called
        tool_call: The tool call dict with name and args

    Returns:
        True if the tool call is read-only and can be parallelized
    """
    if tool is None:
        return False

    # Check if this is the task tool - get read-only status from registry
    if tool.name == "task":
        from ultrathink.tools.task_tool import get_subagent_readonly_map

        readonly_map = get_subagent_readonly_map()
        subagent_name = tool_call.get("args", {}).get("name")
        if subagent_name and readonly_map:
            return readonly_map.get(subagent_name, False)
        return False

    # For other tools, check if they have a read_only attribute
    if hasattr(tool, "read_only"):
        return tool.read_only

    # Default: assume not read-only (safe default)
    return False


class ParallelToolNode(ToolNode):
    """A ToolNode that executes read-only tool calls in parallel.

    This node extends LangGraph's ToolNode to support parallel execution
    of read-only operations while maintaining sequential execution for
    write operations.

    The parallelization is based on the read_only attribute:
    - For 'task' tool: read_only is inherited from the subagent's read_only attribute
    - For other tools: read_only can be set directly on the tool

    Usage:
        tools = [task_tool, other_tools...]
        node = ParallelToolNode(tools)
        # Use this node instead of the default ToolNode in create_react_agent
    """

    def __init__(
        self,
        tools: Sequence[Union[BaseTool, Callable]],
        *,
        name: str = "tools",
        tags: Optional[List[str]] = None,
        handle_tool_errors: bool = True,
    ):
        """Initialize the ParallelToolNode.

        Args:
            tools: Sequence of tools available for execution
            name: Name of this node
            tags: Optional tags for this node
            handle_tool_errors: Whether to handle tool errors gracefully
        """
        super().__init__(
            tools,
            name=name,
            tags=tags,
            handle_tool_errors=handle_tool_errors,
        )
        # Build tool lookup map for quick access
        self._tools_by_name: Dict[str, BaseTool] = {}
        for tool in tools:
            if isinstance(tool, BaseTool):
                self._tools_by_name[tool.name] = tool

    async def _execute_tool_call(
        self,
        tool_call: Dict[str, Any],
    ) -> ToolMessage:
        """Execute a single tool call and return the result as ToolMessage.

        Args:
            tool_call: Tool call dict with 'name', 'args', and 'id'

        Returns:
            ToolMessage with the tool result
        """
        tool_name = tool_call.get("name", "")
        tool = self._tools_by_name.get(tool_name)

        if tool is None:
            return ToolMessage(
                content=f"Error: Tool '{tool_name}' not found",
                tool_call_id=tool_call.get("id", ""),
            )

        try:
            # Try async execution first
            if hasattr(tool, "ainvoke"):
                result = await tool.ainvoke(tool_call.get("args", {}))
            elif hasattr(tool, "invoke"):
                result = tool.invoke(tool_call.get("args", {}))
            else:
                result = f"Error: Tool '{tool_name}' has no invoke method"
        except Exception as e:
            result = f"Error executing tool '{tool_name}': {e}"

        return ToolMessage(
            content=str(result),
            tool_call_id=tool_call.get("id", ""),
        )

    async def __call__(
        self,
        state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process tool calls with parallel execution for read-only operations.

        This method overrides the parent's __call__ to implement:
        1. Separate tool calls into read-only and write groups
        2. Execute all read-only calls in parallel
        3. Execute write calls sequentially (in order)

        Args:
            state: Current state dict with 'messages' key
            config: Optional config dict

        Returns:
            Updated state dict with tool results added to messages
        """
        messages = state.get("messages", [])
        if not messages:
            return state

        last_message = messages[-1]

        # Check if there are tool calls to process
        if not isinstance(last_message, AIMessage):
            return state

        tool_calls = getattr(last_message, "tool_calls", None)
        if not tool_calls:
            return state

        # Separate read-only and write tool calls
        readonly_calls: List[Dict[str, Any]] = []
        write_calls: List[Dict[str, Any]] = []

        for tc in tool_calls:
            tool = self._tools_by_name.get(tc.get("name", ""))
            if is_tool_call_readonly(tool, tc):
                readonly_calls.append(tc)
            else:
                write_calls.append(tc)

        tool_messages: List[ToolMessage] = []

        # Execute read-only calls in parallel
        if readonly_calls:
            readonly_results = await asyncio.gather(
                *[self._execute_tool_call(tc) for tc in readonly_calls],
                return_exceptions=True,
            )
            for tc, result in zip(readonly_calls, readonly_results):
                if isinstance(result, Exception):
                    tool_messages.append(ToolMessage(
                        content=f"Error: {result}",
                        tool_call_id=tc.get("id", ""),
                    ))
                else:
                    tool_messages.append(result)

        # Execute write calls sequentially (preserve order)
        for tc in write_calls:
            result = await self._execute_tool_call(tc)
            tool_messages.append(result)

        # Return updated state with tool messages added
        return {"messages": tool_messages}


def create_parallel_tool_node(
    tools: Sequence[Union[BaseTool, Callable]],
    **kwargs: Any,
) -> ParallelToolNode:
    """Create a ParallelToolNode for use with LangGraph agents.

    This is a convenience function to create a ParallelToolNode that can
    replace the default ToolNode in create_react_agent.

    Args:
        tools: Sequence of tools to make available
        **kwargs: Additional arguments passed to ParallelToolNode

    Returns:
        Configured ParallelToolNode instance
    """
    return ParallelToolNode(tools, **kwargs)
