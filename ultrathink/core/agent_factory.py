"""Agent factory for Ultrathink.

This module creates and configures deep agents with the appropriate
middleware, tools, and system prompts.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from ultrathink.core.config import (
    ModelProfile,
    ProviderType,
    config_manager,
    get_api_key,
    get_base_url,
    get_model_profile,
    get_model_string,
)
from ultrathink.core.session import ConversationSession
from ultrathink.core.system_prompt import build_system_prompt


def get_default_subagents() -> List[Dict[str, Any]]:
    """Get default subagent definitions."""
    from ultrathink.subagents.definitions import get_subagents_as_dicts

    return get_subagents_as_dicts()


def init_model(
    model_string: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    base_url: Optional[str] = None,
    profile: Optional[ModelProfile] = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Initialize a LangChain chat model.

    Args:
        model_string: Model identifier in format 'provider:model' (e.g., 'anthropic:claude-sonnet-4-20250514')
        temperature: Sampling temperature (overrides profile setting)
        max_tokens: Maximum tokens to generate (overrides profile setting)
        base_url: Custom API base URL (overrides environment/config)
        profile: ModelProfile to use for settings (if not provided, uses default profile)
        **kwargs: Additional model configuration

    Returns:
        Initialized chat model
    """
    # Get profile for default settings
    if profile is None:
        profile = get_model_profile("main")

    # Use profile defaults if not explicitly provided
    if profile:
        if temperature is None:
            temperature = profile.temperature
        if max_tokens is None:
            max_tokens = profile.max_tokens
    else:
        # Fallback defaults
        if temperature is None:
            temperature = 0.7
        if max_tokens is None:
            max_tokens = 8192

    if model_string is None:
        model_string = get_model_string()

    # Parse provider from model string
    if ":" in model_string:
        provider_str, model_name = model_string.split(":", 1)
        provider = ProviderType.from_string(provider_str)
    else:
        # Default to Anthropic
        provider = ProviderType.ANTHROPIC
        model_name = model_string

    # Get API key and base URL
    api_key = get_api_key(provider)
    if base_url is None:
        base_url = get_base_url(provider)

    # Special handling for DeepSeek Reasoner
    if provider == ProviderType.DEEPSEEK and "reasoner" in model_name.lower():
        from ultrathink.models.deepseek_reasoner import ChatDeepSeekReasoner

        return ChatDeepSeekReasoner(
            model=model_name,
            api_key=api_key or "",
            base_url=base_url or "https://api.deepseek.com/v1",
            temperature=0.0,  # Reasoner works best with temperature 0
            max_tokens=max_tokens,
            **kwargs,
        )

    # Special handling for DeepSeek (non-reasoner) - use OpenAI-compatible client
    if provider == ProviderType.DEEPSEEK:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            api_key=api_key or "",
            base_url=base_url or "https://api.deepseek.com/v1",
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    # Build kwargs for model initialization
    model_kwargs = {"temperature": temperature, "max_tokens": max_tokens, **kwargs}

    if provider == ProviderType.ANTHROPIC:
        if api_key:
            model_kwargs["anthropic_api_key"] = api_key
        if base_url:
            model_kwargs["anthropic_api_url"] = base_url
    elif provider == ProviderType.OPENAI:
        if api_key:
            model_kwargs["openai_api_key"] = api_key
        if base_url:
            model_kwargs["openai_api_base"] = base_url
    elif provider == ProviderType.GEMINI:
        if api_key:
            model_kwargs["google_api_key"] = api_key
        # Gemini doesn't typically support custom base URL

    return init_chat_model(f"{provider.value}:{model_name}", **model_kwargs)


async def create_ultrathink_agent(
    model: Optional[Union[str, BaseChatModel]] = None,
    tools: Optional[List[BaseTool]] = None,
    system_prompt: Optional[str] = None,
    subagents: Optional[List[Dict[str, Any]]] = None,
    middleware: Optional[List[Any]] = None,
    safe_mode: bool = True,
    verbose: bool = False,
    session: Optional[ConversationSession] = None,
    cwd: Optional[Path] = None,
    mcp_config: Optional[Dict[str, Any]] = None,
    base_url: Optional[str] = None,
    ui_callback: Optional[Callable] = None,
    ui_multi_callback: Optional[Callable] = None,
    recursion_limit: Optional[int] = None,
    max_message_window: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """Create a configured Ultrathink agent.

    Args:
        model: Model string (e.g., 'anthropic:claude-sonnet-4-20250514') or BaseChatModel instance
        tools: Additional tools to provide to the agent
        system_prompt: Custom system prompt (overrides default)
        subagents: List of subagent definitions
        middleware: Custom middleware to add
        safe_mode: Whether to enable permission checks for dangerous operations
        verbose: Whether to enable verbose logging
        session: Conversation session for context
        cwd: Current working directory
        mcp_config: MCP server configuration
        base_url: Custom API base URL
        ui_callback: Callback function for user interaction (for ask_user tool)
        ui_multi_callback: Callback function for multi-question interaction (for ask_user_multi tool)
        recursion_limit: Maximum recursion depth for agent (default: 10000 for main, 50 for subagents)
        max_message_window: Maximum messages to keep in context window (for subagents, prevents context overflow)
        **kwargs: Additional arguments passed to create_deep_agent

    Returns:
        Configured deep agent instance
    """
    # Initialize model if string provided
    if model is None:
        llm = init_model(base_url=base_url)
    elif isinstance(model, str):
        llm = init_model(model, base_url=base_url)
    else:
        llm = model

    # Get working directory
    working_dir = cwd or Path.cwd()

    # Build subagents list
    all_subagents = subagents or get_default_subagents()

    # Collect tools
    all_tools = list(tools) if tools else []

    # Add custom tools
    from ultrathink.tools import get_custom_tools

    custom_tools = get_custom_tools(
        safe_mode=safe_mode,
        cwd=working_dir,
        ui_callback=ui_callback,
        ui_multi_callback=ui_multi_callback,
    )
    all_tools.extend(custom_tools)

    # Add task tool for subagent delegation
    # Note: parallel execution is handled by the parallel executor based on read_only attribute
    if all_subagents:
        from ultrathink.tools.task_tool import create_task_tool

        task_tool = create_task_tool(
            subagents=all_subagents,
            model=llm,
            cwd=working_dir,
            verbose=verbose,
        )
        all_tools.append(task_tool)

        if verbose:
            print(f"Added task tool with {len(all_subagents)} subagent(s)")

    # Load MCP tools if configured (for all agent types)
    # Must be done BEFORE building system prompt so MCP instructions can be included
    mcp_instructions = None
    if mcp_config:
        try:
            from ultrathink.mcp.runtime import get_mcp_runtime, format_mcp_instructions

            if verbose:
                print(f"MCP: Config found with {len(mcp_config)} server(s)")

            mcp_runtime = await get_mcp_runtime(working_dir)

            if verbose:
                print(f"MCP: Runtime initialized, {len(mcp_runtime.tools)} tool(s) available")

            if mcp_runtime.tools:
                all_tools.extend(mcp_runtime.tools)
                # Format MCP instructions for system prompt
                mcp_instructions = format_mcp_instructions(mcp_runtime)
                if verbose:
                    tool_names = [t.name for t in mcp_runtime.tools[:5]]
                    print(f"MCP: Added tools: {tool_names}...")
        except Exception as e:
            import traceback
            print(f"Warning: Failed to load MCP tools: {e}")
            if verbose:
                traceback.print_exc()

    # Load memory instructions from AGENTS.md files
    from ultrathink.utils.memory import build_memory_instructions

    memory_instructions = build_memory_instructions()
    additional_instructions = []
    if memory_instructions:
        additional_instructions.append(memory_instructions)

    # Build system prompt if not provided
    if system_prompt is None:
        system_prompt = build_system_prompt(
            cwd=working_dir,
            subagents=all_subagents,
            additional_instructions=additional_instructions if additional_instructions else None,
            mcp_instructions=mcp_instructions,
        )

    # Check if we're using DeepSeek Reasoner
    is_deepseek_reasoner = False
    if hasattr(llm, "_llm_type"):
        is_deepseek_reasoner = llm._llm_type == "deepseek-reasoner"

    # Use langgraph for DeepSeek Reasoner to avoid deepagents' buggy tool schemas
    if is_deepseek_reasoner:
        # Add filesystem tools for DeepSeek Reasoner
        from ultrathink.tools.filesystem import create_filesystem_tools

        filesystem_tools = create_filesystem_tools(working_dir)
        all_tools.extend(filesystem_tools)

        if verbose:
            print(f"Creating agent with {len(all_tools)} tools:")
            for tool in all_tools:
                print(f"  - {tool.name}")

        return await _create_langgraph_agent(
            model=llm,
            tools=all_tools,
            system_prompt=system_prompt,
            recursion_limit=recursion_limit,
            max_message_window=max_message_window,
        )

    # Use deepagents for other models
    try:
        from deepagents import create_deep_agent
    except ImportError as e:
        raise ImportError(
            "deepagents is required. Install it with: pip install deepagents"
        ) from e

    # Build middleware list
    # Note: MCP tools are loaded directly into all_tools above,
    # so we don't need MCP middleware here
    all_middleware = list(middleware) if middleware else []

    # Create the agent
    # Note: subagents are handled via custom task tool, not SubAgentMiddleware
    agent = create_deep_agent(
        model=llm,
        tools=all_tools if all_tools else None,
        system_prompt=system_prompt,
        middleware=all_middleware if all_middleware else None,
        **kwargs,
    )

    return agent


async def _create_langgraph_agent(
    model: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: str,
    recursion_limit: Optional[int] = None,
    max_message_window: Optional[int] = None,
    use_parallel_tools: bool = True,
) -> Any:
    """Create a langgraph-based agent with parallel tool execution support.

    This builds a custom ReAct agent that uses ParallelToolNode to execute
    read-only tool calls in parallel while keeping write operations sequential.

    Args:
        model: The chat model
        tools: Tools to provide
        system_prompt: System prompt
        recursion_limit: Maximum recursion depth (default: 10000 for main agents)
        max_message_window: Maximum number of recent messages to keep (for subagents)
        use_parallel_tools: Whether to use ParallelToolNode (default True for main agents)

    Returns:
        Langgraph agent
    """
    from typing import Annotated, TypedDict, Sequence
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages

    from ultrathink.core.parallel_executor import ParallelToolNode

    # Define state schema
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    # Bind tools to model
    model_with_tools = model.bind_tools(tools)

    # Create the tool node (parallel or standard based on flag)
    if use_parallel_tools:
        tool_node = ParallelToolNode(tools)
    else:
        from langgraph.prebuilt import ToolNode
        tool_node = ToolNode(tools)

    # Define the agent node that calls the model
    async def agent_node(state: AgentState) -> dict:
        """Call the model with current messages."""
        messages = list(state["messages"])

        # Apply message window trimming if configured
        if max_message_window is not None and max_message_window > 0:
            messages = _trim_messages(messages, max_message_window)

        # Prepend system message
        full_messages = [SystemMessage(content=system_prompt)] + messages

        response = await model_with_tools.ainvoke(full_messages)
        return {"messages": [response]}

    # Define routing logic
    def should_continue(state: AgentState) -> str:
        """Determine if we should call tools or end."""
        messages = state["messages"]
        last_message = messages[-1]

        # If the model made tool calls, route to tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        # Otherwise, end
        return END

    # Build the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # Set entry point
    workflow.set_entry_point("agent")

    # Add edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        }
    )
    workflow.add_edge("tools", "agent")

    # Compile the graph
    agent = workflow.compile()

    # Use provided recursion limit or default to 10000
    effective_limit = recursion_limit if recursion_limit is not None else 10000
    return _AgentWithConfig(agent, recursion_limit=effective_limit)


def _trim_messages(
    messages: List[Any],
    max_window: int,
) -> List[Any]:
    """Trim messages to keep only recent ones while preserving context.

    Keeps:
    - First human message (original task)
    - Last N messages (tool calls and responses)
    """
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    if len(messages) <= max_window + 1:
        # No trimming needed (+1 for first human message)
        return messages

    # Always keep first human message
    preserved = []
    first_human_idx = None

    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage) and first_human_idx is None:
            preserved.append(msg)
            first_human_idx = i
            break

    if first_human_idx is None:
        # No human message found, just keep last N
        return messages[-max_window:]

    # Keep the last N messages (after first human message)
    remaining_messages = messages[first_human_idx + 1:]
    if len(remaining_messages) > max_window:
        # Add a summary message to indicate trimming occurred
        trimmed_count = len(remaining_messages) - max_window
        summary_msg = AIMessage(content=f"[Context trimmed: {trimmed_count} earlier messages removed to maintain focus]")
        preserved.append(summary_msg)
        preserved.extend(remaining_messages[-max_window:])
    else:
        preserved.extend(remaining_messages)

    return preserved


class _AgentWithConfig:
    """Wrapper for agent that applies default config."""

    def __init__(self, agent: Any, recursion_limit: int = 10000):
        self._agent = agent
        self._config = {"recursion_limit": recursion_limit}

    async def ainvoke(self, inputs: dict, config: dict = None) -> dict:
        """Invoke the agent asynchronously."""
        merged_config = {**self._config, **(config or {})}
        return await self._agent.ainvoke(inputs, config=merged_config)

    def invoke(self, inputs: dict, config: dict = None) -> dict:
        """Invoke the agent synchronously."""
        merged_config = {**self._config, **(config or {})}
        return self._agent.invoke(inputs, config=merged_config)

    async def astream_events(self, inputs: dict, **kwargs) -> Any:
        """Stream events from the agent."""
        config = kwargs.pop("config", None)
        merged_config = {**self._config, **(config or {})}
        async for event in self._agent.astream_events(inputs, config=merged_config, **kwargs):
            yield event

    def stream_events(self, inputs: dict, **kwargs) -> Any:
        """Stream events from the agent."""
        config = kwargs.pop("config", None)
        merged_config = {**self._config, **(config or {})}
        for event in self._agent.stream_events(inputs, config=merged_config, **kwargs):
            yield event


def create_sync_agent(
    model: Optional[Union[str, BaseChatModel]] = None,
    **kwargs: Any,
) -> Any:
    """Create an agent synchronously.

    This is a convenience wrapper for create_ultrathink_agent that
    handles the async initialization.

    Args:
        model: Model string or BaseChatModel instance
        **kwargs: Additional arguments passed to create_ultrathink_agent

    Returns:
        Configured deep agent instance
    """
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            create_ultrathink_agent(model=model, **kwargs)
        )
    finally:
        loop.close()
