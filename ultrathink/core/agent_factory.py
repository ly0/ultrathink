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
    return [
        {
            "name": "explore",
            "description": "Fast agent for exploring codebases. Use for file searches, "
            "code searches, and answering questions about the codebase.",
            "system_prompt": (
                "You are a codebase exploration specialist. Your job is to quickly "
                "find files, search code patterns, and understand project structure. "
                "Use glob, grep, and read_file tools efficiently. Return concise findings."
            ),
        },
        {
            "name": "research",
            "description": "Agent for deep research requiring multiple searches "
            "and comprehensive analysis.",
            "system_prompt": (
                "You are an expert researcher. Conduct thorough research on the given topic. "
                "Use multiple sources, verify information, and provide comprehensive findings "
                "with proper citations."
            ),
        },
        {
            "name": "code-review",
            "description": "Agent for reviewing code changes and suggesting improvements.",
            "system_prompt": (
                "You are a senior code reviewer. Analyze code for bugs, security issues, "
                "performance problems, and style violations. Provide actionable feedback "
                "with specific line references."
            ),
        },
    ]


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

    # Build system prompt if not provided
    if system_prompt is None:
        system_prompt = build_system_prompt(
            cwd=working_dir,
            subagents=all_subagents,
        )

    # Collect tools
    all_tools = list(tools) if tools else []

    # Add custom tools
    from ultrathink.tools import get_custom_tools

    custom_tools = get_custom_tools(
        safe_mode=safe_mode,
        cwd=working_dir,
    )
    all_tools.extend(custom_tools)

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

        return await _create_langgraph_agent(
            model=llm,
            tools=all_tools,
            system_prompt=system_prompt,
        )

    # Use deepagents for other models
    try:
        from deepagents import create_deep_agent
    except ImportError as e:
        raise ImportError(
            "deepagents is required. Install it with: pip install deepagents"
        ) from e

    # Build middleware list
    all_middleware = list(middleware) if middleware else []

    # Add MCP middleware if configured
    if mcp_config:
        try:
            from ultrathink.middleware.mcp_integration import MCPMiddleware

            mcp_middleware = MCPMiddleware(
                project_path=working_dir,
                config=mcp_config,
            )
            all_middleware.append(mcp_middleware)
        except ImportError:
            pass  # MCP not available

    # Create the agent
    agent = create_deep_agent(
        model=llm,
        tools=all_tools if all_tools else None,
        system_prompt=system_prompt,
        subagents=all_subagents if all_subagents else None,
        middleware=all_middleware if all_middleware else None,
        **kwargs,
    )

    return agent


async def _create_langgraph_agent(
    model: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: str,
) -> Any:
    """Create a langgraph-based agent for DeepSeek Reasoner.

    This is used instead of deepagents for DeepSeek Reasoner because
    deepagents' tools have schema issues with Callable types.

    Args:
        model: The chat model
        tools: Tools to provide
        system_prompt: System prompt

    Returns:
        Langgraph agent
    """
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import SystemMessage

    # Create agent with system prompt and higher recursion limit
    agent = create_react_agent(
        model,
        tools,
        prompt=SystemMessage(content=system_prompt),
    )

    # Return a wrapper that sets higher recursion limit
    # Set a very high recursion limit - effectively unlimited for practical purposes
    # Each "recursion" is roughly one model call + tool execution
    return _AgentWithConfig(agent, recursion_limit=10000)


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
