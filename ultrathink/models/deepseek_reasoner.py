"""DeepSeek Reasoner model wrapper with reasoning_content support.

This module provides a custom ChatModel that properly handles DeepSeek Reasoner's
special `reasoning_content` field for tool calls.

DeepSeek Reasoner Requirements:
1. During tool call loops, reasoning_content MUST be preserved and sent back
2. After conversation turn completes, reasoning_content should be discarded
3. The model returns both reasoning_content (thinking) and content (answer)
"""

import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool
from pydantic import Field, PrivateAttr
import openai


# Custom key to store reasoning_content in additional_kwargs
REASONING_CONTENT_KEY = "reasoning_content"


class ChatDeepSeekReasoner(BaseChatModel):
    """Chat model wrapper for DeepSeek Reasoner with reasoning_content support.

    This model handles the special requirements of DeepSeek Reasoner:
    - Preserves reasoning_content during tool call loops
    - Strips reasoning_content for new conversation turns
    - Properly formats messages for the DeepSeek API

    Example:
        ```python
        model = ChatDeepSeekReasoner(
            api_key="your-api-key",
            base_url="https://api.deepseek.com/v1",
        )
        response = model.invoke([HumanMessage(content="Hello")])
        ```
    """

    model: str = Field(default="deepseek-reasoner")
    api_key: str = Field(default="")
    base_url: str = Field(default="https://api.deepseek.com/v1")
    temperature: float = Field(default=0.0)  # Reasoner works best with 0
    max_tokens: int = Field(default=8192)
    streaming: bool = Field(default=True)

    # Private attributes (not serialized)
    _client: Optional[openai.OpenAI] = PrivateAttr(default=None)
    _async_client: Optional[openai.AsyncOpenAI] = PrivateAttr(default=None)
    _bound_tools: List[Dict[str, Any]] = PrivateAttr(default_factory=list)

    model_config = {
        "arbitrary_types_allowed": True,
    }

    @property
    def _llm_type(self) -> str:
        return "deepseek-reasoner"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _get_client(self) -> openai.OpenAI:
        """Get or create sync OpenAI client."""
        if self._client is None:
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def _get_async_client(self) -> openai.AsyncOpenAI:
        """Get or create async OpenAI client."""
        if self._async_client is None:
            self._async_client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._async_client

    def _convert_messages_to_deepseek(
        self,
        messages: List[BaseMessage],
        is_new_turn: bool = False,
    ) -> List[Dict[str, Any]]:
        """Convert LangChain messages to DeepSeek API format.

        Args:
            messages: LangChain messages
            is_new_turn: If True, strip reasoning_content from previous AI messages

        Returns:
            List of messages in DeepSeek API format
        """
        result = []

        for i, msg in enumerate(messages):
            if isinstance(msg, SystemMessage):
                result.append({
                    "role": "system",
                    "content": msg.content,
                })
            elif isinstance(msg, HumanMessage):
                result.append({
                    "role": "user",
                    "content": msg.content,
                })
            elif isinstance(msg, AIMessage):
                api_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.content or "",
                }

                # Check if we need to include reasoning_content
                # Only include if:
                # 1. It's not a new turn (we're in a tool call loop)
                # 2. The message has tool_calls (reasoning is part of this step)
                # 3. There are more messages after this (tool results pending)
                has_more_messages = i < len(messages) - 1
                has_tool_calls = bool(msg.tool_calls)

                reasoning = msg.additional_kwargs.get(REASONING_CONTENT_KEY)
                if reasoning and has_tool_calls and has_more_messages and not is_new_turn:
                    api_msg["reasoning_content"] = reasoning

                # Handle tool calls
                if msg.tool_calls:
                    api_msg["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"]) if isinstance(tc["args"], dict) else tc["args"],
                            },
                        }
                        for tc in msg.tool_calls
                    ]

                result.append(api_msg)

            elif isinstance(msg, ToolMessage):
                result.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content if isinstance(msg.content, str) else json.dumps(msg.content),
                })

        return result

    def _convert_response_to_message(
        self,
        response: Any,
        reasoning_content: Optional[str] = None,
    ) -> AIMessage:
        """Convert DeepSeek API response to LangChain AIMessage.

        Args:
            response: API response choice
            reasoning_content: The reasoning content from the response

        Returns:
            AIMessage with preserved reasoning_content
        """
        message = response.message
        content = message.content or ""

        # Build additional_kwargs
        additional_kwargs: Dict[str, Any] = {}
        if reasoning_content:
            additional_kwargs[REASONING_CONTENT_KEY] = reasoning_content

        # Handle tool calls
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": json.loads(tc.function.arguments) if tc.function.arguments else {},
                })

        return AIMessage(
            content=content,
            tool_calls=tool_calls,
            additional_kwargs=additional_kwargs,
            response_metadata={
                "model": self.model,
                "finish_reason": response.finish_reason,
            },
        )

    def bind_tools(
        self,
        tools: List[Union[BaseTool, Dict[str, Any], Any]],
        *,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> "ChatDeepSeekReasoner":
        """Bind tools to the model.

        Args:
            tools: List of tools to bind
            tool_choice: Optional tool choice constraint
            **kwargs: Additional arguments

        Returns:
            New model instance with bound tools
        """
        formatted_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                # Already formatted
                if "type" in tool and tool["type"] == "function":
                    formatted_tools.append(tool)
                elif "function" in tool:
                    formatted_tools.append(tool)
                else:
                    # Assume it's a function definition dict
                    formatted_tools.append({
                        "type": "function",
                        "function": tool,
                    })
            elif isinstance(tool, BaseTool):
                # Convert BaseTool to OpenAI format
                try:
                    args_schema = tool.args_schema
                    if args_schema is None:
                        schema = {"type": "object", "properties": {}}
                    elif isinstance(args_schema, dict):
                        # MCP tools may have args_schema as dict directly
                        schema = args_schema
                    elif hasattr(args_schema, "model_json_schema"):
                        # Pydantic model
                        schema = args_schema.model_json_schema()
                    else:
                        schema = {"type": "object", "properties": {}}
                except Exception:
                    # Fallback for tools with problematic schemas
                    schema = {"type": "object", "properties": {}}

                formatted_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": schema,
                    },
                })
            elif callable(tool):
                # Handle callable tools using langchain's conversion
                try:
                    from langchain_core.utils.function_calling import convert_to_openai_function
                    formatted_tools.append({
                        "type": "function",
                        "function": convert_to_openai_function(tool),
                    })
                except Exception:
                    # Skip tools that can't be converted
                    pass

        # Create new instance with tools
        new_model = ChatDeepSeekReasoner(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            streaming=self.streaming,
        )
        new_model._bound_tools = formatted_tools
        return new_model

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response synchronously."""
        client = self._get_client()

        # Detect if this is a new conversation turn
        # A new turn starts when the last message is from user (not a tool result)
        is_new_turn = len(messages) > 0 and isinstance(messages[-1], HumanMessage)

        api_messages = self._convert_messages_to_deepseek(messages, is_new_turn=is_new_turn)

        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        if self._bound_tools:
            request_kwargs["tools"] = self._bound_tools

        if stop:
            request_kwargs["stop"] = stop

        response = client.chat.completions.create(**request_kwargs)

        choice = response.choices[0]
        reasoning_content = getattr(choice.message, "reasoning_content", None)

        ai_message = self._convert_response_to_message(choice, reasoning_content)

        return ChatResult(
            generations=[ChatGeneration(message=ai_message)],
            llm_output={
                "token_usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                "model_name": self.model,
            },
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response asynchronously."""
        client = self._get_async_client()

        # Detect if this is a new conversation turn
        is_new_turn = len(messages) > 0 and isinstance(messages[-1], HumanMessage)

        api_messages = self._convert_messages_to_deepseek(messages, is_new_turn=is_new_turn)

        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        if self._bound_tools:
            request_kwargs["tools"] = self._bound_tools

        if stop:
            request_kwargs["stop"] = stop

        response = await client.chat.completions.create(**request_kwargs)

        choice = response.choices[0]
        reasoning_content = getattr(choice.message, "reasoning_content", None)

        ai_message = self._convert_response_to_message(choice, reasoning_content)

        return ChatResult(
            generations=[ChatGeneration(message=ai_message)],
            llm_output={
                "token_usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                "model_name": self.model,
            },
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream responses synchronously."""
        client = self._get_client()

        is_new_turn = len(messages) > 0 and isinstance(messages[-1], HumanMessage)
        api_messages = self._convert_messages_to_deepseek(messages, is_new_turn=is_new_turn)

        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": self.max_tokens,
            "stream": True,
        }

        if self._bound_tools:
            request_kwargs["tools"] = self._bound_tools

        if stop:
            request_kwargs["stop"] = stop

        response = client.chat.completions.create(**request_kwargs)

        reasoning_content = ""
        content = ""
        tool_calls_data: Dict[int, Dict[str, Any]] = {}

        for chunk in response:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Accumulate reasoning content
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                reasoning_content += delta.reasoning_content
                # Yield reasoning as a chunk (optional - for display)
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        additional_kwargs={"reasoning_chunk": delta.reasoning_content},
                    )
                )

            # Accumulate content
            if delta.content:
                content += delta.content
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content=delta.content)
                )

            # Handle tool calls
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_data:
                        tool_calls_data[idx] = {
                            "id": tc.id or "",
                            "name": tc.function.name if tc.function else "",
                            "args": "",
                        }
                    if tc.id:
                        tool_calls_data[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls_data[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls_data[idx]["args"] += tc.function.arguments

        # Yield final chunk with tool calls and reasoning_content
        if tool_calls_data:
            tool_calls = [
                {
                    "id": data["id"],
                    "name": data["name"],
                    "args": json.loads(data["args"]) if data["args"] else {},
                }
                for data in tool_calls_data.values()
            ]

            additional_kwargs = {}
            if reasoning_content:
                additional_kwargs[REASONING_CONTENT_KEY] = reasoning_content

            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_calls=tool_calls,
                    additional_kwargs=additional_kwargs,
                )
            )

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream responses asynchronously."""
        client = self._get_async_client()

        is_new_turn = len(messages) > 0 and isinstance(messages[-1], HumanMessage)
        api_messages = self._convert_messages_to_deepseek(messages, is_new_turn=is_new_turn)

        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": self.max_tokens,
            "stream": True,
        }

        if self._bound_tools:
            request_kwargs["tools"] = self._bound_tools

        if stop:
            request_kwargs["stop"] = stop

        response = await client.chat.completions.create(**request_kwargs)

        reasoning_content = ""
        content = ""
        tool_calls_data: Dict[int, Dict[str, Any]] = {}

        async for chunk in response:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Accumulate reasoning content
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                reasoning_content += delta.reasoning_content
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        additional_kwargs={"reasoning_chunk": delta.reasoning_content},
                    )
                )

            # Accumulate content
            if delta.content:
                content += delta.content
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content=delta.content)
                )

            # Handle tool calls
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_data:
                        tool_calls_data[idx] = {
                            "id": tc.id or "",
                            "name": tc.function.name if tc.function else "",
                            "args": "",
                        }
                    if tc.id:
                        tool_calls_data[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls_data[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls_data[idx]["args"] += tc.function.arguments

        # Yield final chunk with tool calls and reasoning_content
        if tool_calls_data:
            tool_calls = [
                {
                    "id": data["id"],
                    "name": data["name"],
                    "args": json.loads(data["args"]) if data["args"] else {},
                }
                for data in tool_calls_data.values()
            ]

            additional_kwargs = {}
            if reasoning_content:
                additional_kwargs[REASONING_CONTENT_KEY] = reasoning_content

            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_calls=tool_calls,
                    additional_kwargs=additional_kwargs,
                )
            )
