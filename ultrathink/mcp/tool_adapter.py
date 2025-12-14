"""MCP tool adapter.

This module converts MCP tools to LangChain tools.
"""

from typing import Any, Dict, Optional, Type

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, create_model


def create_mcp_tool(
    tool_info: Dict[str, Any],
    call_func: Any,
    server_name: str = "unknown",
) -> StructuredTool:
    """Create a LangChain tool from MCP tool info.

    Args:
        tool_info: MCP tool definition
        call_func: Async function to call the MCP tool
        server_name: Name of the MCP server

    Returns:
        LangChain StructuredTool
    """
    tool_name = tool_info.get("name", "unknown")
    description = tool_info.get("description", "MCP tool")
    input_schema = tool_info.get("inputSchema", {})

    # Create input model from schema
    InputModel = _create_input_model(tool_name, input_schema)

    # Create the tool function
    async def call_mcp_tool(**kwargs: Any) -> str:
        """Call the MCP tool."""
        try:
            result = await call_func(server_name, tool_name, kwargs)
            return _format_result(result)
        except Exception as e:
            return f"Error calling MCP tool {tool_name}: {e}"

    def sync_call_mcp_tool(**kwargs: Any) -> str:
        """Sync wrapper for MCP tool."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()

        return loop.run_until_complete(call_mcp_tool(**kwargs))

    # Create unique tool name with server prefix
    full_name = f"mcp__{server_name}__{tool_name}"

    return StructuredTool.from_function(
        name=full_name,
        description=f"[MCP: {server_name}] {description}",
        func=sync_call_mcp_tool,
        coroutine=call_mcp_tool,
        args_schema=InputModel,
    )


def _create_input_model(name: str, schema: Dict[str, Any]) -> Type[BaseModel]:
    """Create a Pydantic model from JSON schema.

    Args:
        name: Model name
        schema: JSON schema

    Returns:
        Pydantic model class
    """
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    field_definitions = {}
    for prop_name, prop_schema in properties.items():
        prop_type = _json_type_to_python(prop_schema.get("type", "string"))
        description = prop_schema.get("description", "")

        if prop_name in required:
            field_definitions[prop_name] = (prop_type, ...)
        else:
            field_definitions[prop_name] = (Optional[prop_type], None)

    model_name = f"MCPInput_{name.replace('-', '_')}"
    return create_model(model_name, **field_definitions)


def _json_type_to_python(json_type: str) -> type:
    """Convert JSON schema type to Python type.

    Args:
        json_type: JSON schema type string

    Returns:
        Python type
    """
    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    return type_map.get(json_type, Any)


def _format_result(result: Any) -> str:
    """Format MCP tool result to string.

    Args:
        result: Raw result from MCP tool

    Returns:
        Formatted string
    """
    if isinstance(result, str):
        return result
    if hasattr(result, "content"):
        return str(result.content)
    if hasattr(result, "text"):
        return str(result.text)
    return str(result)
