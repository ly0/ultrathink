"""Ask user tool for interactive questions.

This tool allows the agent to ask the user questions during execution.
"""

from typing import Any, Callable, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class AskUserInput(BaseModel):
    """Input schema for ask_user tool."""

    question: str = Field(description="The question to ask the user")
    options: List[str] = Field(
        default_factory=list,
        description="Optional list of choices to present to the user",
    )
    context: str = Field(
        default="",
        description="Optional context to help the user understand why you're asking",
    )


def create_ask_user_tool(ui_callback: Callable[[str, List[str], str], str]) -> StructuredTool:
    """Create an ask_user tool with the given UI callback.

    Args:
        ui_callback: Function that takes (question, options, context) and returns user's answer

    Returns:
        Configured StructuredTool for asking user questions
    """

    def ask_user(question: str, options: List[str] = None, context: str = "") -> str:
        """Ask the user a question and return their answer.

        Use this tool when you need to:
        - Get clarification on ambiguous instructions
        - Confirm before making potentially destructive changes
        - Let the user choose between multiple valid approaches
        - Gather additional context that wasn't provided

        Args:
            question: The question to ask
            options: Optional list of choices (user can still provide custom answer)
            context: Optional context explaining why you're asking

        Returns:
            The user's response
        """
        if options is None:
            options = []
        return ui_callback(question, options, context)

    async def aask_user(question: str, options: List[str] = None, context: str = "") -> str:
        """Async version of ask_user."""
        import asyncio

        if options is None:
            options = []

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: ui_callback(question, options, context)
        )

    return StructuredTool.from_function(
        name="ask_user",
        description="""Ask the user a question to get clarification or confirmation.

Use this tool when you need to:
- Get clarification on ambiguous instructions
- Confirm before making potentially destructive changes
- Let the user choose between multiple valid approaches
- Gather additional context that wasn't provided

Keep questions concise and specific. Minimize the number of questions you ask.
When possible, batch related questions together.""",
        func=ask_user,
        coroutine=aask_user,
        args_schema=AskUserInput,
    )
