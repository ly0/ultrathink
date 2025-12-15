"""Ask user tool for interactive questions.

This tool allows the agent to ask the user questions during execution.
"""

import json
from typing import Any, Callable, Dict, List, Optional

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


# Multi-question schemas
class Question(BaseModel):
    """A single question in a multi-question interaction."""

    id: str = Field(description="Unique identifier for this question")
    question: str = Field(description="The question text to display")
    options: List[str] = Field(
        default_factory=list,
        description="Optional list of choices for this question",
    )
    required: bool = Field(
        default=True,
        description="Whether this question requires an answer",
    )
    default: Optional[str] = Field(
        default=None,
        description="Default value if user skips",
    )


class AskUserMultiInput(BaseModel):
    """Input schema for ask_user_multi tool."""

    questions: List[Question] = Field(
        description="List of 1-3 questions to ask the user",
        min_length=1,
        max_length=3,
    )
    context: str = Field(
        default="",
        description="Context explaining why you're asking these questions",
    )
    title: str = Field(
        default="Questions",
        description="Title for the question dialog",
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

        loop = asyncio.get_running_loop()
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


def create_ask_user_multi_tool(
    ui_callback: Callable[[List[Dict[str, Any]], str, str], Dict[str, Any]]
) -> StructuredTool:
    """Create an ask_user_multi tool with the given UI callback.

    Args:
        ui_callback: Function that takes (questions, context, title) and returns
                     dict with 'answers', 'confirmed', and 'cancelled' keys

    Returns:
        Configured StructuredTool for asking multiple questions
    """

    def ask_user_multi(
        questions: List[Question],
        context: str = "",
        title: str = "Questions",
    ) -> str:
        """Ask the user multiple questions with a tab-based interface.

        Use this tool when you need to:
        - Gather multiple related pieces of information at once
        - Get clarification on several aspects simultaneously
        - Confirm multiple settings before proceeding

        The user can navigate between questions using Tab/Shift+Tab
        and review all answers before confirming.

        Args:
            questions: List of 1-3 questions (each with id, question, options, required, default)
            context: Optional context explaining the questions
            title: Title for the dialog

        Returns:
            JSON string with status and answers
        """
        questions_dict = [q.model_dump() for q in questions]
        result = ui_callback(questions_dict, context, title)

        if result.get("cancelled"):
            return json.dumps({"status": "cancelled", "answers": {}})

        return json.dumps({
            "status": "confirmed" if result.get("confirmed") else "incomplete",
            "answers": result.get("answers", {}),
        })

    async def aask_user_multi(
        questions: List[Question],
        context: str = "",
        title: str = "Questions",
    ) -> str:
        """Async version of ask_user_multi."""
        import asyncio

        loop = asyncio.get_running_loop()
        questions_dict = [q.model_dump() for q in questions]

        def run_sync():
            result = ui_callback(questions_dict, context, title)
            if result.get("cancelled"):
                return json.dumps({"status": "cancelled", "answers": {}})
            return json.dumps({
                "status": "confirmed" if result.get("confirmed") else "incomplete",
                "answers": result.get("answers", {}),
            })

        return await loop.run_in_executor(None, run_sync)

    return StructuredTool.from_function(
        name="ask_user_multi",
        description="""Ask the user multiple related questions with a tabbed interface.

Use this tool when you need to gather several pieces of information at once.
The user can navigate between questions using Tab/Shift+Tab and review all
answers before confirming.

IMPORTANT: Prefer this over multiple ask_user calls when questions are related.
Limited to 1-3 questions per interaction.

Example usage:
- Gathering project configuration (name, language, framework)
- Collecting API settings (endpoint, key, timeout)
- Getting user preferences (theme, language, notifications)""",
        func=ask_user_multi,
        coroutine=aask_user_multi,
        args_schema=AskUserMultiInput,
    )
