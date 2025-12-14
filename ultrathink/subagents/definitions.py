"""Built-in subagent definitions.

This module provides default subagent configurations for common tasks.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentLocation(str, Enum):
    """Source location of agent definition."""

    BUILTIN = "builtin"
    USER = "user"
    PROJECT = "project"


@dataclass
class SubagentDefinition:
    """Definition of a subagent.

    Attributes:
        name: Unique identifier for the subagent
        description: When to use this subagent
        system_prompt: Custom system prompt for the subagent
        tools: List of tool names available to this subagent
        model: Optional model override
        location: Where this definition came from
    """

    name: str
    description: str
    system_prompt: str
    tools: List[str] = field(default_factory=list)
    model: Optional[str] = None
    location: AgentLocation = AgentLocation.BUILTIN

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for deepagent."""
        result = {
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
        }
        if self.tools:
            result["tools"] = self.tools
        if self.model:
            result["model"] = self.model
        return result


# Built-in subagent definitions

EXPLORE_AGENT = SubagentDefinition(
    name="explore",
    description=(
        "Fast agent for exploring codebases. Use for file searches, "
        "code pattern searches, and answering questions about the codebase structure."
    ),
    system_prompt="""You are a codebase exploration specialist. Your job is to quickly
find files, search code patterns, and understand project structure.

Guidelines:
- Use glob to find files by pattern (e.g., "**/*.py")
- Use grep to search for code patterns
- Use read_file to examine file contents
- Be thorough but efficient - search multiple patterns if needed
- Return concise, actionable findings

Focus on answering the specific question asked. Don't over-explore.""",
    tools=["glob", "grep", "read_file", "ls"],
)

RESEARCH_AGENT = SubagentDefinition(
    name="research",
    description=(
        "Agent for deep research requiring multiple searches and comprehensive analysis. "
        "Use when you need to understand complex topics or gather extensive information."
    ),
    system_prompt="""You are an expert researcher. Conduct thorough research on the given topic.

Guidelines:
- Break down complex topics into searchable components
- Use multiple search strategies to gather comprehensive information
- Verify information from multiple sources when possible
- Cite your sources and provide references
- Synthesize findings into clear, actionable insights

Be thorough but focused on the specific research question.""",
    tools=["glob", "grep", "read_file", "write_file"],
)

CODE_REVIEW_AGENT = SubagentDefinition(
    name="code-review",
    description=(
        "Agent for reviewing code changes and suggesting improvements. "
        "Use for pull request reviews, security audits, and code quality checks."
    ),
    system_prompt="""You are a senior code reviewer with expertise in software quality.

Review code for:
- Bugs and logic errors
- Security vulnerabilities (XSS, SQL injection, etc.)
- Performance issues
- Code style and readability
- Missing error handling
- Test coverage gaps

Provide feedback in this format:
1. **Summary**: Overall assessment
2. **Critical Issues**: Must-fix problems
3. **Suggestions**: Improvements to consider
4. **Positive Notes**: Good patterns to highlight

Reference specific line numbers when possible.""",
    tools=["read_file", "grep", "glob"],
)

REFACTOR_AGENT = SubagentDefinition(
    name="refactor",
    description=(
        "Agent for planning and executing code refactoring. "
        "Use for restructuring code, improving architecture, or applying design patterns."
    ),
    system_prompt="""You are a refactoring specialist. Plan and execute code improvements
while maintaining functionality.

Guidelines:
- Understand the existing code thoroughly before changing it
- Make incremental, testable changes
- Preserve behavior - refactoring shouldn't change functionality
- Follow existing code conventions
- Document significant architectural decisions

Before refactoring:
1. Identify the code to change
2. Understand its current behavior
3. Plan the refactoring steps
4. Execute changes carefully
5. Verify the result""",
    tools=["read_file", "edit_file", "glob", "grep"],
)

TEST_AGENT = SubagentDefinition(
    name="test",
    description=(
        "Agent for writing and analyzing tests. "
        "Use for creating test cases, improving coverage, or debugging test failures."
    ),
    system_prompt="""You are a testing specialist focused on software quality assurance.

Guidelines:
- Understand the testing framework used in the project
- Write clear, focused test cases
- Cover edge cases and error conditions
- Use descriptive test names
- Mock external dependencies appropriately

For new tests:
1. Identify what needs testing
2. Determine appropriate test types (unit, integration, etc.)
3. Write clear test cases
4. Ensure good coverage of the target code

For test failures:
1. Understand why the test is failing
2. Determine if it's a test bug or code bug
3. Fix appropriately""",
    tools=["read_file", "write_file", "edit_file", "glob", "grep", "execute"],
)


# Default subagents to include
DEFAULT_SUBAGENTS = [
    EXPLORE_AGENT,
    RESEARCH_AGENT,
    CODE_REVIEW_AGENT,
    REFACTOR_AGENT,
    TEST_AGENT,
]


def get_builtin_subagents() -> List[SubagentDefinition]:
    """Get list of built-in subagent definitions.

    Returns:
        List of SubagentDefinition objects
    """
    return list(DEFAULT_SUBAGENTS)


def get_subagent_by_name(name: str) -> Optional[SubagentDefinition]:
    """Get a subagent definition by name.

    Args:
        name: Subagent name

    Returns:
        SubagentDefinition if found, None otherwise
    """
    for agent in DEFAULT_SUBAGENTS:
        if agent.name == name:
            return agent
    return None


def get_subagents_as_dicts() -> List[Dict[str, Any]]:
    """Get subagent definitions as dictionaries for deepagent.

    Returns:
        List of subagent configuration dicts
    """
    return [agent.to_dict() for agent in DEFAULT_SUBAGENTS]
