"""Subagent definition loader.

This module loads custom subagent definitions from files.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ultrathink.subagents.definitions import (
    AgentLocation,
    SubagentDefinition,
    get_builtin_subagents,
)


def load_subagent_from_yaml(path: Path, location: AgentLocation) -> Optional[SubagentDefinition]:
    """Load a subagent definition from a YAML file.

    Args:
        path: Path to YAML file
        location: Source location (user or project)

    Returns:
        SubagentDefinition if valid, None otherwise
    """
    try:
        with open(path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            return None

        return SubagentDefinition(
            name=data.get("name", path.stem),
            description=data.get("description", "Custom subagent"),
            system_prompt=data.get("system_prompt", ""),
            tools=data.get("tools", []),
            model=data.get("model"),
            location=location,
        )

    except Exception:
        return None


def load_subagent_from_markdown(path: Path, location: AgentLocation) -> Optional[SubagentDefinition]:
    """Load a subagent definition from a Markdown file.

    Expected format:
    ---
    name: agent-name
    description: When to use this agent
    tools:
      - tool1
      - tool2
    ---

    System prompt content here...

    Args:
        path: Path to Markdown file
        location: Source location

    Returns:
        SubagentDefinition if valid, None otherwise
    """
    try:
        content = path.read_text()

        # Check for YAML frontmatter
        if not content.startswith("---"):
            return None

        # Split frontmatter and content
        parts = content.split("---", 2)
        if len(parts) < 3:
            return None

        frontmatter = yaml.safe_load(parts[1])
        system_prompt = parts[2].strip()

        if not isinstance(frontmatter, dict):
            return None

        return SubagentDefinition(
            name=frontmatter.get("name", path.stem),
            description=frontmatter.get("description", "Custom subagent"),
            system_prompt=system_prompt,
            tools=frontmatter.get("tools", []),
            model=frontmatter.get("model"),
            location=location,
        )

    except Exception:
        return None


def load_subagents_from_directory(
    directory: Path,
    location: AgentLocation,
) -> List[SubagentDefinition]:
    """Load all subagent definitions from a directory.

    Args:
        directory: Directory to search
        location: Source location

    Returns:
        List of loaded subagent definitions
    """
    agents = []

    if not directory.exists():
        return agents

    # Load YAML files
    for yaml_file in directory.glob("*.yaml"):
        agent = load_subagent_from_yaml(yaml_file, location)
        if agent:
            agents.append(agent)

    for yml_file in directory.glob("*.yml"):
        agent = load_subagent_from_yaml(yml_file, location)
        if agent:
            agents.append(agent)

    # Load Markdown files
    for md_file in directory.glob("*.md"):
        agent = load_subagent_from_markdown(md_file, location)
        if agent:
            agents.append(agent)

    return agents


def load_all_subagents(
    project_path: Optional[Path] = None,
    include_builtin: bool = True,
) -> List[SubagentDefinition]:
    """Load all subagent definitions from all sources.

    Loads from:
    1. Built-in definitions
    2. User definitions (~/.ultrathink/agents/)
    3. Project definitions (.ultrathink/agents/)

    Args:
        project_path: Project directory
        include_builtin: Whether to include built-in agents

    Returns:
        List of all subagent definitions
    """
    agents = []

    # Built-in agents
    if include_builtin:
        agents.extend(get_builtin_subagents())

    # User agents
    user_agents_dir = Path.home() / ".ultrathink" / "agents"
    agents.extend(load_subagents_from_directory(user_agents_dir, AgentLocation.USER))

    # Project agents
    if project_path:
        project_agents_dir = project_path / ".ultrathink" / "agents"
        agents.extend(load_subagents_from_directory(project_agents_dir, AgentLocation.PROJECT))

    return agents


def get_subagents_for_agent(
    project_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Get subagent definitions as dicts for create_deep_agent.

    Args:
        project_path: Project directory

    Returns:
        List of subagent configuration dicts
    """
    agents = load_all_subagents(project_path)
    return [agent.to_dict() for agent in agents]
