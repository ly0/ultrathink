"""Base classes and utilities for slash commands.

This module provides the infrastructure for implementing slash commands
in the Ultrathink CLI.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type


@dataclass
class SlashCommand:
    """Definition of a slash command.

    Attributes:
        name: Command name (without the slash)
        description: Short description of what the command does
        handler: Function to call when command is invoked
        aliases: Alternative names for the command
        usage: Usage example
        hidden: Whether to hide from help listing
    """

    name: str
    description: str
    handler: Callable[[Any, str], None]
    aliases: List[str] = field(default_factory=list)
    usage: Optional[str] = None
    hidden: bool = False


# Registry of all slash commands
_command_registry: Dict[str, SlashCommand] = {}


def register_command(command: SlashCommand) -> SlashCommand:
    """Register a slash command.

    Args:
        command: The command to register

    Returns:
        The registered command
    """
    _command_registry[command.name] = command
    for alias in command.aliases:
        _command_registry[alias] = command
    return command


def get_slash_command(name: str) -> Optional[SlashCommand]:
    """Get a slash command by name.

    Args:
        name: Command name (without the slash)

    Returns:
        The command if found, None otherwise
    """
    return _command_registry.get(name.lower())


def list_slash_commands() -> List[SlashCommand]:
    """Get all registered slash commands.

    Returns:
        List of unique commands (not including aliases)
    """
    seen = set()
    commands = []
    for cmd in _command_registry.values():
        if cmd.name not in seen and not cmd.hidden:
            commands.append(cmd)
            seen.add(cmd.name)
    return sorted(commands, key=lambda c: c.name)


# Built-in command handlers

def _help_handler(ui: Any, arg: str) -> None:
    """Show help information."""
    from rich.table import Table

    table = Table(title="Available Commands", show_header=True)
    table.add_column("Command", style="cyan")
    table.add_column("Description", style="white")

    for cmd in list_slash_commands():
        cmd_str = f"/{cmd.name}"
        if cmd.aliases:
            cmd_str += f" ({', '.join('/' + a for a in cmd.aliases)})"
        table.add_row(cmd_str, cmd.description)

    ui.console.print(table)


def _clear_handler(ui: Any, arg: str) -> None:
    """Clear conversation history."""
    ui._clear_conversation()


def _exit_handler(ui: Any, arg: str) -> None:
    """Exit the application."""
    ui._should_exit = True
    ui.console.print("[yellow]Goodbye![/yellow]")


def _history_handler(ui: Any, arg: str) -> None:
    """Show conversation history."""
    ui._show_history()


def _stats_handler(ui: Any, arg: str) -> None:
    """Show session statistics."""
    ui._show_stats()


def _compact_handler(ui: Any, arg: str) -> None:
    """Compact conversation history."""
    ui._compact_history()


def _model_handler(ui: Any, arg: str) -> None:
    """Show or change the current model."""
    from ultrathink.core.config import get_model_string

    if not arg:
        current = get_model_string()
        ui.console.print(f"[cyan]Current model:[/cyan] {current}")
    else:
        ui.model = arg
        ui._agent = None  # Reset agent to use new model
        ui.console.print(f"[green]Model changed to:[/green] {arg}")


def _verbose_handler(ui: Any, arg: str) -> None:
    """Toggle verbose mode."""
    ui.verbose = not ui.verbose
    status = "enabled" if ui.verbose else "disabled"
    ui.console.print(f"[cyan]Verbose mode:[/cyan] {status}")


# Register built-in commands
register_command(SlashCommand(
    name="help",
    description="Show available commands",
    handler=_help_handler,
    aliases=["h", "?"],
))

register_command(SlashCommand(
    name="clear",
    description="Clear conversation history",
    handler=_clear_handler,
    aliases=["c"],
))

register_command(SlashCommand(
    name="exit",
    description="Exit Ultrathink",
    handler=_exit_handler,
    aliases=["quit", "q"],
))

register_command(SlashCommand(
    name="history",
    description="Show conversation history",
    handler=_history_handler,
))

register_command(SlashCommand(
    name="stats",
    description="Show session statistics",
    handler=_stats_handler,
))

register_command(SlashCommand(
    name="compact",
    description="Compact conversation history",
    handler=_compact_handler,
))

register_command(SlashCommand(
    name="model",
    description="Show or change the current model",
    handler=_model_handler,
    usage="/model [provider:model]",
))

register_command(SlashCommand(
    name="verbose",
    description="Toggle verbose mode",
    handler=_verbose_handler,
    aliases=["v"],
))
