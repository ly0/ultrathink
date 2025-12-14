"""Rich-based interactive console for Ultrathink.

This module provides the main interactive terminal interface.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from ultrathink import __version__
from ultrathink.core.session import ConversationSession
from ultrathink.cli.ui.message_renderer import render_message, render_tool_use, render_error
from ultrathink.cli.ui.thinking_spinner import ThinkingSpinner


# Prompt toolkit style
PROMPT_STYLE = Style.from_dict({
    "prompt": "cyan bold",
})


# Slash commands with descriptions
SLASH_COMMANDS = {
    "/help": "Show help message",
    "/clear": "Clear conversation history",
    "/history": "Show conversation history",
    "/stats": "Show session statistics",
    "/compact": "Compact conversation history",
    "/models": "Manage model profiles and aliases",
    "/exit": "Exit Ultrathink",
    "/quit": "Exit Ultrathink",
}

# Subcommands for commands that have them
SLASH_SUBCOMMANDS = {
    "models": {
        "list": "List all profiles and aliases",
        "add": "Add a new profile: <name> <provider:model> [--api-key KEY] [--base-url URL] [--alias ALIAS]",
        "update": "Update existing profile: <name> [--api-key KEY] [--base-url URL] [--model MODEL]",
        "remove": "Remove a profile: <name>",
        "alias": "Set alias to profile: <alias> <profile>",
        "unalias": "Remove a custom alias: <alias>",
        "use": "Set main to use profile/alias: <name>",
        "help": "Show /models help",
    },
}


class SlashCommandCompleter(Completer):
    """Completer for slash commands and subcommands."""

    def get_completions(self, document, complete_event):
        """Generate completions for slash commands.

        Args:
            document: The current document
            complete_event: The completion event

        Yields:
            Completion objects
        """
        text = document.text_before_cursor

        # Only complete if the line starts with /
        if not text.startswith("/"):
            return

        # Parse command and arguments
        content = text[1:]  # Remove leading /
        parts = content.split()

        if len(parts) == 0:
            # Just "/", complete commands
            for cmd, description in SLASH_COMMANDS.items():
                cmd_name = cmd[1:]
                yield Completion(
                    cmd_name,
                    start_position=0,
                    display=cmd,
                    display_meta=description,
                )
        elif len(parts) == 1 and not text.endswith(" "):
            # Completing command name (e.g., "/mod" -> "/models")
            word = parts[0]
            for cmd, description in SLASH_COMMANDS.items():
                cmd_name = cmd[1:]
                if cmd_name.startswith(word):
                    yield Completion(
                        cmd_name[len(word):],
                        start_position=0,
                        display=cmd,
                        display_meta=description,
                    )
        else:
            # Completing subcommands
            cmd_name = parts[0].lower()
            if cmd_name in SLASH_SUBCOMMANDS:
                subcommands = SLASH_SUBCOMMANDS[cmd_name]

                if len(parts) == 1 and text.endswith(" "):
                    # After "/models ", show all subcommands
                    for subcmd, desc in subcommands.items():
                        yield Completion(
                            subcmd,
                            start_position=0,
                            display=subcmd,
                            display_meta=desc,
                        )
                elif len(parts) == 2 and not text.endswith(" "):
                    # Completing subcommand (e.g., "/models al" -> "/models alias")
                    word = parts[1]
                    for subcmd, desc in subcommands.items():
                        if subcmd.startswith(word):
                            yield Completion(
                                subcmd[len(word):],
                                start_position=0,
                                display=subcmd,
                                display_meta=desc,
                            )


class UltrathinkUI:
    """Rich-based interactive terminal interface for Ultrathink.

    This class manages the main interaction loop, handling user input,
    agent responses, and tool calls.
    """

    def __init__(
        self,
        safe_mode: bool = True,
        verbose: bool = False,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        cwd: Optional[Path] = None,
    ) -> None:
        """Initialize the UI.

        Args:
            safe_mode: Whether to enable permission checks
            verbose: Whether to show verbose output
            model: Optional model override
            base_url: Optional API base URL override
            cwd: Working directory
        """
        self.console = Console()
        self.safe_mode = safe_mode
        self.verbose = verbose
        self.model = model
        self.base_url = base_url
        self.cwd = cwd or Path.cwd()

        self.session = ConversationSession()
        self._agent = None
        self._should_exit = False

        # Set up prompt history
        history_path = Path.home() / ".ultrathink_history"
        try:
            self._history = FileHistory(str(history_path))
        except Exception:
            self._history = InMemoryHistory()

        self._prompt_session = PromptSession(
            history=self._history,
            style=PROMPT_STYLE,
            completer=SlashCommandCompleter(),
            complete_while_typing=True,
        )

    async def _get_agent(self) -> Any:
        """Lazily initialize and return the agent."""
        if self._agent is None:
            from ultrathink.core.agent_factory import create_ultrathink_agent

            self._agent = await create_ultrathink_agent(
                model=self.model,
                safe_mode=self.safe_mode,
                verbose=self.verbose,
                session=self.session,
                cwd=self.cwd,
                base_url=self.base_url,
            )
        return self._agent

    def run(self) -> None:
        """Start the interactive loop."""
        self._display_welcome()
        asyncio.run(self._main_loop())

    def _display_welcome(self) -> None:
        """Display the welcome banner."""
        welcome_text = Text()
        welcome_text.append("Welcome to ", style="bold")
        welcome_text.append("Ultrathink", style="bold cyan")
        welcome_text.append(f" v{__version__}\n\n", style="dim")
        welcome_text.append("AI-powered coding assistant built on deepagent.\n", style="")
        welcome_text.append("Type your questions. Press ", style="dim")
        welcome_text.append("Ctrl+C", style="dim bold")
        welcome_text.append(" to exit. Type ", style="dim")
        welcome_text.append("/help", style="dim bold")
        welcome_text.append(" for commands.", style="dim")

        self.console.print(Panel(
            welcome_text,
            border_style="cyan",
            padding=(1, 2),
        ))
        self.console.print()

    async def _main_loop(self) -> None:
        """Main interaction loop."""
        while not self._should_exit:
            try:
                # Get user input
                user_input = await self._get_user_input()

                if user_input is None:
                    continue

                user_input = user_input.strip()
                if not user_input:
                    continue

                # Handle slash commands
                if user_input.startswith("/"):
                    self._handle_slash_command(user_input)
                    continue

                # Process the query
                await self._process_query(user_input)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted. Press Ctrl+C again to exit.[/yellow]")
                try:
                    await asyncio.sleep(1)
                except KeyboardInterrupt:
                    self.console.print("\n[yellow]Goodbye![/yellow]")
                    break
            except EOFError:
                self.console.print("\n[yellow]Goodbye![/yellow]")
                break
            except Exception as e:
                render_error(self.console, e)

    async def _get_user_input(self) -> Optional[str]:
        """Get input from the user using prompt_toolkit.

        Returns:
            User input string or None if cancelled
        """
        try:
            # Run prompt in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self._prompt_session.prompt(
                    [("class:prompt", "> ")],
                    style=PROMPT_STYLE,
                )
            )
        except (KeyboardInterrupt, EOFError):
            raise

    async def _process_query(self, user_input: str) -> None:
        """Process a user query through the agent.

        Args:
            user_input: The user's message
        """
        agent = await self._get_agent()

        # Add user message to session
        self.session.add_message("user", user_input)

        # Create spinner for thinking indicator
        prompt_tokens = len(user_input.split()) * 2  # Rough estimate
        spinner = ThinkingSpinner(self.console, prompt_tokens=prompt_tokens)

        try:
            spinner.start()

            # Prepare messages for agent
            messages = [{"role": m.role, "content": m.content} for m in self.session.messages]

            # Stream the response
            output_tokens = 0
            final_content = ""

            async for event in agent.astream_events(
                {"messages": messages},
                version="v2",
            ):
                event_type = event.get("event", "")

                if event_type == "on_chat_model_stream":
                    # Update token count
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        output_tokens += 1
                        spinner.update_tokens(output_tokens)

                elif event_type == "on_tool_start":
                    # Show tool call
                    spinner.stop()
                    tool_name = event.get("name", "unknown")
                    tool_input = event.get("data", {}).get("input", {})
                    render_tool_use(self.console, tool_name, tool_input, "call")
                    spinner.start()

                elif event_type == "on_tool_end":
                    # Show tool result
                    spinner.stop()
                    tool_name = event.get("name", "unknown")
                    output = event.get("data", {}).get("output", "")
                    if self.verbose:
                        render_tool_use(self.console, tool_name, output, "result")
                    spinner.start()

                elif event_type == "on_chain_end":
                    # Final response
                    output = event.get("data", {}).get("output", {})
                    if "messages" in output and output["messages"]:
                        final_message = output["messages"][-1]
                        if hasattr(final_message, "content"):
                            final_content = final_message.content

            spinner.stop()

            # Display final response
            if final_content:
                self.console.print()
                try:
                    md = Markdown(final_content)
                    self.console.print(md)
                except Exception:
                    self.console.print(final_content)
                self.console.print()

                # Add to session
                self.session.add_message("assistant", final_content)

        except Exception as e:
            spinner.stop()
            render_error(self.console, e)
        finally:
            spinner.stop()

    def _handle_slash_command(self, command: str) -> None:
        """Handle a slash command.

        Args:
            command: The full command string including the slash
        """
        parts = command[1:].strip().split(maxsplit=1)
        if not parts:
            return

        cmd_name = parts[0].lower()
        cmd_arg = parts[1] if len(parts) > 1 else ""

        if cmd_name == "help":
            self._show_help()
        elif cmd_name == "clear":
            self._clear_conversation()
        elif cmd_name == "exit" or cmd_name == "quit":
            self._should_exit = True
            self.console.print("[yellow]Goodbye![/yellow]")
        elif cmd_name == "history":
            self._show_history()
        elif cmd_name == "stats":
            self._show_stats()
        elif cmd_name == "compact":
            self._compact_history()
        elif cmd_name == "models":
            self._handle_models_command(cmd_arg)
        else:
            self.console.print(f"[red]Unknown command: /{cmd_name}[/red]")
            self.console.print("[dim]Type /help for available commands[/dim]")

    def _show_help(self) -> None:
        """Display help information."""
        help_text = """[bold cyan]Available Commands[/bold cyan]

[cyan]/help[/cyan]      Show this help message
[cyan]/clear[/cyan]     Clear conversation history
[cyan]/history[/cyan]   Show conversation history
[cyan]/stats[/cyan]     Show session statistics
[cyan]/compact[/cyan]   Compact conversation history
[cyan]/models[/cyan]    Manage model profiles and aliases
[cyan]/exit[/cyan]      Exit Ultrathink

[bold cyan]Tips[/bold cyan]
- Press [bold]Ctrl+C[/bold] to interrupt a response
- Press [bold]Ctrl+C[/bold] twice to exit
- Use [bold]Up/Down[/bold] arrows for history"""

        self.console.print(Panel(help_text, title="Help", border_style="cyan"))

    def _clear_conversation(self) -> None:
        """Clear the conversation history."""
        self.session.clear()
        self._agent = None  # Reset agent to clear its context
        self.console.print("[green]Conversation cleared.[/green]")

    def _show_history(self) -> None:
        """Show conversation history."""
        if not self.session.messages:
            self.console.print("[dim]No messages in history.[/dim]")
            return

        for i, msg in enumerate(self.session.messages, 1):
            role_style = "blue" if msg.role == "user" else "green"
            content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            self.console.print(f"[{role_style}]{i}. [{msg.role}][/{role_style}] {content_preview}")

    def _show_stats(self) -> None:
        """Show session statistics."""
        from rich.table import Table

        stats = self.session.stats
        table = Table(title="Session Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Messages", str(stats.total_messages))
        table.add_row("User Messages", str(stats.user_messages))
        table.add_row("Assistant Messages", str(stats.assistant_messages))
        table.add_row("Tool Calls", str(stats.tool_calls))
        table.add_row("Tokens In", str(stats.total_tokens_in))
        table.add_row("Tokens Out", str(stats.total_tokens_out))

        if stats.total_cost_usd > 0:
            table.add_row("Estimated Cost", f"${stats.total_cost_usd:.4f}")

        self.console.print(table)

    def _compact_history(self) -> None:
        """Compact conversation history (summarize old messages)."""
        # For now, just truncate to last 10 messages
        if len(self.session.messages) > 10:
            self.session.messages = self.session.messages[-10:]
            self.console.print("[green]History compacted to last 10 messages.[/green]")
        else:
            self.console.print("[dim]History is already compact.[/dim]")

    def _handle_models_command(self, args: str) -> None:
        """Handle the /models command for managing model profiles and aliases.

        Subcommands:
            /models              List all profiles and aliases
            /models add <name> <provider:model> [--api-key KEY] [--alias ALIAS]
            /models remove <name>
            /models alias <alias> <profile>
            /models unalias <alias>
            /models use <alias|profile>
        """
        from rich.table import Table

        from ultrathink.core.config import (
            ModelProfile,
            ProviderType,
            config_manager,
        )

        parts = args.strip().split()
        subcmd = parts[0].lower() if parts else "list"

        if subcmd == "list" or subcmd == "":
            self._models_list()

        elif subcmd == "add":
            self._models_add(parts[1:])

        elif subcmd == "update" or subcmd == "set":
            if len(parts) < 2:
                self.console.print("[red]Usage: /models update <name> [--api-key KEY] [--base-url URL] [--model MODEL][/red]")
                return
            self._models_update(parts[1:])

        elif subcmd == "remove" or subcmd == "rm":
            if len(parts) < 2:
                self.console.print("[red]Usage: /models remove <profile_name>[/red]")
                return
            self._models_remove(parts[1])

        elif subcmd == "alias":
            if len(parts) < 3:
                self.console.print("[red]Usage: /models alias <alias> <profile_name>[/red]")
                return
            self._models_alias(parts[1], parts[2])

        elif subcmd == "unalias":
            if len(parts) < 2:
                self.console.print("[red]Usage: /models unalias <alias>[/red]")
                return
            self._models_unalias(parts[1])

        elif subcmd == "use":
            if len(parts) < 2:
                self.console.print("[red]Usage: /models use <alias|profile>[/red]")
                return
            self._models_use(parts[1])

        elif subcmd == "help":
            self._models_help()

        else:
            self.console.print(f"[red]Unknown subcommand: {subcmd}[/red]")
            self._models_help()

    def _models_list(self) -> None:
        """List all model profiles and aliases."""
        from rich.table import Table

        from ultrathink.core.config import config_manager

        profiles = config_manager.list_profiles()
        aliases = config_manager.list_aliases()

        # Profiles table
        if profiles:
            table = Table(title="Model Profiles", show_header=True)
            table.add_column("Name", style="cyan")
            table.add_column("Provider", style="green")
            table.add_column("Model", style="yellow")
            table.add_column("API Key", style="dim")
            table.add_column("Base URL", style="dim")

            for name, profile in profiles.items():
                api_key_status = "[green]✓[/green]" if profile.api_key else "[dim]env[/dim]"
                base_url = profile.api_base or "(default)"
                table.add_row(name, profile.provider.value, profile.model, api_key_status, base_url)

            self.console.print(table)
        else:
            self.console.print("[dim]No model profiles configured.[/dim]")

        self.console.print()

        # Aliases table
        if aliases:
            table = Table(title="Aliases", show_header=True)
            table.add_column("Alias", style="cyan")
            table.add_column("Profile", style="green")
            table.add_column("Type", style="dim")

            builtin = {"main", "task", "quick"}
            for alias, profile in aliases.items():
                alias_type = "builtin" if alias in builtin else "custom"
                table.add_row(alias, profile, alias_type)

            self.console.print(table)

    def _models_add(self, args: List[str]) -> None:
        """Add a new model profile."""
        from ultrathink.core.config import ModelProfile, ProviderType, config_manager

        if len(args) < 2:
            self.console.print(
                "[red]Usage: /models add <name> <provider:model> "
                "[--api-key KEY] [--base-url URL] [--alias ALIAS][/red]"
            )
            return

        name = args[0]
        model_string = args[1]

        # Parse options
        api_key = None
        base_url = None
        alias = None
        i = 2
        while i < len(args):
            if args[i] == "--api-key" and i + 1 < len(args):
                api_key = args[i + 1]
                i += 2
            elif args[i] == "--base-url" and i + 1 < len(args):
                base_url = args[i + 1]
                i += 2
            elif args[i] == "--alias" and i + 1 < len(args):
                alias = args[i + 1]
                i += 2
            else:
                i += 1

        # Parse provider:model
        if ":" in model_string:
            provider_str, model = model_string.split(":", 1)
            try:
                provider = ProviderType.from_string(provider_str)
            except ValueError:
                self.console.print(f"[red]Unknown provider: {provider_str}[/red]")
                return
        else:
            provider = ProviderType.ANTHROPIC
            model = model_string

        profile = ModelProfile(
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=base_url,
        )

        try:
            config_manager.add_model_profile(name, profile, set_alias=alias)
            self.console.print(f"[green]Added profile '{name}': {provider.value}:{model}[/green]")
            if alias:
                self.console.print(f"[green]Set alias '{alias}' -> '{name}'[/green]")
        except ValueError as e:
            self.console.print(f"[red]Error: {e}[/red]")

    def _models_update(self, args: List[str]) -> None:
        """Update an existing model profile."""
        from ultrathink.core.config import config_manager

        if len(args) < 1:
            self.console.print(
                "[red]Usage: /models update <name> "
                "[--api-key KEY] [--base-url URL] [--model MODEL][/red]"
            )
            return

        name = args[0]

        # Parse options
        api_key = None
        base_url = None
        model = None
        i = 1
        while i < len(args):
            if args[i] == "--api-key" and i + 1 < len(args):
                api_key = args[i + 1]
                i += 2
            elif args[i] == "--base-url" and i + 1 < len(args):
                base_url = args[i + 1]
                i += 2
            elif args[i] == "--model" and i + 1 < len(args):
                model = args[i + 1]
                i += 2
            else:
                i += 1

        if api_key is None and base_url is None and model is None:
            self.console.print("[yellow]No options specified. Nothing to update.[/yellow]")
            self.console.print("[dim]Options: --api-key KEY, --base-url URL, --model MODEL[/dim]")
            return

        if config_manager.update_profile(name, api_key=api_key, api_base=base_url, model=model):
            updates = []
            if api_key:
                updates.append("api-key")
            if base_url:
                updates.append(f"base-url={base_url}")
            if model:
                updates.append(f"model={model}")
            self.console.print(f"[green]Updated profile '{name}': {', '.join(updates)}[/green]")
        else:
            self.console.print(f"[red]Profile '{name}' not found[/red]")

    def _models_remove(self, name: str) -> None:
        """Remove a model profile."""
        from ultrathink.core.config import config_manager

        try:
            if config_manager.remove_profile(name):
                self.console.print(f"[green]Removed profile '{name}'[/green]")
            else:
                self.console.print(f"[red]Profile '{name}' not found[/red]")
        except ValueError as e:
            self.console.print(f"[red]Error: {e}[/red]")

    def _models_alias(self, alias: str, profile: str) -> None:
        """Set an alias to point to a profile."""
        from ultrathink.core.config import config_manager

        try:
            config_manager.set_alias(alias, profile)
            self.console.print(f"[green]Set alias '{alias}' -> '{profile}'[/green]")
        except ValueError as e:
            self.console.print(f"[red]Error: {e}[/red]")

    def _models_unalias(self, alias: str) -> None:
        """Remove a custom alias."""
        from ultrathink.core.config import config_manager

        if alias in ("main", "task", "quick"):
            self.console.print(f"[red]Cannot remove builtin alias '{alias}'[/red]")
            return

        if config_manager.remove_alias(alias):
            self.console.print(f"[green]Removed alias '{alias}'[/green]")
        else:
            self.console.print(f"[red]Alias '{alias}' not found[/red]")

    def _models_use(self, name: str) -> None:
        """Set the main alias to use a profile."""
        from ultrathink.core.config import config_manager

        # Check if it's a profile name or an existing alias
        profiles = config_manager.list_profiles()
        aliases = config_manager.list_aliases()

        if name in profiles:
            # It's a profile, set main to it
            config_manager.set_alias("main", name)
            self.console.print(f"[green]Set main model to profile '{name}'[/green]")
            # Reset agent to use new model
            self._agent = None
        elif name in aliases:
            # It's an alias, set main to the same profile
            profile = aliases[name]
            config_manager.set_alias("main", profile)
            self.console.print(f"[green]Set main model to '{profile}' (from alias '{name}')[/green]")
            self._agent = None
        else:
            self.console.print(f"[red]Profile or alias '{name}' not found[/red]")

    def _models_help(self) -> None:
        """Show help for /models command."""
        help_text = """[bold cyan]/models - Manage Model Profiles and Aliases[/bold cyan]

[bold]Subcommands:[/bold]
  [cyan]/models[/cyan]                           List all profiles and aliases
  [cyan]/models add[/cyan] <name> <provider:model> [options]
                                    Add a new profile
  [cyan]/models update[/cyan] <name> [options]   Update existing profile
  [cyan]/models remove[/cyan] <name>             Remove a profile
  [cyan]/models alias[/cyan] <alias> <profile>   Set alias to point to profile
  [cyan]/models unalias[/cyan] <alias>           Remove a custom alias
  [cyan]/models use[/cyan] <name>                Set main to use profile/alias
  [cyan]/models help[/cyan]                      Show this help

[bold]Options for add/update:[/bold]
  --api-key KEY                   Set/update API key for this profile
  --base-url URL                  Set/update custom API base URL
  --model MODEL                   Update model name (update only)
  --alias ALIAS                   Create alias pointing to profile (add only)

[bold]Supported Providers:[/bold]
  [green]anthropic[/green] (claude)    Claude models (claude-sonnet-4-20250514, claude-opus-4-20250514, ...)
  [green]openai[/green] (gpt)          OpenAI models (gpt-4o, gpt-4o-mini, o1, ...)
  [green]deepseek[/green] (ds)         DeepSeek models (deepseek-chat, deepseek-reasoner)
  [green]gemini[/green] (google)       Google Gemini models (gemini-pro, gemini-1.5-pro, ...)

[bold]Builtin Aliases:[/bold]
  [dim]main[/dim]   Primary model for main agent interactions
  [dim]task[/dim]   Model for subagent/task execution
  [dim]quick[/dim]  Model for quick operations (summarization, etc.)

[bold]Custom Aliases:[/bold]
  Create your own aliases for use in subagents or tool configurations.
  Example: /models alias research my-research-profile

[bold]API Key Priority:[/bold]
  1. Profile-specific key (--api-key in add/update)
  2. Environment variable (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
  3. ~/.ultrathink.json config

[bold]List Output:[/bold]
  API Key column: [green]✓[/green] = configured in profile, [dim]env[/dim] = from environment

[bold]Examples:[/bold]
  [dim]# Add a new profile with API key[/dim]
  /models add claude anthropic:claude-sonnet-4-20250514 --api-key sk-ant-xxx

  [dim]# Add DeepSeek Reasoner with custom base URL[/dim]
  /models add reasoner deepseek:deepseek-reasoner --api-key sk-xxx --base-url https://api.deepseek.com/v1

  [dim]# Add fast model and set as 'quick' alias[/dim]
  /models add fast openai:gpt-4o-mini --api-key sk-xxx --alias quick

  [dim]# Update existing profile's API key[/dim]
  /models update default --api-key sk-new-key

  [dim]# Update model name[/dim]
  /models update default --model deepseek-chat

  [dim]# Create custom alias for subagent use[/dim]
  /models alias research reasoner

  [dim]# Switch main model[/dim]
  /models use fast"""

        self.console.print(Panel(help_text, title="/models Help", border_style="cyan"))

    def ask_user(self, question: str, options: List[str], context: str) -> str:
        """Callback for the ask_user tool.

        Args:
            question: Question to ask
            options: Available options
            context: Additional context

        Returns:
            User's response
        """
        self.console.print()

        if context:
            self.console.print(f"[dim]{context}[/dim]")

        self.console.print(f"[bold cyan]Question:[/bold cyan] {question}")

        if options:
            self.console.print("[dim]Options:[/dim]")
            for i, opt in enumerate(options, 1):
                self.console.print(f"  [cyan]{i}.[/cyan] {opt}")
            self.console.print("[dim]Enter a number or type your own response:[/dim]")

        try:
            response = self._prompt_session.prompt(
                [("class:prompt", "? ")],
                style=PROMPT_STYLE,
            )

            # Check if it's a number selecting an option
            if options and response.isdigit():
                idx = int(response) - 1
                if 0 <= idx < len(options):
                    return options[idx]

            return response

        except (KeyboardInterrupt, EOFError):
            return ""
