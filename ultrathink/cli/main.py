"""Ultrathink CLI - Main entry point.

This module provides the command-line interface for Ultrathink.
"""

import asyncio
import os
from pathlib import Path
from typing import Optional

import click

from ultrathink import __version__
from ultrathink.core.config import ensure_onboarding


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="ultrathink")
@click.option(
    "--cwd",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Set working directory",
)
@click.option(
    "--unsafe",
    is_flag=True,
    help="Disable safe mode (skip permission checks)",
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "-p", "--prompt",
    type=str,
    help="Execute a single prompt and exit (non-interactive mode)",
)
@click.option(
    "--model",
    type=str,
    help="Model to use (e.g., 'anthropic:claude-sonnet-4-20250514')",
)
@click.option(
    "--base-url",
    type=str,
    envvar=["ANTHROPIC_BASE_URL", "OPENAI_BASE_URL"],
    help="Custom API base URL",
)
@click.pass_context
def cli(
    ctx: click.Context,
    cwd: Optional[Path],
    unsafe: bool,
    verbose: bool,
    prompt: Optional[str],
    model: Optional[str],
    base_url: Optional[str],
) -> None:
    """Ultrathink - AI-powered coding assistant built on deepagent.

    Run without arguments to start interactive mode.
    Use --prompt/-p to execute a single command and exit.
    """
    # Set working directory if specified
    if cwd:
        os.chdir(cwd)

    # Check onboarding / API keys
    if not ensure_onboarding():
        ctx.exit(1)

    # Store options in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["safe_mode"] = not unsafe
    ctx.obj["verbose"] = verbose
    ctx.obj["model"] = model
    ctx.obj["base_url"] = base_url
    ctx.obj["cwd"] = cwd or Path.cwd()

    # If a prompt is provided, run in non-interactive mode
    if prompt:
        asyncio.run(_run_single_query(
            prompt,
            safe_mode=not unsafe,
            verbose=verbose,
            model=model,
            base_url=base_url,
        ))
        return

    # If no subcommand, start interactive mode
    if ctx.invoked_subcommand is None:
        _run_interactive(
            safe_mode=not unsafe,
            verbose=verbose,
            model=model,
            base_url=base_url,
        )


async def _run_single_query(
    prompt: str,
    safe_mode: bool = True,
    verbose: bool = False,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> None:
    """Run a single query in non-interactive mode.

    Args:
        prompt: The user's prompt
        safe_mode: Whether to enable permission checks
        verbose: Whether to enable verbose output
        model: Optional model override
        base_url: Optional API base URL override
    """
    from rich.console import Console

    from ultrathink.core.agent_factory import create_ultrathink_agent
    from ultrathink.cli.ui.message_renderer import render_message, render_error
    from ultrathink.mcp.config_loader import load_mcp_config

    console = Console()

    # Load MCP configuration
    mcp_config = load_mcp_config(Path.cwd())
    if mcp_config and verbose:
        console.print(f"[dim]MCP: Found {len(mcp_config)} server(s)[/dim]")

    try:
        agent = await create_ultrathink_agent(
            model=model,
            safe_mode=safe_mode,
            verbose=verbose,
            base_url=base_url,
            mcp_config=mcp_config,
        )

        # Run the query
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": prompt}]
        })

        # Display the result
        if "messages" in result and result["messages"]:
            final_message = result["messages"][-1]
            render_message(console, final_message)

    except Exception as e:
        render_error(console, e)
        raise click.Abort()


def _run_interactive(
    safe_mode: bool = True,
    verbose: bool = False,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> None:
    """Start the interactive CLI mode.

    Args:
        safe_mode: Whether to enable permission checks
        verbose: Whether to enable verbose output
        model: Optional model override
        base_url: Optional API base URL override
    """
    from ultrathink.cli.ui.rich_console import UltrathinkUI

    ui = UltrathinkUI(
        safe_mode=safe_mode,
        verbose=verbose,
        model=model,
        base_url=base_url,
    )
    ui.run()


@cli.command()
@click.pass_context
def config(ctx: click.Context) -> None:
    """View and edit configuration."""
    from rich.console import Console
    from rich.table import Table

    from ultrathink.core.config import get_global_config

    console = Console()
    config = get_global_config()

    table = Table(title="Ultrathink Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Safe Mode", str(config.safe_mode))
    table.add_row("Verbose", str(config.verbose))
    table.add_row("Theme", config.theme)
    table.add_row("Onboarding Complete", str(config.has_completed_onboarding))

    # Model profiles
    for name, profile in config.model_profiles.items():
        table.add_row(f"Model: {name}", f"{profile.provider.value}:{profile.model}")

    console.print(table)


@cli.command()
@click.argument("name")
@click.argument("model_string")
@click.option("--set-main", is_flag=True, help="Set as main model")
@click.pass_context
def add_model(
    ctx: click.Context,
    name: str,
    model_string: str,
    set_main: bool,
) -> None:
    """Add a model profile.

    NAME: Profile name (e.g., 'fast', 'smart')
    MODEL_STRING: Model identifier (e.g., 'anthropic:claude-sonnet-4-20250514')
    """
    from rich.console import Console

    from ultrathink.core.config import (
        ModelProfile,
        ProviderType,
        add_model_profile,
    )

    console = Console()

    # Parse model string
    if ":" in model_string:
        provider_str, model = model_string.split(":", 1)
        provider = ProviderType.from_string(provider_str)
    else:
        provider = ProviderType.ANTHROPIC
        model = model_string

    profile = ModelProfile(provider=provider, model=model)

    try:
        add_model_profile(name, profile, set_as_main=set_main)
        console.print(f"[green]Added model profile '{name}': {model_string}[/green]")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@cli.command()
@click.pass_context
def version(ctx: click.Context) -> None:
    """Show version information."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    info = f"""[bold cyan]Ultrathink[/bold cyan] v{__version__}

AI-powered coding assistant built on deepagent.

[dim]https://github.com/ultrathink/ultrathink[/dim]"""

    console.print(Panel(info, border_style="cyan"))


if __name__ == "__main__":
    cli()
