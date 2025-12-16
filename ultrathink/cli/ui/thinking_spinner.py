"""Thinking spinner with playful verbs and token progress display."""

from __future__ import annotations

import asyncio
import random
import time
from typing import Callable, Optional, TYPE_CHECKING

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner as RichSpinner
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from asyncio import Task


THINKING_WORDS: list[str] = [
    "Accomplishing",
    "Actualizing",
    "Brewing",
    "Calculating",
    "Cerebrating",
    "Cogitating",
    "Computing",
    "Conjuring",
    "Contemplating",
    "Cooking",
    "Crafting",
    "Creating",
    "Deliberating",
    "Divining",
    "Enchanting",
    "Forging",
    "Generating",
    "Hatching",
    "Ideating",
    "Imagining",
    "Manifesting",
    "Mulling",
    "Musing",
    "Percolating",
    "Pondering",
    "Processing",
    "Reasoning",
    "Ruminating",
    "Scheming",
    "Synthesizing",
    "Thinking",
    "Tinkering",
    "Transmuting",
    "Unravelling",
    "Working",
    "Wrangling",
]


class ThinkingSpinner:
    """Spinner that shows elapsed time and token progress.

    Displays a playful thinking message with:
    - Random thinking verb (e.g., "Pondering", "Cogitating")
    - Elapsed time in seconds
    - Input/output token counts
    - Optional status suffix
    - Model status bar at the bottom
    """

    def __init__(
        self,
        console: Console,
        prompt_tokens: int = 0,
        spinner_style: str = "dots",
        status_func: Optional[Callable[[], str]] = None,
        user_input: Optional[str] = None,
    ) -> None:
        """Initialize the thinking spinner.

        Args:
            console: Rich console for output
            prompt_tokens: Number of input tokens (for display)
            spinner_style: Rich spinner animation style
            status_func: Optional callable that returns model status string
            user_input: Optional user input for dynamic thinking word generation
        """
        self.console = console
        self.prompt_tokens = prompt_tokens
        self.start_time = time.monotonic()
        self.out_tokens = 0
        self.thinking_word = random.choice(THINKING_WORDS)
        self._spinner_style = spinner_style
        self._live: Optional[Live] = None
        self._running = False
        self._current_suffix: Optional[str] = None
        self._status_func = status_func
        self._user_input = user_input
        self._generation_task: Optional[Task] = None
        self._sync_task: Optional[Task] = None
        self._current_todo_id: Optional[str] = None
        self._model_generated: bool = False  # Track if we've tried model generation

    def _format_text(self, suffix: Optional[str] = None) -> Text:
        """Format the spinner text with current state.

        Args:
            suffix: Optional status message to append

        Returns:
            Formatted Rich Text object
        """
        elapsed = int(time.monotonic() - self.start_time)

        # Build the message parts
        parts = [
            ("* ", "cyan bold"),
            (f"{self.thinking_word}... ", "cyan"),
            ("(", "dim"),
            ("esc to interrupt", "dim italic"),
            (" | ", "dim"),
            (f"{elapsed}s", "dim"),
        ]

        # Add token info
        if self.out_tokens > 0:
            parts.extend([
                (" | ", "dim"),
                (f"{self.out_tokens} tokens out", "dim green"),
            ])
        elif self.prompt_tokens > 0:
            parts.extend([
                (" | ", "dim"),
                (f"{self.prompt_tokens} tokens in", "dim blue"),
            ])

        # Add suffix if provided
        if suffix:
            parts.extend([
                (" | ", "dim"),
                (suffix, "dim yellow"),
            ])

        parts.append((")", "dim"))

        # Build Text object
        text = Text()
        for content, style in parts:
            text.append(content, style=style)

        return text

    def _make_renderable(self, suffix: Optional[str] = None) -> Table:
        """Create a renderable with spinner and status bar.

        Args:
            suffix: Optional status message to append to spinner

        Returns:
            Table containing spinner and status bar
        """
        # Create a table with no borders for layout
        table = Table.grid(expand=True)
        table.add_column(ratio=1)

        # Add spinner row
        spinner_text = self._format_text(suffix)
        spinner = RichSpinner(self._spinner_style, text=spinner_text)
        table.add_row(spinner)

        # Add status bar row if status_func is provided
        if self._status_func:
            status_text = Text()
            status_text.append(" " + self._status_func() + " ", style="reverse dim")
            table.add_row(status_text)

        return table

    def start(self) -> None:
        """Start the spinner animation."""
        if self._running:
            return

        self._running = True
        renderable = self._make_renderable()
        self._live = Live(
            renderable,
            console=self.console,
            refresh_per_second=10,
            transient=True,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the spinner animation."""
        if not self._running:
            return

        self._running = False
        # Cancel background tasks
        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
        if self._generation_task and not self._generation_task.done():
            self._generation_task.cancel()
        if self._live:
            self._live.stop()
            self._live = None

    def update(self, text: Optional[Text] = None) -> None:
        """Update the spinner display.

        Args:
            text: New text to display (uses current state if None)
        """
        if not self._running or not self._live:
            return

        renderable = self._make_renderable(self._current_suffix)
        self._live.update(renderable)

    def update_tokens(self, out_tokens: int, suffix: Optional[str] = None) -> None:
        """Update the output token count and optional suffix.

        Args:
            out_tokens: Current output token count
            suffix: Optional status message
        """
        self.out_tokens = max(0, out_tokens)
        self._current_suffix = suffix
        self.update()

    def set_suffix(self, suffix: str) -> None:
        """Set a status suffix message.

        Args:
            suffix: Status message to display
        """
        self._current_suffix = suffix
        self.update()

    def clear_suffix(self) -> None:
        """Clear the status suffix."""
        self._current_suffix = None
        self.update()

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time since spinner started."""
        return time.monotonic() - self.start_time

    def __enter__(self) -> "ThinkingSpinner":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()

    def update_thinking_word(self, word: str) -> None:
        """Dynamically update the thinking word and refresh display.

        Args:
            word: New thinking word/phrase to display
        """
        self.thinking_word = word
        self.update()

    def _get_current_todo(self) -> Optional[tuple]:
        """Get current actionable TODO item (in_progress first, then pending).

        Returns:
            Tuple of (todo_id, todo_content) or None if no actionable TODO
        """
        try:
            from ultrathink.utils.todo import get_next_actionable, load_todos

            todos = load_todos()
            actionable_todo = get_next_actionable(todos)
            if actionable_todo:
                return (actionable_todo.id, actionable_todo.content)
        except Exception:
            pass
        return None

    async def _sync_todo_loop(self) -> None:
        """Background loop to sync thinking word with TODO status.

        Runs every 0.5 seconds to check for TODO changes.
        """
        while self._running:
            try:
                todo_info = self._get_current_todo()

                if todo_info:
                    todo_id, todo_content = todo_info
                    # Only update if TODO changed
                    if todo_id != self._current_todo_id:
                        self._current_todo_id = todo_id
                        word = todo_content
                        if len(word) > 50:
                            word = word[:47] + "..."
                        self.update_thinking_word(word)
                elif self._current_todo_id is not None:
                    # TODO was cleared, try model generation if not done yet
                    self._current_todo_id = None
                    if not self._model_generated:
                        self._model_generated = True
                        await self._generate_from_model()

            except asyncio.CancelledError:
                break
            except Exception:
                pass

            await asyncio.sleep(0.5)

    async def _generate_from_model(self) -> None:
        """Generate thinking word from user input via quick model."""
        if not self._user_input:
            return

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            from ultrathink.core.agent_factory import init_model
            from ultrathink.core.config import get_model_profile

            quick_profile = get_model_profile("quick")
            if quick_profile is None:
                quick_profile = get_model_profile("main")

            model = init_model(profile=quick_profile)
            response = await model.ainvoke([
                SystemMessage(content=(
                    "根据用户输入，生成一个不超过8个词的动词短语描述'我正在做什么'。"
                    "只输出短语本身，不要其他内容。语言与用户输入保持一致。"
                )),
                HumanMessage(content=self._user_input),
            ])

            word = response.content.strip()
            # Validate: non-empty and within word limit
            if word and len(word.split()) <= 8:
                self.update_thinking_word(word)
        except Exception:
            pass

    def start_dynamic_generation(self) -> None:
        """Start background task to sync TODO status and generate dynamic thinking word."""
        if self._sync_task is None:
            self._sync_task = asyncio.create_task(self._sync_todo_loop())
