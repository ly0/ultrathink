"""Thinking spinner with playful verbs and token progress display."""

from __future__ import annotations

import random
import time
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner as RichSpinner
from rich.text import Text


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
    """

    def __init__(
        self,
        console: Console,
        prompt_tokens: int = 0,
        spinner_style: str = "dots",
    ) -> None:
        """Initialize the thinking spinner.

        Args:
            console: Rich console for output
            prompt_tokens: Number of input tokens (for display)
            spinner_style: Rich spinner animation style
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

    def start(self) -> None:
        """Start the spinner animation."""
        if self._running:
            return

        self._running = True
        spinner = RichSpinner(self._spinner_style, text=self._format_text())
        self._live = Live(
            spinner,
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

        display_text = text or self._format_text(self._current_suffix)
        spinner = RichSpinner(self._spinner_style, text=display_text)
        self._live.update(spinner)

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
