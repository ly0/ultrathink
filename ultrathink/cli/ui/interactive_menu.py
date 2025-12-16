"""Interactive menu component using prompt_toolkit.

Provides arrow key navigation for CLI menus.
"""

from typing import Any, List, Optional, Tuple

from prompt_toolkit import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style


class InteractiveMenu:
    """Interactive menu with arrow key navigation.

    Features:
    - Up/Down arrow keys to navigate
    - Enter to select
    - Esc/q to go back
    - Visual highlight for current selection
    """

    def __init__(
        self,
        title: str,
        items: List[Tuple[str, str, Any]],  # (label, status, data)
        subtitle: str = "",
        header_lines: Optional[List[Tuple[str, str]]] = None,  # Extra header content
        footer: str = "↑↓ Navigate · Enter Select · Esc Back",
    ) -> None:
        """Initialize the menu.

        Args:
            title: Menu title
            items: List of (label, status, data) tuples
            subtitle: Optional subtitle
            header_lines: Optional list of (style, text) tuples for header content
            footer: Footer text with instructions
        """
        self.title = title
        self.items = items
        self.subtitle = subtitle
        self.header_lines = header_lines or []
        self.footer = footer
        self.selected_index = 0
        self.result: Optional[Any] = None
        self.cancelled = False

    def _get_formatted_text(self) -> List[Tuple[str, str]]:
        """Generate formatted text for the menu."""
        lines: List[Tuple[str, str]] = []

        # Title
        if self.title:
            lines.append(("class:title", f"\n  {self.title}\n"))

        # Subtitle
        if self.subtitle:
            lines.append(("class:subtitle", f"  {self.subtitle}\n"))

        if self.title or self.subtitle:
            lines.append(("", "\n"))

        # Header content (extra info lines)
        if self.header_lines:
            for style, text in self.header_lines:
                lines.append((style, f"  {text}\n"))
            lines.append(("", "\n"))

        # Items
        for i, (label, status, _) in enumerate(self.items):
            if i == self.selected_index:
                # Selected item
                lines.append(("class:selected-marker", "  › "))
                lines.append(("class:selected", f"{i + 1}. {label}"))
                if status:
                    lines.append(("class:selected-status", f" {status}"))
                lines.append(("", "\n"))
            else:
                # Normal item
                lines.append(("", "    "))
                lines.append(("class:item", f"{i + 1}. {label}"))
                if status:
                    lines.append(("class:status", f" {status}"))
                lines.append(("", "\n"))

        # Footer
        lines.append(("", "\n"))
        lines.append(("class:footer", f"  {self.footer}\n"))

        return lines

    async def run_async(self) -> Optional[Any]:
        """Run the menu asynchronously and return selected item's data.

        Returns:
            Selected item's data or None if cancelled
        """
        if not self.items:
            return None

        # Key bindings
        kb = KeyBindings()

        @kb.add("up")
        @kb.add("k")
        def move_up(event):
            self.selected_index = (self.selected_index - 1) % len(self.items)

        @kb.add("down")
        @kb.add("j")
        def move_down(event):
            self.selected_index = (self.selected_index + 1) % len(self.items)

        @kb.add("enter")
        def select(event):
            self.result = self.items[self.selected_index][2]
            event.app.exit()

        @kb.add("escape")
        @kb.add("q")
        def cancel(event):
            self.cancelled = True
            event.app.exit()

        # Number keys for direct selection
        for i in range(min(9, len(self.items))):
            @kb.add(str(i + 1))
            def select_number(event, idx=i):
                if idx < len(self.items):
                    self.result = self.items[idx][2]
                    event.app.exit()

        # Style
        style = Style.from_dict({
            "title": "bold cyan",
            "subtitle": "gray",
            "header-label": "yellow",
            "header-value": "white",
            "header-status-ok": "green",
            "header-status-dim": "gray",
            "selected-marker": "cyan bold",
            "selected": "cyan bold",
            "selected-status": "green",
            "item": "white",
            "status": "green",
            "footer": "gray italic",
        })

        # Layout
        layout = Layout(
            HSplit([
                Window(
                    content=FormattedTextControl(self._get_formatted_text),
                    always_hide_cursor=True,
                ),
            ])
        )

        # Application - use erase_when_done for automatic cleanup
        app = Application(
            layout=layout,
            key_bindings=kb,
            style=style,
            full_screen=False,
            mouse_support=True,
            erase_when_done=True,
        )

        try:
            await app.run_async()
        except (KeyboardInterrupt, EOFError):
            self.cancelled = True

        if self.cancelled:
            return None
        return self.result


class InteractiveList:
    """Paginated interactive list with arrow key navigation.

    Features:
    - Up/Down to navigate items
    - Left/Right or PageUp/PageDown for pagination
    - Enter to select
    - Esc/q to go back
    """

    def __init__(
        self,
        title: str,
        items: List[Tuple[str, Any]],  # (label, data)
        page_size: int = 10,
        footer: str = "↑↓ Navigate · ←→ Pages · Enter Select · Esc Back",
    ) -> None:
        """Initialize the list.

        Args:
            title: List title
            items: List of (label, data) tuples
            page_size: Items per page
            footer: Footer text
        """
        self.title = title
        self.items = items
        self.page_size = page_size
        self.footer = footer
        self.selected_index = 0
        self.current_page = 0
        self.total_pages = (len(items) + page_size - 1) // page_size if items else 1
        self.result: Optional[Any] = None
        self.cancelled = False

    def _get_page_items(self) -> List[Tuple[str, Any]]:
        """Get items for current page."""
        start = self.current_page * self.page_size
        end = min(start + self.page_size, len(self.items))
        return self.items[start:end]

    def _get_formatted_text(self) -> List[Tuple[str, str]]:
        """Generate formatted text for the list."""
        lines: List[Tuple[str, str]] = []

        # Title with count
        lines.append(("class:title", f"\n  {self.title}"))
        lines.append(("class:count", f" ({len(self.items)} items)\n"))
        lines.append(("", "\n"))

        # Items
        page_items = self._get_page_items()
        start_idx = self.current_page * self.page_size

        if not page_items:
            lines.append(("class:empty", "  No items\n"))
        else:
            for i, (label, _) in enumerate(page_items):
                global_idx = start_idx + i
                if global_idx == self.selected_index:
                    lines.append(("class:selected-marker", "  › "))
                    lines.append(("class:selected", f"{global_idx + 1}. {label}"))
                    lines.append(("", "\n"))
                else:
                    lines.append(("", "    "))
                    lines.append(("class:item", f"{global_idx + 1}. {label}"))
                    lines.append(("", "\n"))

        # Pagination
        if self.total_pages > 1:
            lines.append(("", "\n"))
            lines.append(("class:page", f"  Page {self.current_page + 1}/{self.total_pages}\n"))

        # Footer
        lines.append(("", "\n"))
        lines.append(("class:footer", f"  {self.footer}\n"))

        return lines

    async def run_async(self) -> Optional[Any]:
        """Run the list asynchronously and return selected item's data."""
        if not self.items:
            return None

        kb = KeyBindings()

        @kb.add("up")
        @kb.add("k")
        def move_up(event):
            if self.selected_index > 0:
                self.selected_index -= 1
                # Check if we need to go to previous page
                if self.selected_index < self.current_page * self.page_size:
                    self.current_page -= 1

        @kb.add("down")
        @kb.add("j")
        def move_down(event):
            if self.selected_index < len(self.items) - 1:
                self.selected_index += 1
                # Check if we need to go to next page
                if self.selected_index >= (self.current_page + 1) * self.page_size:
                    self.current_page += 1

        @kb.add("left")
        @kb.add("pageup")
        @kb.add("h")
        def prev_page(event):
            if self.current_page > 0:
                self.current_page -= 1
                self.selected_index = self.current_page * self.page_size

        @kb.add("right")
        @kb.add("pagedown")
        @kb.add("l")
        def next_page(event):
            if self.current_page < self.total_pages - 1:
                self.current_page += 1
                self.selected_index = self.current_page * self.page_size

        @kb.add("enter")
        def select(event):
            self.result = self.items[self.selected_index][1]
            event.app.exit()

        @kb.add("escape")
        @kb.add("q")
        def cancel(event):
            self.cancelled = True
            event.app.exit()

        style = Style.from_dict({
            "title": "bold cyan",
            "count": "gray",
            "selected-marker": "cyan bold",
            "selected": "cyan bold",
            "item": "white",
            "page": "gray",
            "footer": "gray italic",
            "empty": "gray italic",
        })

        layout = Layout(
            HSplit([
                Window(
                    content=FormattedTextControl(self._get_formatted_text),
                    always_hide_cursor=True,
                ),
            ])
        )

        app = Application(
            layout=layout,
            key_bindings=kb,
            style=style,
            full_screen=False,
            mouse_support=True,
            erase_when_done=True,
        )

        try:
            await app.run_async()
        except (KeyboardInterrupt, EOFError):
            self.cancelled = True

        if self.cancelled:
            return None
        return self.result


class DetailView:
    """Display detail information with key to go back.

    Simple view that shows formatted content and waits for
    any key to return.
    """

    def __init__(
        self,
        title: str,
        content: List[Tuple[str, str]],  # List of (label, value)
        footer: str = "Press any key to go back",
    ) -> None:
        """Initialize the detail view.

        Args:
            title: View title
            content: List of (label, value) tuples
            footer: Footer text
        """
        self.title = title
        self.content = content
        self.footer = footer

    def _get_formatted_text(self) -> List[Tuple[str, str]]:
        """Generate formatted text."""
        lines: List[Tuple[str, str]] = []

        # Title
        lines.append(("class:title", f"\n  {self.title}\n"))
        lines.append(("", "\n"))

        # Content
        for label, value in self.content:
            if label:
                lines.append(("class:label", f"  {label}: "))
                lines.append(("class:value", f"{value}\n"))
            else:
                # Plain text line
                lines.append(("class:value", f"  {value}\n"))

        # Footer
        lines.append(("", "\n"))
        lines.append(("class:footer", f"  {self.footer}\n"))

        return lines

    async def run_async(self) -> None:
        """Display the view asynchronously and wait for key press."""
        kb = KeyBindings()

        @kb.add("<any>")
        def any_key(event):
            event.app.exit()

        style = Style.from_dict({
            "title": "bold cyan",
            "label": "yellow",
            "value": "white",
            "footer": "gray italic",
        })

        layout = Layout(
            HSplit([
                Window(
                    content=FormattedTextControl(self._get_formatted_text),
                    always_hide_cursor=True,
                ),
            ])
        )

        app = Application(
            layout=layout,
            key_bindings=kb,
            style=style,
            full_screen=False,
            erase_when_done=True,
        )

        try:
            await app.run_async()
        except (KeyboardInterrupt, EOFError):
            pass
