"""Tab-based multi-question UI using prompt_toolkit."""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import (
    FormattedTextControl,
    HSplit,
    Layout,
    Window,
)
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, TextArea


@dataclass
class QuestionData:
    """Data for a single question."""

    id: str
    question: str
    options: List[str] = field(default_factory=list)
    required: bool = True
    default: Optional[str] = None
    answer: str = ""


@dataclass
class TabQuestionState:
    """State for the tab-based question UI."""

    questions: List[QuestionData]
    context: str
    title: str
    current_tab: int = 0
    confirmed: bool = False
    cancelled: bool = False

    @property
    def is_on_confirmation_tab(self) -> bool:
        return self.current_tab >= len(self.questions)

    @property
    def total_tabs(self) -> int:
        # questions + confirmation tab
        return len(self.questions) + 1


class TabQuestionsUI:
    """Interactive Tab-based question UI."""

    STYLE = Style.from_dict({
        "tab-bar": "bg:#3a3a3a",
        "tab": "#888888",
        "tab.active": "#00cccc bold",
        "tab.number": "#6688ff",
        "question": "#00cccc bold",
        "context": "#888888 italic",
        "options": "#88cc88",
        "option-number": "#cccc00 bold",
        "input-area": "bg:#1a1a1a",
        "confirmation-title": "#00cccc bold",
        "confirmation-q": "#888888",
        "confirmation-a": "#88cc88",
        "hint": "#666666 italic",
        "frame": "#00cccc",
        "separator": "#444444",
    })

    def __init__(self, state: TabQuestionState):
        self.state = state
        self.text_areas: List[TextArea] = []
        self._init_text_areas()
        self.app: Optional[Application] = None

    def _init_text_areas(self) -> None:
        """Initialize text areas for each question."""
        for q in self.state.questions:
            area = TextArea(
                text=q.answer or q.default or "",
                multiline=False,
                style="class:input-area",
                height=1,
            )
            self.text_areas.append(area)

    def _get_tab_bar_tokens(self) -> List[tuple]:
        """Generate tokens for the tab bar."""
        tokens = []
        for i, q in enumerate(self.state.questions):
            is_active = i == self.state.current_tab

            # Separator
            if i > 0:
                tokens.append(("class:tab", " | "))

            # Tab indicator
            style = "class:tab.active" if is_active else "class:tab"
            indicator = ">" if is_active else " "
            tokens.append((style, indicator))

            # Tab number and short label
            tokens.append(("class:tab.number", f"[{i + 1}] "))
            label = q.question[:15] + "..." if len(q.question) > 15 else q.question
            tokens.append((style, label))

        # Confirmation tab
        tokens.append(("class:tab", " | "))
        is_confirm_active = self.state.is_on_confirmation_tab
        style = "class:tab.active" if is_confirm_active else "class:tab"
        indicator = ">" if is_confirm_active else " "
        tokens.append((style, indicator))
        tokens.append(("class:tab.number", f"[{len(self.state.questions) + 1}] "))
        tokens.append((style, "Confirm"))

        return tokens

    def _get_question_content_tokens(self, idx: int) -> List[tuple]:
        """Generate tokens for question content."""
        q = self.state.questions[idx]
        tokens = []

        # Question text
        tokens.append(("class:question", f"\n  {q.question}\n"))

        # Options if present
        if q.options:
            tokens.append(("", "\n"))
            for i, opt in enumerate(q.options, 1):
                tokens.append(("class:option-number", f"    {i}. "))
                tokens.append(("class:options", f"{opt}\n"))

        # Hint
        tokens.append(("", "\n"))
        hint = "  Type your answer below. "
        if q.options:
            hint = "  Enter a number to select, or type custom response. "
        hint += "Tab: next | Shift+Tab: back"
        if not q.required:
            hint += " | (Optional)"
        tokens.append(("class:hint", hint + "\n"))

        return tokens

    def _get_confirmation_tokens(self) -> List[tuple]:
        """Generate tokens for confirmation page."""
        tokens = []

        tokens.append(("class:confirmation-title", "\n  Review Your Answers\n"))
        tokens.append(("class:separator", "  " + "-" * 40 + "\n"))

        for i, q in enumerate(self.state.questions):
            answer = self.text_areas[i].text or q.default or "(no answer)"
            # Resolve option selection
            if q.options and answer.isdigit():
                opt_idx = int(answer) - 1
                if 0 <= opt_idx < len(q.options):
                    answer = q.options[opt_idx]

            tokens.append(("class:confirmation-q", f"\n  Q{i + 1}: {q.question}\n"))
            tokens.append(("class:confirmation-a", f"      A: {answer}\n"))

        tokens.append(("", "\n"))
        tokens.append(("class:hint", "  Press Enter to confirm and submit.\n"))
        tokens.append(("class:hint", "  Press Shift+Tab to go back and edit.\n"))
        tokens.append(("class:hint", "  Press Escape or Ctrl+C to cancel.\n"))

        return tokens

    def _create_keybindings(self) -> KeyBindings:
        """Create key bindings for navigation."""
        kb = KeyBindings()

        @kb.add("tab")
        def next_tab(event):
            """Move to next tab."""
            self._save_current_answer()
            if self.state.current_tab < self.state.total_tabs - 1:
                self.state.current_tab += 1
                self._rebuild_layout()

        @kb.add("s-tab")  # Shift+Tab
        def prev_tab(event):
            """Move to previous tab."""
            self._save_current_answer()
            if self.state.current_tab > 0:
                self.state.current_tab -= 1
                self._rebuild_layout()

        @kb.add("enter")
        def handle_enter(event):
            """Handle Enter key."""
            if self.state.is_on_confirmation_tab:
                self._save_current_answer()
                self.state.confirmed = True
                event.app.exit()
            else:
                # Move to next tab
                next_tab(event)

        @kb.add("c-c")
        @kb.add("c-d")
        def cancel(event):
            """Cancel the interaction."""
            self.state.cancelled = True
            event.app.exit()

        @kb.add("escape")
        def escape(event):
            """Go back or cancel."""
            if self.state.current_tab > 0:
                self.state.current_tab -= 1
                self._rebuild_layout()
            else:
                self.state.cancelled = True
                event.app.exit()

        return kb

    def _save_current_answer(self) -> None:
        """Save the current text area content to state."""
        if not self.state.is_on_confirmation_tab:
            idx = self.state.current_tab
            self.state.questions[idx].answer = self.text_areas[idx].text

    def _rebuild_layout(self) -> None:
        """Rebuild the application layout for current tab."""
        if self.app:
            self.app.layout = self._create_layout()
            # Ensure the correct TextArea has focus
            if not self.state.is_on_confirmation_tab:
                self.app.layout.focus(self.text_areas[self.state.current_tab])

    def _create_layout(self) -> Layout:
        """Create the full application layout."""
        elements = []

        # Context (if provided)
        if self.state.context:
            elements.append(Window(
                content=FormattedTextControl([
                    ("class:context", f"\n  {self.state.context}\n")
                ]),
                height=3,
            ))

        # Tab bar
        elements.append(Window(
            content=FormattedTextControl(self._get_tab_bar_tokens),
            height=1,
        ))

        # Separator
        elements.append(Window(
            content=FormattedTextControl([("class:separator", "─" * 50)]),
            height=1,
        ))

        # Content area based on current tab
        if self.state.is_on_confirmation_tab:
            # Confirmation page
            elements.append(Window(
                content=FormattedTextControl(self._get_confirmation_tokens),
                height=10 + len(self.state.questions) * 2,
            ))
        else:
            # Question content
            idx = self.state.current_tab
            elements.append(Window(
                content=FormattedTextControl(
                    lambda idx=idx: self._get_question_content_tokens(idx)
                ),
                height=6 + len(self.state.questions[idx].options),
            ))

            # Input area
            elements.append(Window(height=1))  # spacer
            elements.append(Window(
                content=FormattedTextControl([("", "  Answer: ")]),
                height=1,
            ))
            elements.append(self.text_areas[idx])

        body = HSplit(elements)

        # Determine which element should have focus
        focused_element = None
        if not self.state.is_on_confirmation_tab:
            focused_element = self.text_areas[self.state.current_tab]

        return Layout(
            Frame(body, title=self.state.title, style="class:frame"),
            focused_element=focused_element,
        )

    def run(self) -> Dict[str, Any]:
        """Run the interactive UI and return results.

        This creates a new event loop for the thread.
        """
        import sys

        # Clear any lingering output (like spinner) before starting
        # Move cursor to beginning of line and clear
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            self.app = Application(
                layout=self._create_layout(),
                key_bindings=self._create_keybindings(),
                style=self.STYLE,
                full_screen=False,
                mouse_support=True,
            )

            loop.run_until_complete(self.app.run_async())

            # Collect answers
            answers = {}
            for i, q in enumerate(self.state.questions):
                answer = self.text_areas[i].text
                # Resolve option numbers to actual values
                if q.options and answer.isdigit():
                    opt_idx = int(answer) - 1
                    if 0 <= opt_idx < len(q.options):
                        answer = q.options[opt_idx]
                answers[q.id] = answer

            return {
                "answers": answers,
                "confirmed": self.state.confirmed,
                "cancelled": self.state.cancelled,
            }
        finally:
            loop.close()


def run_tab_questions(
    questions: List[Dict[str, Any]],
    context: str = "",
    title: str = "Questions",
) -> Dict[str, Any]:
    """Run the tab-based question UI.

    Args:
        questions: List of question dictionaries with keys:
            - id: unique identifier
            - question: question text
            - options: optional list of choices
            - required: whether answer is required (default True)
            - default: default value
        context: Overall context text
        title: Dialog title

    Returns:
        Dictionary with:
            - answers: dict mapping question id to answer
            - confirmed: whether user confirmed
            - cancelled: whether user cancelled
    """
    question_data = [
        QuestionData(
            id=q["id"],
            question=q["question"],
            options=q.get("options", []),
            required=q.get("required", True),
            default=q.get("default"),
        )
        for q in questions
    ]

    state = TabQuestionState(
        questions=question_data,
        context=context,
        title=title,
    )

    ui = TabQuestionsUI(state)
    return ui.run()
