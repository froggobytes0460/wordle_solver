"""prompt_toolkit Application wiring: key bindings and rendering for WordleGame."""

from prompt_toolkit import Application
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl

from tui.game import WordleGame
from tui.pattern import STATE_BG


def build_app(game: WordleGame) -> Application[None]:
    kb = KeyBindings()

    @kb.add("left")
    def _(_event) -> None:
        if not game.auto:
            game.move_cursor(-1)

    @kb.add("right")
    def _(_event) -> None:
        if not game.auto:
            game.move_cursor(1)

    @kb.add("space")
    def _(_event) -> None:
        if not game.finished and not game.auto:
            game.cycle_state()

    @kb.add("enter")
    def _(event) -> None:
        if game.finished:
            event.app.exit()
        else:
            game.confirm_row()

    @kb.add("q")
    @kb.add("c-c")
    def _(event) -> None:
        event.app.exit()

    def render() -> FormattedText:
        lines: list[tuple[str, str]] = []
        lines.append(("bold", "  Wordle Solver\n\n"))

        for guess, states in game.rows:
            for letter, state in zip(guess.upper(), states, strict=True):
                lines.append((f"bg:{STATE_BG[state]} bold fg:white", f" {letter} "))
                lines.append(("", " "))
            lines.append(("", "\n"))

        if not game.finished:
            for i, letter in enumerate(game.cur_guess.upper()):
                style = f"bg:{STATE_BG[game.cur_states[i]]} bold fg:white"
                if not game.auto and i == game.cursor:
                    style += " underline"
                lines.append((style, f" {letter} "))
                lines.append(("", " "))
            lines.append(("", "\n"))

        lines.append(("", "\n"))
        lines.append(("italic", f"  {game.status}\n"))

        if not game.finished:
            if game.auto:
                lines.append(("dim", "  enter step  q quit\n"))
            else:
                lines.append(
                    (
                        "dim",
                        "  ←/→ move  space cycle grey→yellow→green  "
                        "enter confirm row  q quit\n",
                    )
                )
        else:
            lines.append(("dim", "  enter/q to exit\n"))

        return FormattedText(lines)

    control = FormattedTextControl(text=render)
    layout = Layout(HSplit([Window(content=control)]))
    return Application(layout=layout, key_bindings=kb, full_screen=True)
