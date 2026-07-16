"""prompt_toolkit Application wiring: key bindings and rendering for WordleGame."""

import asyncio

from prompt_toolkit import Application
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl

from tui.game import WordleGame
from tui.pattern import STATE_BG


def build_app(game: WordleGame) -> Application[None]:
    kb = KeyBindings()

    async def _compute_next_guess() -> None:
        # best_guess() is CPU-bound numpy work; run off the event loop so
        # the "Computing…" status actually renders instead of the UI
        # freezing silently until the call returns.
        loop = asyncio.get_running_loop()
        guess_idx = await loop.run_in_executor(None, game.solver.best_guess)
        game.apply_guess(guess_idx)
        app.invalidate()

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
    async def _(event) -> None:
        if game.finished:
            event.app.exit()
        elif not game.thinking and game.confirm_row():
            await _compute_next_guess()

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
    app = Application(layout=layout, key_bindings=kb, full_screen=True)
    app.pre_run = lambda: app.create_background_task(  # pyright: ignore[reportAttributeAccessIssue]
        coroutine=_compute_next_guess()
    )
    return app
