"""Interactive full-screen REPL game loop for the Wordle solver."""

from pathlib import Path

from rich.console import Console

from ai.solver import WordleSolver
from tui.app import build_app
from tui.game import WordleGame
from tui.loader import load_words_and_lut_and_weights


def play(
    console: Console, word_list: Path, matrix: Path, target: str | None = None
) -> None:
    """Runs the interactive full-screen Wordle solver game loop.

    :param console: Rich console for pre/post-game messages.
    :type console: Console
    :param word_list: Path to newline-separated word list file.
    :type word_list: Path
    :param matrix: Path to the precomputed .npy pattern matrix file.
    :type matrix: Path
    :param target: If given, the solver auto-plays against this known target
        word instead of waiting for manual G/Y/B input.
    :type target: str | None
    """

    words, state_lut, weights = load_words_and_lut_and_weights(
        word_list=word_list, matrix=matrix
    )
    if len(words) != state_lut.shape[0]:
        console.print(
            f"[red]Error:[/red] word list has {len(words):,} words but matrix "
            f"has {state_lut.shape[0]:,} rows. Rebuild the matrix for this word list."
        )
        raise SystemExit(1)

    target_idx: int | None = None
    if target is not None:
        target = target.strip().lower()
        try:
            target_idx = words.index(target)
        except ValueError:
            console.print(
                f"[red]Error:[/red] target word '{target}' not found in word list"
            )
            raise SystemExit(1) from None

    solver = WordleSolver(words=words, state_lut=state_lut, weights=weights)
    game = WordleGame(words=words, solver=solver, target_idx=target_idx)
    app = build_app(game)
    app.run()

    if game.won:
        console.print(
            f"[bold green]Solved![/bold green] "
            f"{game.rows[-1][0].upper()} in {len(game.rows)} attempt(s)."
        )
    elif game.finished:
        console.print(f"[yellow]{game.status}[/yellow]")
    else:
        console.print("[yellow]Aborted.[/yellow]")
