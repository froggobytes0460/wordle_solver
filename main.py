from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

__version__ = version("wordle-solver")

app = typer.Typer(help="Wordle Solver CLI")
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"wordle-solver {__version__}")
        raise typer.Exit()


def _validate_npy_suffix(path: Path) -> Path:
    if path.suffix != ".npy":
        raise typer.BadParameter(
            message=f"State LUT must be a '.npy' file, got '{path.suffix}'",
        )
    return path


@app.command()
def main(
    word_list: Annotated[
        Path,
        typer.Argument(
            help="Path to newline-separated word list file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            shell_complete=lambda _context, _param, incomplete: [
                str(p) for p in Path.cwd().glob(f"{incomplete}*.txt")
            ],
        ),
    ],
    matrix: Annotated[
        Path,
        typer.Argument(
            help="Path to precomputed .npy pattern matrix file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            callback=_validate_npy_suffix,
        ),
    ],
    target: Annotated[
        str | None,
        typer.Option(
            "--target",
            help="Known target word; solver auto-plays against it without "
            "manual G/Y/B input.",
        ),
    ] = None,
    _: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = False,
) -> None:
    """Play an interactive Wordle solving session."""

    from tui.play import play as run_play

    run_play(console=console, word_list=word_list, matrix=matrix, target=target)


if __name__ == "__main__":
    app()
