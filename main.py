import time
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from numpy.typing import NDArray
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskProgressColumn,
    TextColumn,
)
from rich.text import Text

__version__ = version("wordle-solver")


class MinSecMsColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        elapsed = task.elapsed or 0.0
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        ms = int((elapsed % 1) * 1000)
        return Text(f"{minutes:02d}:{seconds:02d}.{ms:03d}", style="progress.elapsed")


app = typer.Typer(help="Wordle Solver CLI")
tools_app = typer.Typer(help="Utility tools (matrix building, etc.)")
app.add_typer(typer_instance=tools_app, name="tools")
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"wordle-solver {__version__}")
        raise typer.Exit()


def _validate_npy_suffix(path: Path) -> Path:
    if path.suffix != ".npy":
        raise typer.BadParameter(
            message=f"Output must be a '.npy' file, got '{path.suffix}'",
        )
    return path


def _format_size(size_bytes: int) -> str:
    """Converts bytes to a human-readable string (kB, MB, GB)."""
    if size_bytes == 0:
        return "0 B"
    for unit in ["B", "kB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024  # pyright: ignore[reportAssignmentType]
    return f"{size_bytes:.2f} PB"


@app.callback()
def version_callback(
    _: Annotated[
        bool,
        typer.Option(
            default="--version",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = False,
) -> None:
    pass


@tools_app.command()
def build_matrix(
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
    output: Annotated[
        Path,
        typer.Argument(
            help="Output .npy file path",
            dir_okay=False,
            writable=True,
            resolve_path=True,
            callback=_validate_npy_suffix,
        ),
    ],
) -> None:
    """Build a Wordle pattern matrix and save to a .npy file."""

    if output.exists():
        overwrite = typer.confirm(f"'{output}' already exists. Overwrite?")
        if not overwrite:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(code=0)

    words = [
        valid_w for w in word_list.read_text().splitlines() if (valid_w := w.strip())
    ]

    if not words:
        console.print("[red]Error:[/red] Word list is empty")
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold]Word list:[/bold] {word_list}\n"
            f"[bold]Words loaded:[/bold] {len(words):,}\n"
            f"[bold]Output:[/bold] {output}",
            title="[bold blue]Build Pattern Matrix[/bold blue]",
            expand=True,
        )
    )
    console.print()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        MinSecMsColumn(),
        console=console,
        expand=True,
        get_time=time.perf_counter,
    ) as progress:
        from word_crunch import build_pattern_matrix

        n = len(words)
        task = progress.add_task("Computing pattern matrix...", total=n)
        result: tuple[NDArray[np.uint8], float] | None = None
        for rows_done, result_matrix, result_elapsed in build_pattern_matrix(words):
            progress.update(task_id=task, completed=rows_done)
            result = (result_matrix, result_elapsed)
        progress.update(task_id=task, description="Done.")

        if result is None:
            raise RuntimeError("Pattern matrix computation yielded no results")
        matrix, elapsed = result

    np.save(file=output, arr=matrix)
    console.print()

    console.print(
        Panel(
            renderable=f"[bold green]Saved:[/bold green] {output}\n"
            f"[bold]File size:[/bold] {_format_size(output.stat().st_size)}\n"
            f"[bold]Matrix shape:[/bold] {matrix.shape[0]:,} x {matrix.shape[1]:,}\n"
            f"[bold]Time taken:[/bold] {elapsed:.3f}s",
            title="[bold green]Complete[/bold green]",
            expand=True,
        )
    )


if __name__ == "__main__":
    app()
