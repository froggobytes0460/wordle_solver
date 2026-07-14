from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, ProgressColumn, SpinnerColumn, Task, TextColumn
from rich.text import Text

from word_crunch import build_pattern_matrix


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


@tools_app.command()
def build_matrix(
    word_list: Annotated[
        Path,
        typer.Argument(
            ...,
            help="Path to newline-separated word list file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output: Annotated[
        Path,
        typer.Argument(
            ...,
            help="Output .npy file path",
        ),
    ],
) -> None:
    """Build a Wordle pattern matrix and save to a .npy file."""

    if output.suffix != ".npy":
        console.print(
            f"[red]Error:[/red] Output must be a .npy file, got '{output.suffix}'"
        )
        raise typer.Exit(code=1)

    if output.exists():
        overwrite = typer.confirm(f"'{output}' already exists. Overwrite?")
        if not overwrite:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(code=0)

    words = word_list.read_text().splitlines()
    words = [w.strip() for w in words if w.strip()]

    console.print(
        Panel(
            f"[bold]Word list:[/bold] {word_list}\n"
            f"[bold]Words loaded:[/bold] {len(words):,}\n"
            f"[bold]Output:[/bold] {output}",
            title="[bold blue]Build Pattern Matrix[/bold blue]",
            expand=False,
        )
    )
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        MinSecMsColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Computing pattern matrix...", total=None)
        matrix, elapsed = build_pattern_matrix(words)
        progress.update(task, description="Done.", completed=True)

    np.save(file=output, arr=matrix)
    console.print()

    console.print(
        Panel(
            renderable=f"[bold green]Saved:[/bold green] {output}\n"
            f"[bold]Matrix shape:[/bold] {matrix.shape[0]:,} x {matrix.shape[1]:,}\n"
            f"[bold]Time taken:[/bold] {elapsed:.3f}s",
            title="[bold green]Complete[/bold green]",
            expand=False,
        )
    )


if __name__ == "__main__":
    app()
