from pathlib import Path
from typing import Optional

import typer
from experiment_utils.utils.log_utils import get_runs, print_runs, rename_runs
from rich import print


def show_runs(dir_name: Optional[str] = typer.Argument(None),
              rename: bool = typer.Option(False),
              thresold: float = typer.Option(0.8, '-t')):
    """Print results."""
    if dir_name is None:
        dir_name = Path.cwd()
    else:
        dir_name = Path(dir_name)
    print(f"log dir: {dir_name}")
    runs = get_runs(dir_name)
    if len(runs) == 0:
        typer.echo(f"No logs in dir: {dir_name}")
        raise typer.Exit()
    print_runs(runs, thresold=thresold)
    if rename:
        rename_runs(runs, thresold)


if __name__ == "__main__":
    typer.run(show_runs)
