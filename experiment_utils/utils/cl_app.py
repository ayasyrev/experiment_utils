from pathlib import Path
from typing import Optional

import typer
from experiment_utils.utils.log_utils import (
    filter_runs,
    get_runs,
    print_runs,
    rename_runs,
)
from rich import print


def show_runs(
    dir_name: Optional[str] = typer.Argument(None),
    rename: bool = typer.Option(
        False,
        "-R",
        help="Rename directory with log with accuracy, if threshold - only filtered.",
    ),
    print_parent: bool = typer.Option(False, "-P", help="Print parent name"),
    threshold: float = typer.Option(
        0, "-t", help="Print only runs with accuracy more than `threshold`"
    ),
    limit: int = typer.Option(0, "-l", help="Print only `limit` lines."),
    last: bool = typer.Option(False),
):
    """Print results."""
    if dir_name is None:
        dir_name = Path.cwd()
    else:
        dir_name = Path(dir_name)
    print(f"log dir: {dir_name}")
    runs = get_runs(dir_name, sort=not last)
    len_runs = len(runs)
    if len_runs == 0:
        typer.echo(f"No logs in dir: {dir_name}")
        raise typer.Exit()
    if last:
        limit = limit or 20
        print_runs(runs[-limit:], header="last dirs", limit=limit, print_num=True)
        raise typer.Exit()
    if threshold:
        runs = filter_runs(runs, threshold)
        thresholded = f", {len(runs)} with acc > {threshold:.2%}"
        if len(runs) == 0:
            typer.echo(f"{len_runs} runs, no run with threshold {threshold}")
            raise typer.Exit()
    else:
        thresholded = ""

    print_runs(
        runs,
        header=f"{len_runs} log dirs{thresholded}",
        limit=limit,
        print_parent=print_parent,
    )
    if rename:
        rename_runs(runs, threshold)


if __name__ == "__main__":
    typer.run(show_runs)
