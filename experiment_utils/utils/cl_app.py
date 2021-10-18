from pathlib import Path
from typing import Optional

import typer
from experiment_utils.utils.log_utils import get_runs, filter_runs, print_runs, rename_runs
from rich import print


def show_runs(dir_name: Optional[str] = typer.Argument(None),
              rename: bool = typer.Option(False),
              thresold: float = typer.Option(0, '-t', help='Print only runs with accurasy more than `thresold`'),
              limit: int = typer.Option(0, '-l', help='Print only `limit` lines.'),
              last: bool = typer.Option(False)):
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
        # print_runs(runs, header=f"last dirs", limit=20, print_num=True)
        limit = limit or 20
        print_runs(runs[-limit:], header=f"last dirs", limit=limit, print_num=True)
        raise typer.Exit()
    if thresold:
        runs = filter_runs(runs, thresold)
        thresolded = f", {len(runs)} with acc > {thresold:.2%}"
        if len(runs) == 0:
            typer.echo(f"{len_runs} runs, no run with thresold {thresold}")
            raise typer.Exit()
    else:
        thresolded = ''

    print_runs(runs, header=f"{len_runs} log dirs{thresolded}", limit=limit)
    if rename:
        rename_runs(runs, thresold)


if __name__ == "__main__":
    typer.run(show_runs)
