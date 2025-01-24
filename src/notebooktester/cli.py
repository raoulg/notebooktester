from pathlib import Path

import click

from .main import NotebookTester


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--timeout", "-t", default=60, help="Timeout in seconds for each notebook"
)
@click.option("--workers", "-w", type=int, help="Number of parallel workers")
@click.option(
    "--cache-dir",
    "-c",
    default=".notebookcache",
    type=click.Path(),
    help="Cache directory for test results",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option(
    "--force", "-f", is_flag=True, help="Ignore cache and force test execution"
)
def main(
    path: str, timeout: int, workers: int, cache_dir: str, verbose: bool, force: bool
):
    tester = NotebookTester(
        dir=Path(path),
        timeout=timeout,
        cache_dir=Path(cache_dir) if cache_dir else None,
        verbose=verbose,
        force=force,
    )
    tester.run_tests(max_workers=workers)
