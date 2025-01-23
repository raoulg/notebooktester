import json
import multiprocessing
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import click
import nbformat
from loguru import logger
from nbconvert.preprocessors import ExecutePreprocessor
from tqdm import tqdm


class NotebookTester:
    def __init__(
        self,
        dir: Path,
        timeout: int = 600,
        cache_dir: Optional[Path] = Path(".notebookcache"),
        verbose: bool = False,
        force: bool = False,
    ):
        self.notebooks_dir = Path(dir)
        self.timeout = timeout
        self.verbose = verbose
        self.cache_dir = cache_dir
        self.force = force
        if cache_dir:
            logger.info(f"creating {cache_dir}")
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            logger.info("no cache")

        # Setup logging
        log_path = Path("logs/")
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / "notebookstests.log"
        logger.remove()
        logger.add(log_file, level="INFO")
        logger.add(sys.stderr, level="SUCCESS" if not verbose else "INFO")

    def _get_cache_key(self, notebook_path: Path) -> str:
        return str(notebook_path.resolve()).replace("/", "_").replace("\\", "_")

    def _should_run_test(self, notebook_path: Path) -> bool:
        if self.force or not self.cache_dir:
            return True

        cache_key = self._get_cache_key(notebook_path)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return True

        with open(cache_file) as f:
            cache = json.load(f)

        return notebook_path.stat().st_mtime > cache["last_run"]

    def _update_cache(self, notebook_path: Path, success: bool, message: str):
        if not self.cache_dir:
            return

        cache_key = self._get_cache_key(notebook_path)
        cache_file = self.cache_dir / f"{cache_key}.json"

        cache_data = {
            "last_run": notebook_path.stat().st_mtime,
            "success": success,
            "message": message,
        }

        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

    def test_notebook(self, notebook_path: Path) -> Tuple[str, bool, str]:
        if not self._should_run_test(notebook_path):
            assert self.cache_dir is not None
            cache_key = self._get_cache_key(notebook_path)
            with open(self.cache_dir / f"{cache_key}.json") as f:
                cache = json.load(f)
            return (
                str(notebook_path),
                cache["success"],
                f"[Cached] {cache['message']}",
            )

        try:
            with open(notebook_path) as f:
                nb = nbformat.read(f, as_version=4)

            ep = ExecutePreprocessor(timeout=self.timeout, kernel_name="python3")
            ep.preprocess(nb, {"metadata": {"path": notebook_path.parent}})

            self._update_cache(notebook_path, True, "Success")
            return (str(notebook_path), True, "Success")
        except Exception as e:
            error_msg = str(e)
            self._update_cache(notebook_path, False, error_msg)
            return (str(notebook_path), False, error_msg)

    def find_notebooks(self) -> List[Path]:
        """Find all notebooks in the specified directory"""
        if self.notebooks_dir.is_file():
            return [self.notebooks_dir]
        return sorted(
            [
                path
                for path in self.notebooks_dir.rglob("*.ipynb")
                if ".ipynb_checkpoints" not in str(path)
            ]
        )

    def run_tests(self, max_workers: Optional[int] = None):
        """Run tests on all notebooks"""
        notebooks = self.find_notebooks()
        if not max_workers:
            max_workers = multiprocessing.cpu_count()
        logger.info(f"Running tests with {max_workers} workers")

        logger.info(f"Starting notebook tests - found {len(notebooks)} notebooks")
        successful = failed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.test_notebook, nb): nb for nb in notebooks}

            with tqdm(total=len(notebooks), disable=self.verbose) as pbar:
                for future in as_completed(futures):
                    path, success, message = future.result()
                    status = "✅ PASSED" if success else "❌ FAILED"
                    successful += 1 if success else 0
                    failed += 0 if success else 1

                    logger.info(f"{status} - {path}")
                    if not success:
                        logger.error(f"Error in {path}: {message}")
                    pbar.update(1)

        logger.info(f"\nTest Summary: {successful} passed, {failed} failed")


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


if __name__ == "__main__":
    main()
