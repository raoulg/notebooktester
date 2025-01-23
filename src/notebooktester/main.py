import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import nbformat
from loguru import logger
from nbconvert.preprocessors import ExecutePreprocessor


class NotebookTester:
    def __init__(
        self,
        dir: Path,
        timeout: int = 600,
    ):
        self.notebooks_dir = Path(dir)
        self.timeout = timeout
        self.results: Dict[str, Dict] = {}

        # Setup logging
        log_path = Path("logs/notebook_tests")
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logger.remove()  # Remove default handler
        logger.add(sys.stderr, level="INFO")
        logger.add(log_file, rotation="10 MB")

    def test_notebook(self, notebook_path: Path) -> Tuple[str, bool, str]:
        """Test a single notebook and return results"""
        return self._test_locally(notebook_path)

    def _test_locally(self, notebook_path: Path) -> Tuple[str, bool, str]:
        """Test notebook in local environment"""
        try:
            with open(notebook_path) as f:
                nb = nbformat.read(f, as_version=4)

            ep = ExecutePreprocessor(timeout=self.timeout, kernel_name="python3")
            ep.preprocess(nb, {"metadata": {"path": notebook_path.parent}})

            return (str(notebook_path), True, "Success")
        except Exception as e:
            return (str(notebook_path), False, str(e))

    def find_notebooks(self) -> List[Path]:
        """Find all notebooks in the specified directory"""
        return sorted(
            [
                path
                for path in self.notebooks_dir.rglob("*.ipynb")
                if ".ipynb_checkpoints" not in str(path)
            ]
        )

    def run_tests(self, max_workers: int = 4):
        """Run tests on all notebooks"""
        notebooks = self.find_notebooks()
        logger.info(f"Starting notebook tests - found {len(notebooks)} notebooks")

        successful = failed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.test_notebook, nb): nb for nb in notebooks}

            for future in as_completed(futures):
                path, success, message = future.result()
                status = "✅ PASSED" if success else "❌ FAILED"
                successful += 1 if success else 0
                failed += 0 if success else 1

                logger.info(f"{status} - {path}")
                if not success:
                    logger.error(f"Error in {path}: {message}")

        logger.info(f"\nTest Summary: {successful} passed, {failed} failed")


if __name__ == "__main__":
    tester = NotebookTester(dir=Path("notebooks"), timeout=60)
    tester.run_tests()
