import asyncio
import inspect
import json
import multiprocessing
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import nbformat
from loguru import logger
from nbclient import NotebookClient
from tqdm import tqdm


class GracefulExit(Exception):
    pass


class NotebookTester:
    def __init__(
        self,
        dir: Path,
        timeout: int = 60,
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

        self.executor = None
        self.successful = 0
        self.failed = 0
        self.no_more_time = 0
        self.interrupted = False
        self._setup_logging()

        # signal handler
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _setup_logging(self):
        """Configure structured logging"""
        log_path = Path("logs/")
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / "notebookstests.log"

        # Remove default logger
        logger.remove()

        # Add TIMEOUT level if it doesn't exist
        if "TIMEOUT" not in logger._core.levels:  # type: ignore
            logger.level("TIMEOUT", no=20, color="<yellow>")

        # Add handlers
        logger.add(log_file, level="DEBUG")
        logger.add(sys.stderr, level="SUCCESS" if not self.verbose else "DEBUG")

    def _signal_handler(self, signum, _):
        """Handle termination signals gracefully"""
        signal_name = signal.Signals(signum).name
        logger.warning(f"\nReceived {signal_name}. Initiating graceful shutdown...")
        self.interrupted = True

        if self.executor:
            logger.info("Shutting down executor...")
            self.executor.shutdown(wait=False)

        raise GracefulExit()

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

        if cache["timeout"] >= self.timeout and "A cell timed out" in cache["message"]:
            logger.debug(f"cache: {cache['timeout']}, current timeout {self.timeout}")
            return False

        return notebook_path.stat().st_mtime > cache["last_run"] or not cache["success"]

    def _update_cache(self, notebook_path: Path, success: bool, message: str):
        if not self.cache_dir:
            return

        cache_key = self._get_cache_key(notebook_path)
        cache_file = self.cache_dir / f"{cache_key}.json"
        cache_data = {
            "last_run": notebook_path.stat().st_mtime,
            "success": success,
            "message": message,
            "timeout": self.timeout,
        }

        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

    async def _cleanup_notebook_client(self, client: Optional[NotebookClient]) -> None:
        """Clean up notebook client resources asynchronously.

        Args:
            client: The notebook client to clean up
        """
        if not client:
            return

        async def _safe_stop_channels(kc):
            """Safely stop kernel channels if they exist."""
            if not kc:
                return
            try:
                if hasattr(kc, "stop_channels"):
                    if inspect.iscoroutinefunction(kc.stop_channels):
                        await kc.stop_channels()
                    else:
                        kc.stop_channels()
            except Exception as e:
                logger.debug(f"Error stopping channels: {e}")

        async def _safe_shutdown_kernel(km):
            """Safely shutdown kernel if it exists."""
            if not km:
                return
            try:
                if hasattr(km, "shutdown_kernel"):
                    # Add timeout to prevent hanging
                    await asyncio.wait_for(
                        asyncio.to_thread(km.shutdown_kernel, now=True), timeout=5.0
                    )
            except asyncio.TimeoutError:
                logger.warning("Kernel shutdown timed out")
            except Exception as e:
                logger.debug(f"Error shutting down kernel: {e}")

        # Run cleanup tasks concurrently
        await asyncio.gather(
            _safe_stop_channels(getattr(client, "kc", None)),
            _safe_shutdown_kernel(getattr(client, "km", None)),
        )

    async def _execute_notebook(self, notebook_path: Path) -> Tuple[bool, str]:
        """Execute a single notebook asynchronously"""
        client = None
        try:
            with open(notebook_path) as f:
                nb = nbformat.read(f, as_version=4)

            client = NotebookClient(
                nb,
                timeout=self.timeout,
                kernel_name="python3",
                resources={"metadata": {"path": notebook_path.parent}},
            )

            await client.async_execute()
            return True, "Success"
        except Exception as e:
            return False, str(e)
        finally:
            if client:
                await self._cleanup_notebook_client(client)

    def test_notebook(self, notebook_path: Path) -> Tuple[str, bool, str, bool]:
        """Test a single notebook."""
        if not self._should_run_test(notebook_path):
            if self.cache_dir is None:  # Handle the case when cache_dir is None
                return str(notebook_path), False, "No cache directory specified", False

            cache_key = self._get_cache_key(notebook_path)
            try:
                with open(self.cache_dir / f"{cache_key}.json") as f:
                    cache = json.load(f)
                return (
                    str(notebook_path),
                    cache["success"],
                    f"{cache['message']}",
                    True,
                )
            except Exception as e:
                return str(notebook_path), False, f"Error reading cache: {e}", False

        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run the execution in a way that lets us clean up while loop is still running
            async def run_and_cleanup():
                try:
                    success, message = await self._execute_notebook(notebook_path)
                    return success, message
                finally:
                    # Cancel any remaining tasks while loop is still running
                    for task in asyncio.all_tasks():
                        if not task.done() and task is not asyncio.current_task():
                            task.cancel()
                            try:
                                await asyncio.wait_for(task, timeout=1.0)
                            except asyncio.TimeoutError:
                                pass
                            except asyncio.CancelledError:
                                pass

            success, message = loop.run_until_complete(run_and_cleanup())
            self._update_cache(notebook_path, success, message)
            return str(notebook_path), success, message, False
        finally:
            loop.close()

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
        try:
            notebooks = self.find_notebooks()
            if not max_workers:
                max_workers = multiprocessing.cpu_count()
            logger.info(f"Running tests with {max_workers} workers")
            logger.info(f"Starting notebook tests - found {len(notebooks)} notebooks")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                self.executor = executor
                futures = {
                    executor.submit(self.test_notebook, nb): nb for nb in notebooks
                }

                with tqdm(total=len(notebooks), disable=self.verbose) as pbar:
                    for future in as_completed(futures):
                        try:
                            path, success, message, cached = future.result()
                            if success:
                                status = "📦✅ CACHED" if cached else "✅ PASSED"
                                logger.info(f"{status} - {path}: {message}")
                                self.successful += 1
                            elif "A cell timed out" in message:
                                logger.log("TIMEOUT", f"⏰ TIMEOUT - {path}: {message}")
                                self.no_more_time += 1
                            else:
                                logger.error(f"❌ FAILED - {path}: {message}")
                                self.failed += 1
                            pbar.update(1)
                        except Exception as e:
                            logger.error(f"Error processing future: {str(e)}")
                            self.failed += 1
                            pbar.update(1)

        except GracefulExit:
            logger.warning("Graceful exit requested")

        logger.success(
            f"\nTest Summary: {self.successful} passed, {self.no_more_time} timed out, {self.failed} failed"
        )
