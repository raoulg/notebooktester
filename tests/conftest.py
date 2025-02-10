# tests/conftest.py

import os
import shutil
from pathlib import Path

import pytest

from notebooktester.main import NotebookTester
from tests.helpers.notebook_creator import NotebookCreator

os.environ["JUPYTER_PLATFORM_DIRS"] = "1"


@pytest.fixture(scope="session")
def test_notebooks_dir():
    """Create and manage test notebooks directory"""
    test_dir = Path(__file__).parent / "test_notebooks"
    test_dir.mkdir(exist_ok=True)

    # Create test notebooks
    NotebookCreator.create_basic_notebook(test_dir)
    NotebookCreator.create_timeout_notebook(test_dir)

    yield test_dir

    # Cleanup after tests
    shutil.rmtree(test_dir)


@pytest.fixture(scope="session")
def test_cache_dir():
    """Create and manage test cache directory"""
    cache_dir = Path(__file__).parent / ".test_cache"
    cache_dir.mkdir(exist_ok=True)

    yield cache_dir

    # Cleanup after tests
    shutil.rmtree(cache_dir)


@pytest.fixture
def notebook_tester(test_notebooks_dir, test_cache_dir):
    """Create a NotebookTester instance with short timeout"""
    tester = NotebookTester(
        dir=test_notebooks_dir,
        timeout=1,  # Short timeout for testing
        cache_dir=test_cache_dir,
        verbose=True,
    )
    yield tester
