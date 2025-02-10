# tests/conftest.py

import os
import shutil
from pathlib import Path

import pytest

from tests.helpers.notebook_creator import NotebookCreator

os.environ["JUPYTER_PLATFORM_DIRS"] = "1"


@pytest.fixture(scope="session")
def test_notebooks_dir():
    """Create and manage test notebooks directory"""
    test_dir = Path(__file__).parent / "test_notebooks"
    test_dir.mkdir(exist_ok=True)

    # Create test notebooks
    NotebookCreator.create_basic_notebook(test_dir)
    NotebookCreator.create_ray_tune_notebook(test_dir)
    NotebookCreator.create_ray_tune_notebook_slow(test_dir)
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
