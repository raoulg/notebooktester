# tests/test_notebooktester.py


import pytest
import ray

from notebooktester.main import NotebookTester


@pytest.fixture
def notebook_tester(test_notebooks_dir, test_cache_dir):
    """Create a NotebookTester instance with short timeout"""
    tester = NotebookTester(
        dir=test_notebooks_dir,
        timeout=5,  # Short timeout for testing
        cache_dir=test_cache_dir,
        verbose=True,
    )
    yield tester

    # Ensure ray is shut down after each test
    if ray.is_initialized():
        ray.shutdown()


def test_basic_notebook_execution(notebook_tester, test_notebooks_dir):
    """Test execution of a simple notebook"""
    basic_nb = test_notebooks_dir / "basic.ipynb"
    result = notebook_tester.test_notebook(basic_nb)
    _, success, message, cached = result

    assert success is True
    assert "Success" in message
    assert cached is False


def test_cached_execution(notebook_tester, test_notebooks_dir):
    """Test that notebook results are properly cached"""
    basic_nb = test_notebooks_dir / "basic.ipynb"

    # Second execution should use cache
    result2 = notebook_tester.test_notebook(basic_nb)
    assert result2[3] is True  # Should be cached
    assert result2[1] is True  # Should still be successful


def test_timeout_handling(notebook_tester, test_notebooks_dir, test_cache_dir):
    """Test that notebooks that take too long timeout properly"""
    timeout_nb = test_notebooks_dir / "timeout.ipynb"
    result = notebook_tester.test_notebook(timeout_nb)
    _, success, message, _ = result

    assert success is False
    assert "timed out" in message.lower()

    patient_tester = NotebookTester(
        dir=test_notebooks_dir, timeout=15, cache_dir=test_cache_dir, force=False
    )
    result2 = patient_tester.test_notebook(timeout_nb)
    assert result2[1] is True  # Should succeed with longer timeout


def test_ray_tune_cleanup(notebook_tester, test_notebooks_dir):
    """Test that Ray resources are properly cleaned up after execution"""
    ray_nb = test_notebooks_dir / "ray_tune.ipynb"

    # First verify Ray isn't initialized
    assert not ray.is_initialized()

    # Run notebook
    result = notebook_tester.test_notebook(ray_nb)
    _, success, _, _ = result

    assert success is True

    # Verify Ray was properly shut down
    assert not ray.is_initialized()


def test_ray_tune_slow(notebook_tester, test_notebooks_dir):
    """Test that Ray resources are properly cleaned up after execution"""
    ray_nb = test_notebooks_dir / "ray_tune_slow.ipynb"

    # First verify Ray isn't initialized
    assert not ray.is_initialized()

    # Run notebook
    result = notebook_tester.test_notebook(ray_nb)
    _, success, message, _ = result

    # Should fail due to timeout
    assert success is False
    assert "timed out" in message.lower()

    # Verify Ray was properly shut down
    assert not ray.is_initialized()


def test_find_notebooks(notebook_tester):
    """Test that notebook discovery works correctly"""
    notebooks = notebook_tester.find_notebooks()

    # Should find all three test notebooks
    assert len(notebooks) == 4

    # All found files should be .ipynb files
    assert all(nb.suffix == ".ipynb" for nb in notebooks)

    # Should not include any checkpoints
    assert not any(".ipynb_checkpoints" in str(nb) for nb in notebooks)


def test_batch_execution(notebook_tester):
    """Test running multiple notebooks in parallel"""
    notebook_tester.run_tests(max_workers=2)

    # Check final statistics
    assert notebook_tester.successful >= 1  # At least basic notebook should succeed
    assert notebook_tester.no_more_time >= 1  # Timeout notebook should fail
    assert not ray.is_initialized()  # Ray should be cleaned up


def test_force_rerun(test_notebooks_dir, test_cache_dir):
    """Test that force=True bypasses cache"""
    # Create two testers - one normal, one forced
    normal_tester = NotebookTester(
        dir=test_notebooks_dir, timeout=5, cache_dir=test_cache_dir, force=False
    )
    force_tester = NotebookTester(
        dir=test_notebooks_dir, timeout=5, cache_dir=test_cache_dir, force=True
    )

    basic_nb = test_notebooks_dir / "basic.ipynb"

    result2 = normal_tester.test_notebook(basic_nb)
    assert result2[3] is True  # Second run, should be cached

    # Run with force=True
    result3 = force_tester.test_notebook(basic_nb)
    assert result3[3] is False  # Should not use cache when forced
