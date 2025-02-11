# tests/test_notebooktester.py

from notebooktester.main import NotebookTester


def test_basic_notebook_execution(notebook_tester, test_notebooks_dir):
    """Test execution of a simple notebook"""
    basic_nb = test_notebooks_dir / "basic.ipynb"
    result = notebook_tester.test_notebook(basic_nb)

    assert result.success is True
    assert "Success" in result.message
    assert result.cached is False
    assert result.execution_time < result.timeout  # Should be faster


def test_cached_execution(notebook_tester, test_notebooks_dir):
    """Test that notebook results are properly cached"""
    basic_nb = test_notebooks_dir / "basic.ipynb"

    # Second execution should use cache
    result2 = notebook_tester.test_notebook(basic_nb)
    assert result2.cached is True  # Should be cached
    assert result2.success is True  # Should still be successful


def test_timeout_handling(notebook_tester, test_notebooks_dir, test_cache_dir):
    """Test that notebooks that take too long timeout properly"""
    timeout_nb = test_notebooks_dir / "timeout.ipynb"
    result = notebook_tester.test_notebook(timeout_nb)

    assert result.success is False
    assert "timed out" in result.message.lower()
    assert result.execution_time == float("inf")

    patient_tester = NotebookTester(
        dir=test_notebooks_dir, timeout=5, cache_dir=test_cache_dir, force=False
    )
    result2 = patient_tester.test_notebook(timeout_nb)
    assert result2.success is True  # Should succeed with longer timeout
    assert result2.execution_time < result2.timeout


def test_failing_notebook(notebook_tester, test_notebooks_dir):
    """Test that notebooks with errors fail appropriately"""
    failing_nb = test_notebooks_dir / "failing.ipynb"
    result = notebook_tester.test_notebook(failing_nb)

    assert result.success is False
    assert result.cached is False
    assert result.execution_time == float("inf")


def test_find_notebooks(notebook_tester):
    """Test that notebook discovery works correctly"""
    notebooks = notebook_tester.find_notebooks()

    # Should find all test notebooks
    assert len(notebooks) == 3

    # All found files should be .ipynb files
    assert all(nb.suffix == ".ipynb" for nb in notebooks)

    # Should not include any checkpoints
    assert not any(".ipynb_checkpoints" in str(nb) for nb in notebooks)


def test_batch_execution(test_notebooks_dir, test_cache_dir):
    """Test running multiple notebooks in parallel"""
    tester = NotebookTester(
        dir=test_notebooks_dir, timeout=1, cache_dir=test_cache_dir, force=True
    )
    tester.run_tests(max_workers=2)

    assert tester.successful >= 1
    assert tester.no_more_time >= 1


def test_force_rerun(test_notebooks_dir, test_cache_dir):
    """Test that force=True bypasses cache"""
    # Create two testers - one normal, one forced
    normal_tester = NotebookTester(
        dir=test_notebooks_dir, timeout=1, cache_dir=test_cache_dir, force=False
    )
    force_tester = NotebookTester(
        dir=test_notebooks_dir, timeout=1, cache_dir=test_cache_dir, force=True
    )

    basic_nb = test_notebooks_dir / "basic.ipynb"

    result2 = normal_tester.test_notebook(basic_nb)
    assert result2.cached is True  # Second run, should be cached

    # Run with force=True
    result3 = force_tester.test_notebook(basic_nb)
    assert result3.cached is False  # Should not use cache when forced
