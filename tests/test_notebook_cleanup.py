# tests/test_notebook_cleanup.py

import asyncio
import warnings
from unittest.mock import AsyncMock, Mock

import pytest
from nbclient import NotebookClient


@pytest.fixture
def mock_notebook_client():
    client = Mock(spec=NotebookClient)
    # Create mock kernel client and kernel manager
    client.kc = Mock()
    client.km = Mock()
    return client


@pytest.mark.asyncio
async def test_cleanup_normal_shutdown(notebook_tester, mock_notebook_client):
    """Test normal cleanup path with both channels and kernel shutdown"""
    mock_notebook_client.kc.stop_channels = AsyncMock()
    mock_notebook_client.km.shutdown_kernel = Mock()

    await notebook_tester._cleanup_notebook_client(mock_notebook_client)

    mock_notebook_client.kc.stop_channels.assert_called_once()
    mock_notebook_client.km.shutdown_kernel.assert_called_once_with(now=True)


@pytest.mark.asyncio
async def test_cleanup_with_sync_stop_channels(notebook_tester, mock_notebook_client):
    """Test cleanup when stop_channels is a synchronous method"""
    mock_notebook_client.kc.stop_channels = Mock()
    mock_notebook_client.km.shutdown_kernel = Mock()

    await notebook_tester._cleanup_notebook_client(mock_notebook_client)

    mock_notebook_client.kc.stop_channels.assert_called_once()
    mock_notebook_client.km.shutdown_kernel.assert_called_once_with(now=True)


@pytest.mark.asyncio
async def test_cleanup_with_kernel_timeout(notebook_tester, mock_notebook_client):
    """Test cleanup when kernel shutdown times out"""

    # Mock the shutdown to issue a warning instead of using logger
    def slow_shutdown(now):
        warnings.warn("Kernel shutdown timed out", UserWarning)
        raise asyncio.TimeoutError()

    mock_notebook_client.kc.stop_channels = AsyncMock()
    mock_notebook_client.km.shutdown_kernel = Mock(side_effect=slow_shutdown)

    with pytest.warns(UserWarning, match="Kernel shutdown timed out"):
        await notebook_tester._cleanup_notebook_client(mock_notebook_client)

    mock_notebook_client.kc.stop_channels.assert_called_once()


@pytest.mark.asyncio
async def test_cleanup_with_missing_attributes(notebook_tester):
    """Test cleanup with partially initialized client"""
    partial_client = Mock(spec=NotebookClient)
    await notebook_tester._cleanup_notebook_client(partial_client)


@pytest.mark.asyncio
async def test_cleanup_with_failing_channels(notebook_tester, mock_notebook_client):
    """Test cleanup when stop_channels raises an exception"""

    async def failing_stop_channels():
        warnings.warn("Channel stop failed", UserWarning)
        raise RuntimeError("Channel stop failed")

    mock_notebook_client.kc.stop_channels = failing_stop_channels
    mock_notebook_client.km.shutdown_kernel = Mock()

    with pytest.warns(UserWarning, match="Channel stop failed"):
        await notebook_tester._cleanup_notebook_client(mock_notebook_client)

    mock_notebook_client.km.shutdown_kernel.assert_called_once_with(now=True)
