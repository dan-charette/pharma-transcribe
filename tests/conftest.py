"""Pytest fixtures and mocks for PharmaTranscribe AI tests."""

import pytest
from unittest.mock import Mock, MagicMock


@pytest.fixture
def mock_genai_client():
    """Return a mocked genai.Client."""
    client = MagicMock()
    return client


@pytest.fixture
def mock_active_file():
    """Return a mock File object with state=ACTIVE."""
    file = Mock()
    file.name = "files/test123"
    file.state = "ACTIVE"
    file.uri = "https://generativelanguage.googleapis.com/v1/files/test123"
    return file


@pytest.fixture
def mock_processing_file():
    """Return a mock File object with state=PROCESSING."""
    file = Mock()
    file.name = "files/test456"
    file.state = "PROCESSING"
    return file


@pytest.fixture
def mock_failed_file():
    """Return a mock File object with state=FAILED."""
    file = Mock()
    file.name = "files/test789"
    file.state = "FAILED"
    return file


@pytest.fixture
def sample_keywords():
    """Return sample pharmaceutical keywords."""
    return "Keytruda, pembrolizumab, VRTX, Vertex Pharmaceuticals, tezacaftor"


@pytest.fixture
def mock_transcript_response():
    """Return a mock streaming response with transcript chunks."""
    chunks = [
        Mock(text="[00:00] Operator: Good morning and welcome to the "),
        Mock(text="Q3 earnings call for Vertex Pharmaceuticals.\n\n"),
        Mock(text="[00:15] CEO: Thank you. Today we'll discuss our "),
        Mock(text="Keytruda partnership and tezacaftor results."),
    ]
    return chunks
