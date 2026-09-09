"""Integration tests for Gemini client with mocked API."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.gemini_client import (
    get_client,
    upload_audio,
    wait_for_active,
    transcribe,
    delete_file,
    FileProcessingError,
    call_with_retries,
)
from src import gemini_client as gemini_client_module


class TestGetClient:
    """Tests for get_client function."""

    @patch("src.gemini_client.genai.Client")
    def test_get_client_returns_client(self, mock_client_class):
        """Test that get_client returns a Client instance."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        result = get_client("test-api-key")

        mock_client_class.assert_called_once_with(api_key="test-api-key")
        assert result == mock_client


class TestUploadAudio:
    """Tests for upload_audio function."""

    def test_upload_calls_files_upload(self, mock_genai_client, mock_active_file, tmp_path):
        """Test that upload_audio calls client.files.upload with correct params."""
        audio_path = tmp_path / "audio.mp3"
        audio_path.write_bytes(b"fake audio bytes")
        mock_genai_client.files.upload.return_value = mock_active_file

        result = upload_audio(mock_genai_client, str(audio_path), "audio/mpeg")

        mock_genai_client.files.upload.assert_called_once_with(
            file=str(audio_path),
            config={"mime_type": "audio/mpeg"}
        )
        assert result == mock_active_file


class TestWaitForActive:
    """Tests for wait_for_active function."""

    def test_wait_for_active_immediate_success(self, mock_genai_client, mock_active_file):
        """Test immediate return when file is already ACTIVE."""
        mock_genai_client.files.get.return_value = mock_active_file

        result = wait_for_active(mock_genai_client, mock_active_file)

        assert result.state == "ACTIVE"
        mock_genai_client.files.get.assert_called_once_with(name="files/test123")

    @patch("time.sleep")
    def test_wait_for_active_polls_until_active(self, mock_sleep, mock_genai_client):
        """Test polling behavior until file becomes ACTIVE."""
        processing_file = Mock()
        processing_file.name = "files/test123"
        processing_file.state = "PROCESSING"

        active_file = Mock()
        active_file.name = "files/test123"
        active_file.state = "ACTIVE"

        # First call returns PROCESSING, second returns ACTIVE
        mock_genai_client.files.get.side_effect = [processing_file, active_file]

        result = wait_for_active(mock_genai_client, processing_file, poll_interval=0)

        assert result.state == "ACTIVE"
        assert mock_genai_client.files.get.call_count == 2

    @patch("time.sleep")
    @patch("time.time")
    def test_wait_for_active_timeout(self, mock_time, mock_sleep, mock_genai_client, mock_processing_file):
        """Test TimeoutError raised after timeout."""
        mock_genai_client.files.get.return_value = mock_processing_file
        # Simulate time passing beyond timeout
        mock_time.side_effect = [0, 0, 301]  # Start, first check, timeout exceeded

        with pytest.raises(TimeoutError) as exc_info:
            wait_for_active(
                mock_genai_client,
                mock_processing_file,
                timeout_seconds=300
            )

        assert "did not become ACTIVE" in str(exc_info.value)

    def test_wait_for_active_failed_state(self, mock_genai_client, mock_failed_file):
        """Test FileProcessingError raised when state is FAILED."""
        mock_genai_client.files.get.return_value = mock_failed_file

        with pytest.raises(FileProcessingError) as exc_info:
            wait_for_active(mock_genai_client, mock_failed_file)

        assert "processing failed" in str(exc_info.value)


class TestTranscribe:
    """Tests for transcribe function."""

    def test_transcribe_yields_chunks(self, mock_genai_client, mock_active_file, mock_transcript_response):
        """Test that transcribe yields text chunks."""
        mock_genai_client.models.generate_content_stream.return_value = iter(mock_transcript_response)

        chunks = list(transcribe(mock_genai_client, mock_active_file, "Test prompt"))

        assert len(chunks) == 4
        assert "Operator:" in chunks[0]
        assert "Vertex Pharmaceuticals" in chunks[1]
        assert "Keytruda" in chunks[3]

    def test_transcribe_calls_api_correctly(self, mock_genai_client, mock_active_file):
        """Test that transcribe calls the API with correct parameters."""
        mock_genai_client.models.generate_content_stream.return_value = iter([])

        list(transcribe(mock_genai_client, mock_active_file, "Test prompt"))

        mock_genai_client.models.generate_content_stream.assert_called_once()
        call_kwargs = mock_genai_client.models.generate_content_stream.call_args
        assert call_kwargs.kwargs["model"] == "gemini-3.5-flash"
        assert "Test prompt" in call_kwargs.kwargs["contents"]

    def test_transcribe_handles_empty_chunks(self, mock_genai_client, mock_active_file):
        """Test that empty text chunks are skipped."""
        chunks_with_empty = [
            Mock(text="Hello "),
            Mock(text=""),
            Mock(text=None),
            Mock(text="World"),
        ]
        mock_genai_client.models.generate_content_stream.return_value = iter(chunks_with_empty)

        result = list(transcribe(mock_genai_client, mock_active_file, "Test prompt"))

        assert result == ["Hello ", "World"]


class TestDeleteFile:
    """Tests for delete_file function."""

    def test_delete_file_success(self, mock_genai_client):
        """Test successful file deletion."""
        mock_genai_client.files.delete.return_value = None

        result = delete_file(mock_genai_client, "files/test123")

        assert result is True
        mock_genai_client.files.delete.assert_called_once_with(name="files/test123")

    def test_delete_file_handles_error(self, mock_genai_client):
        """Test that errors are handled gracefully."""
        mock_genai_client.files.delete.side_effect = Exception("File not found")

        result = delete_file(mock_genai_client, "files/nonexistent")

        assert result is False

    def test_delete_file_not_found(self, mock_genai_client):
        """Test handling of file not found."""
        mock_genai_client.files.delete.side_effect = Exception("404 Not Found")

        result = delete_file(mock_genai_client, "files/deleted")

        assert result is False


class TestFileProcessingError:
    """Tests for FileProcessingError exception."""

    def test_file_processing_error_message(self):
        """Test that FileProcessingError contains the right message."""
        error = FileProcessingError("Test error message")
        assert str(error) == "Test error message"

    def test_file_processing_error_is_exception(self):
        """Test that FileProcessingError is an Exception."""
        error = FileProcessingError("Test")
        assert isinstance(error, Exception)


class TestCallWithRetries:
    def test_returns_result_on_first_success(self):
        assert call_with_retries(lambda: 42) == 42

    def test_retries_on_connection_error_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(gemini_client_module.time, "sleep", lambda s: None)
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("transient network blip")
            return "ok"

        assert call_with_retries(flaky, attempts=3) == "ok"
        assert calls["n"] == 3

    def test_raises_after_exhausting_attempts(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(gemini_client_module.time, "sleep", sleeps.append)

        def always_fails():
            raise ConnectionError("still down")

        with pytest.raises(ConnectionError):
            call_with_retries(always_fails, attempts=3, base_delay=2.0)
        # Two sleeps between three attempts, exponential backoff
        assert sleeps == [2.0, 4.0]

    def test_non_retriable_error_propagates_immediately(self, monkeypatch):
        monkeypatch.setattr(gemini_client_module.time, "sleep", lambda s: None)
        calls = {"n": 0}

        def bad_request():
            calls["n"] += 1
            raise ValueError("bad input")

        with pytest.raises(ValueError):
            call_with_retries(bad_request, attempts=3)
        assert calls["n"] == 1


class TestWaitForActiveToleratesTransientErrors:
    def test_transient_poll_error_does_not_abort(self, monkeypatch):
        monkeypatch.setattr(gemini_client_module.time, "sleep", lambda s: None)

        active_file = MagicMock()
        active_file.state = "ACTIVE"
        pending_file = MagicMock()
        pending_file.state = "PROCESSING"
        pending_file.name = "files/test123"

        client = MagicMock()
        client.files.get.side_effect = [
            ConnectionError("network blip"),
            active_file,
        ]

        result = wait_for_active(client, pending_file, timeout_seconds=30, poll_interval=0)
        assert result is active_file
        assert client.files.get.call_count == 2
