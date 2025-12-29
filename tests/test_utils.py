"""Unit tests for file handling utilities."""

import os
import tempfile
import pytest
from src.utils import get_mime_type, validate_audio_file, SUPPORTED_EXTENSIONS


class TestGetMimeType:
    """Tests for get_mime_type function."""

    def test_mp3_mime_type(self):
        """Test MP3 file returns correct MIME type."""
        assert get_mime_type("audio.mp3") == "audio/mpeg"
        assert get_mime_type("/path/to/file.mp3") == "audio/mpeg"

    def test_wav_mime_type(self):
        """Test WAV file returns correct MIME type."""
        assert get_mime_type("audio.wav") == "audio/wav"
        assert get_mime_type("/path/to/file.wav") == "audio/wav"

    def test_m4a_mime_type(self):
        """Test M4A file returns correct MIME type."""
        assert get_mime_type("audio.m4a") == "audio/mp4"
        assert get_mime_type("/path/to/file.m4a") == "audio/mp4"

    def test_mpeg_mime_type(self):
        """Test MPEG file returns correct MIME type."""
        assert get_mime_type("audio.mpeg") == "audio/mpeg"
        assert get_mime_type("/path/to/file.mpeg") == "audio/mpeg"

    def test_uppercase_extension(self):
        """Test that uppercase extensions work."""
        assert get_mime_type("audio.MP3") == "audio/mpeg"
        assert get_mime_type("audio.WAV") == "audio/wav"
        assert get_mime_type("audio.M4A") == "audio/mp4"

    def test_mixed_case_extension(self):
        """Test that mixed case extensions work."""
        assert get_mime_type("audio.Mp3") == "audio/mpeg"
        assert get_mime_type("audio.WaV") == "audio/wav"

    def test_unsupported_extension_raises_error(self):
        """Test that unsupported extensions raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_mime_type("audio.ogg")
        assert "Unsupported file type" in str(exc_info.value)
        assert ".ogg" in str(exc_info.value)

    def test_no_extension_raises_error(self):
        """Test that files without extension raise ValueError."""
        with pytest.raises(ValueError):
            get_mime_type("audiofile")

    def test_txt_extension_raises_error(self):
        """Test that text files raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_mime_type("document.txt")
        assert "Unsupported file type" in str(exc_info.value)


class TestValidateAudioFile:
    """Tests for validate_audio_file function."""

    def test_valid_mp3_file(self):
        """Test validation of a valid MP3 file."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(b"fake audio content")
            tmp_path = tmp.name

        try:
            assert validate_audio_file(tmp_path) is True
        finally:
            os.unlink(tmp_path)

    def test_valid_wav_file(self):
        """Test validation of a valid WAV file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"fake audio content")
            tmp_path = tmp.name

        try:
            assert validate_audio_file(tmp_path) is True
        finally:
            os.unlink(tmp_path)

    def test_file_not_found(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            validate_audio_file("/nonexistent/path/audio.mp3")
        assert "File not found" in str(exc_info.value)

    def test_unsupported_extension(self):
        """Test that unsupported extension raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"text content")
            tmp_path = tmp.name

        try:
            with pytest.raises(ValueError) as exc_info:
                validate_audio_file(tmp_path)
            assert "Unsupported file type" in str(exc_info.value)
        finally:
            os.unlink(tmp_path)

    def test_file_exceeds_size_limit(self):
        """Test that oversized file raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(b"x" * 1024)  # 1KB file
            tmp_path = tmp.name

        try:
            # Set a tiny limit to trigger the error
            with pytest.raises(ValueError) as exc_info:
                validate_audio_file(tmp_path, max_size_mb=0.0001)
            assert "exceeds limit" in str(exc_info.value)
        finally:
            os.unlink(tmp_path)

    def test_custom_size_limit(self):
        """Test that custom size limit is respected."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(b"x" * 1024)  # 1KB file
            tmp_path = tmp.name

        try:
            # Should pass with 1MB limit
            assert validate_audio_file(tmp_path, max_size_mb=1) is True
        finally:
            os.unlink(tmp_path)


class TestSupportedExtensions:
    """Tests for SUPPORTED_EXTENSIONS constant."""

    def test_supported_extensions_contains_mp3(self):
        """Test that MP3 is supported."""
        assert ".mp3" in SUPPORTED_EXTENSIONS

    def test_supported_extensions_contains_wav(self):
        """Test that WAV is supported."""
        assert ".wav" in SUPPORTED_EXTENSIONS

    def test_supported_extensions_contains_m4a(self):
        """Test that M4A is supported."""
        assert ".m4a" in SUPPORTED_EXTENSIONS

    def test_supported_extensions_contains_mpeg(self):
        """Test that MPEG is supported."""
        assert ".mpeg" in SUPPORTED_EXTENSIONS

    def test_supported_extensions_count(self):
        """Test that we have exactly 4 supported formats."""
        assert len(SUPPORTED_EXTENSIONS) == 4
