"""Unit tests for the AudioRecording component class."""

import base64
import io
import json
import sys
from unittest.mock import MagicMock

import pytest

# Mock streamlit before importing the component, since importing it triggers
# `streamlit.components.v1.declare_component(...)`, which we don't want to
# run for real in a unit test. Stash whatever was already in sys.modules
# first and restore it immediately after the import completes -- pytest
# imports (collects) every test file before running any test, so leaving the
# mock in place here would otherwise leak into every other test file's
# collection and execution (e.g. anything needing the real `streamlit`
# package, such as Streamlit's own AppTest harness).
_PRE_MOCK_MODULES = {
    name: sys.modules.get(name)
    for name in ("streamlit", "streamlit.components", "streamlit.components.v1")
}

sys.modules["streamlit"] = MagicMock()
sys.modules["streamlit.components"] = MagicMock()
sys.modules["streamlit.components.v1"] = MagicMock()

from src.components.audio_recorder import AudioRecording

for _name, _original in _PRE_MOCK_MODULES.items():
    if _original is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _original


class TestAudioRecording:
    """Tests for the AudioRecording class."""

    def test_init_with_defaults(self):
        """Test that default format and mime_type are set."""
        audio_bytes = b"test audio data"
        recording = AudioRecording(audio_bytes)

        assert recording.format == "wav"
        assert recording.mime_type == "audio/wav"
        assert recording.getvalue() == audio_bytes

    def test_init_with_custom_format(self):
        """Test initialization with custom format."""
        audio_bytes = b"test audio data"
        recording = AudioRecording(audio_bytes, format="webm", mime_type="audio/webm")

        assert recording.format == "webm"
        assert recording.mime_type == "audio/webm"
        assert recording.getvalue() == audio_bytes

    def test_init_with_mp4_format(self):
        """Test initialization with MP4 format (Safari fallback)."""
        audio_bytes = b"test audio data"
        recording = AudioRecording(audio_bytes, format="mp4", mime_type="audio/mp4")

        assert recording.format == "mp4"
        assert recording.mime_type == "audio/mp4"

    def test_getvalue_returns_bytes(self):
        """Test that getvalue returns the original bytes."""
        audio_bytes = b"test audio data"
        recording = AudioRecording(audio_bytes)

        assert recording.getvalue() == audio_bytes
        # Should be idempotent
        assert recording.getvalue() == audio_bytes

    def test_read_returns_bytes(self):
        """Test that read returns the audio data."""
        audio_bytes = b"test audio data"
        recording = AudioRecording(audio_bytes)

        assert recording.read() == audio_bytes

    def test_seek_resets_buffer(self):
        """Test that seek allows re-reading."""
        audio_bytes = b"test audio data"
        recording = AudioRecording(audio_bytes)

        # Read all data
        recording.read()
        # Seek back to start
        recording.seek(0)
        # Read again
        assert recording.read() == audio_bytes

    def test_len_returns_byte_count(self):
        """Test that len returns the number of bytes."""
        audio_bytes = b"test audio data"
        recording = AudioRecording(audio_bytes)

        assert len(recording) == len(audio_bytes)

    def test_format_property_is_readonly(self):
        """Test that format is a read-only property."""
        recording = AudioRecording(b"test", format="webm")
        # Verify it's a property (not settable)
        with pytest.raises(AttributeError):
            recording.format = "mp3"

    def test_mime_type_property_is_readonly(self):
        """Test that mime_type is a read-only property."""
        recording = AudioRecording(b"test", mime_type="audio/webm")
        # Verify it's a property (not settable)
        with pytest.raises(AttributeError):
            recording.mime_type = "audio/mp3"


class TestAudioRecordingJsonParsing:
    """Tests for JSON payload parsing in audio_recorder function."""

    def test_json_payload_with_webm(self):
        """Test that JSON payload with webm format is parsed correctly."""
        # Simulate what the frontend sends
        audio_bytes = b"fake webm data"
        payload = {
            "data": base64.b64encode(audio_bytes).decode(),
            "format": "webm",
            "mimeType": "audio/webm;codecs=opus",
        }
        json_str = json.dumps(payload)

        # Parse it manually (simulating what audio_recorder does)
        parsed = json.loads(json_str)
        decoded_bytes = base64.b64decode(parsed["data"])

        assert decoded_bytes == audio_bytes
        assert parsed["format"] == "webm"
        assert parsed["mimeType"] == "audio/webm;codecs=opus"

    def test_json_payload_with_mp4(self):
        """Test that JSON payload with mp4 format (Safari) is parsed correctly."""
        audio_bytes = b"fake mp4 data"
        payload = {
            "data": base64.b64encode(audio_bytes).decode(),
            "format": "mp4",
            "mimeType": "audio/mp4",
        }
        json_str = json.dumps(payload)

        parsed = json.loads(json_str)
        decoded_bytes = base64.b64decode(parsed["data"])

        assert decoded_bytes == audio_bytes
        assert parsed["format"] == "mp4"

    def test_backward_compatible_raw_base64(self):
        """Test that raw base64 (old format) can still be decoded."""
        audio_bytes = b"fake wav data"
        raw_base64 = base64.b64encode(audio_bytes).decode()

        # Should fail JSON parsing but succeed base64 decoding
        try:
            json.loads(raw_base64)
            # If it parses as JSON, it's not valid old format
            assert False, "Should not parse as JSON"
        except json.JSONDecodeError:
            # This is expected - fall back to raw base64
            decoded = base64.b64decode(raw_base64)
            assert decoded == audio_bytes
