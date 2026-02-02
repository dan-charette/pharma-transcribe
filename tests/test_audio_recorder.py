"""Unit tests for audio recording utilities."""

import io
import os
import tempfile

import pytest

from src.audio_recorder import (
    AudioConversionError,
    convert_audio_to_mp3,
    convert_wav_to_mp3,
    get_audio_duration_seconds,
    save_recording_as_mp3,
)


@pytest.fixture
def sample_wav_bytes():
    """Generate minimal valid WAV file bytes for testing."""
    from pydub import AudioSegment
    from pydub.generators import Sine

    # Generate 1 second of 440Hz sine wave
    audio = Sine(440).to_audio_segment(duration=1000)

    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    buffer.seek(0)
    return buffer.read()


class TestConvertWavToMp3:
    """Tests for convert_wav_to_mp3 function."""

    def test_converts_valid_wav_to_mp3(self, sample_wav_bytes):
        """Test that valid WAV converts to MP3."""
        result = convert_wav_to_mp3(sample_wav_bytes)
        assert result is not None
        assert len(result) > 0
        # MP3 files start with ID3 tag or sync word
        assert result[:3] == b"ID3" or result[:2] == b"\xff\xfb"

    def test_raises_error_on_invalid_input(self):
        """Test that invalid input raises AudioConversionError."""
        with pytest.raises(AudioConversionError):
            convert_wav_to_mp3(b"not valid wav data")

    def test_respects_bitrate_parameter(self, sample_wav_bytes):
        """Test that different bitrates produce different sizes."""
        low_bitrate = convert_wav_to_mp3(sample_wav_bytes, bitrate="64k")
        high_bitrate = convert_wav_to_mp3(sample_wav_bytes, bitrate="320k")
        # Higher bitrate should produce larger file
        assert len(high_bitrate) > len(low_bitrate)


class TestSaveRecordingAsMp3:
    """Tests for save_recording_as_mp3 function."""

    def test_creates_temp_file_when_no_path_given(self, sample_wav_bytes):
        """Test that temp file is created when no output path specified."""
        result_path = save_recording_as_mp3(sample_wav_bytes)
        try:
            assert os.path.exists(result_path)
            assert result_path.endswith(".mp3")
        finally:
            os.unlink(result_path)

    def test_saves_to_specified_path(self, sample_wav_bytes):
        """Test saving to a specific output path."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            output_path = tmp.name

        try:
            result_path = save_recording_as_mp3(sample_wav_bytes, output_path)
            assert result_path == output_path
            assert os.path.exists(result_path)
        finally:
            os.unlink(output_path)

    def test_raises_error_on_invalid_wav(self):
        """Test that invalid WAV data raises error."""
        with pytest.raises(AudioConversionError):
            save_recording_as_mp3(b"invalid wav data")


class TestConvertAudioToMp3:
    """Tests for convert_audio_to_mp3 function."""

    def test_converts_wav_to_mp3(self, sample_wav_bytes):
        """Test that WAV converts to MP3 using generic function."""
        result = convert_audio_to_mp3(sample_wav_bytes, input_format="wav")
        assert result is not None
        assert len(result) > 0
        # MP3 files start with ID3 tag or sync word
        assert result[:3] == b"ID3" or result[:2] == b"\xff\xfb"

    def test_raises_error_on_invalid_input(self):
        """Test that invalid input raises AudioConversionError."""
        with pytest.raises(AudioConversionError):
            convert_audio_to_mp3(b"not valid audio data", input_format="wav")

    def test_raises_error_with_format_in_message(self):
        """Test that error message includes the input format."""
        with pytest.raises(AudioConversionError, match="webm"):
            convert_audio_to_mp3(b"not valid audio data", input_format="webm")

    def test_respects_bitrate_parameter(self, sample_wav_bytes):
        """Test that different bitrates produce different sizes."""
        low_bitrate = convert_audio_to_mp3(sample_wav_bytes, input_format="wav", bitrate="64k")
        high_bitrate = convert_audio_to_mp3(sample_wav_bytes, input_format="wav", bitrate="320k")
        # Higher bitrate should produce larger file
        assert len(high_bitrate) > len(low_bitrate)


class TestGetAudioDurationSeconds:
    """Tests for get_audio_duration_seconds function."""

    def test_returns_correct_duration_for_wav(self, sample_wav_bytes):
        """Test that duration is approximately correct for WAV."""
        duration = get_audio_duration_seconds(sample_wav_bytes, format="wav")
        # Our fixture creates 1 second of audio
        assert 0.9 <= duration <= 1.1

    def test_returns_correct_duration_for_mp3(self, sample_wav_bytes):
        """Test that duration is approximately correct for MP3."""
        mp3_data = convert_wav_to_mp3(sample_wav_bytes)
        duration = get_audio_duration_seconds(mp3_data, format="mp3")
        # Should still be approximately 1 second
        assert 0.9 <= duration <= 1.1

    def test_raises_error_for_unsupported_format(self, sample_wav_bytes):
        """Test that unsupported format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported format"):
            get_audio_duration_seconds(sample_wav_bytes, format="ogg")
