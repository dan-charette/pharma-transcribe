"""Unit tests for audio recording utilities."""

import io
import os
import subprocess
import tempfile
import wave

import pytest

from src.audio_recorder import (
    AudioConversionError,
    convert_audio_to_mp3,
    convert_file_to_mp3,
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


def _write_test_wav(path, seconds=1, framerate=8000):
    """Write a short silent WAV file using only the stdlib."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes(b"\x00\x00" * framerate * seconds)


class TestConvertFileToMp3:
    def test_converts_wav_file_to_mp3(self, tmp_path):
        wav_path = tmp_path / "input.wav"
        _write_test_wav(wav_path)

        result = convert_file_to_mp3(wav_path)

        assert result == tmp_path / "input.mp3"
        assert result.exists()
        assert result.stat().st_size > 0

    def test_explicit_output_path(self, tmp_path):
        wav_path = tmp_path / "input.wav"
        _write_test_wav(wav_path)
        out = tmp_path / "custom.mp3"

        result = convert_file_to_mp3(wav_path, output_path=out)

        assert result == out
        assert out.exists()

    def test_missing_input_raises_conversion_error(self, tmp_path):
        with pytest.raises(AudioConversionError):
            convert_file_to_mp3(tmp_path / "nope.wav")

    def test_corrupt_input_raises_conversion_error(self, tmp_path):
        bad = tmp_path / "bad.wav"
        bad.write_bytes(b"this is not audio")
        with pytest.raises(AudioConversionError):
            convert_file_to_mp3(bad)

    def test_missing_ffmpeg_raises_conversion_error(self, tmp_path, monkeypatch):
        wav_path = tmp_path / "input.wav"
        _write_test_wav(wav_path)

        def fake_run(*args, **kwargs):
            raise FileNotFoundError("ffmpeg")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(AudioConversionError, match="ffmpeg"):
            convert_file_to_mp3(wav_path)
