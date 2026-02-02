"""Audio recording and conversion utilities for PharmaTranscribe AI."""

import io
import tempfile
from pathlib import Path
from typing import Optional

from pydub import AudioSegment


class AudioConversionError(Exception):
    """Raised when audio conversion fails."""

    pass


def convert_wav_to_mp3(
    wav_data: bytes,
    bitrate: str = "192k",
    sample_rate: int = 44100,
) -> bytes:
    """Convert WAV audio bytes to MP3 format.

    Args:
        wav_data: Raw WAV audio data as bytes
        bitrate: MP3 bitrate (default: "192k")
        sample_rate: Output sample rate in Hz (default: 44100)

    Returns:
        MP3 audio data as bytes

    Raises:
        AudioConversionError: If conversion fails
    """
    try:
        # Load WAV from bytes
        audio = AudioSegment.from_wav(io.BytesIO(wav_data))

        # Set sample rate if different
        if audio.frame_rate != sample_rate:
            audio = audio.set_frame_rate(sample_rate)

        # Export to MP3
        mp3_buffer = io.BytesIO()
        audio.export(mp3_buffer, format="mp3", bitrate=bitrate)
        mp3_buffer.seek(0)

        return mp3_buffer.read()

    except Exception as e:
        raise AudioConversionError(f"Failed to convert WAV to MP3: {e}") from e


def convert_audio_to_mp3(
    audio_data: bytes,
    input_format: str,
    bitrate: str = "192k",
    sample_rate: int = 44100,
) -> bytes:
    """Convert audio bytes from any format to MP3.

    Args:
        audio_data: Raw audio data as bytes
        input_format: Input format (e.g., "webm", "mp4", "wav")
        bitrate: MP3 bitrate (default: "192k")
        sample_rate: Output sample rate in Hz (default: 44100)

    Returns:
        MP3 audio data as bytes

    Raises:
        AudioConversionError: If conversion fails
    """
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_data), format=input_format)

        if audio.frame_rate != sample_rate:
            audio = audio.set_frame_rate(sample_rate)

        mp3_buffer = io.BytesIO()
        audio.export(mp3_buffer, format="mp3", bitrate=bitrate)
        mp3_buffer.seek(0)

        return mp3_buffer.read()

    except Exception as e:
        raise AudioConversionError(f"Failed to convert {input_format} to MP3: {e}") from e


def save_recording_as_mp3(
    wav_data: bytes,
    output_path: Optional[str] = None,
    bitrate: str = "192k",
) -> str:
    """Save WAV recording as MP3 file.

    Args:
        wav_data: Raw WAV audio data from st.audio_input
        output_path: Optional output path. If None, creates temp file.
        bitrate: MP3 bitrate (default: "192k")

    Returns:
        Path to the saved MP3 file

    Raises:
        AudioConversionError: If conversion or save fails
    """
    try:
        mp3_data = convert_wav_to_mp3(wav_data, bitrate)

        if output_path is None:
            # Create temporary file that won't be auto-deleted
            with tempfile.NamedTemporaryFile(
                suffix=".mp3",
                delete=False,
            ) as tmp:
                tmp.write(mp3_data)
                return tmp.name
        else:
            Path(output_path).write_bytes(mp3_data)
            return output_path

    except AudioConversionError:
        raise
    except Exception as e:
        raise AudioConversionError(f"Failed to save MP3: {e}") from e


def get_audio_duration_seconds(audio_data: bytes, format: str = "wav") -> float:
    """Get duration of audio data in seconds.

    Args:
        audio_data: Audio data as bytes
        format: Audio format ("wav" or "mp3")

    Returns:
        Duration in seconds

    Raises:
        ValueError: If format is not supported
    """
    if format == "wav":
        audio = AudioSegment.from_wav(io.BytesIO(audio_data))
    elif format == "mp3":
        audio = AudioSegment.from_mp3(io.BytesIO(audio_data))
    else:
        raise ValueError(f"Unsupported format: {format}")

    return len(audio) / 1000.0  # pydub uses milliseconds
