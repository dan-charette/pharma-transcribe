"""Audio recording and conversion utilities for PharmaTranscribe AI."""

import io
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from pydub import AudioSegment

from src.logging_config import get_logger

logger = get_logger("audio")


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


def convert_file_to_mp3(
    input_path,
    output_path=None,
    bitrate: str = "192k",
    timeout: int = 600,
) -> Path:
    """Convert an audio file to MP3 on disk using the ffmpeg CLI.

    Unlike the byte-based converters above, this streams file-to-file with
    constant memory, so hour-long recordings never inflate to raw PCM in RAM.

    Args:
        input_path: Path to the source audio file (webm/mp4/wav/...).
        output_path: Target MP3 path. Defaults to input path with .mp3 suffix.
        bitrate: MP3 bitrate (default: "192k").
        timeout: Max seconds to allow ffmpeg to run (default: 600).

    Returns:
        Path to the written MP3 file.

    Raises:
        AudioConversionError: If ffmpeg is missing, fails, or times out.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(".mp3")
    output_path = Path(output_path)

    if not input_path.exists():
        raise AudioConversionError(f"Input file not found: {input_path}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except FileNotFoundError as e:
        raise AudioConversionError(
            "ffmpeg not found. Install it (e.g. `brew install ffmpeg`) to enable MP3 export."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise AudioConversionError(f"ffmpeg timed out after {timeout}s converting {input_path.name}") from e

    if result.returncode != 0:
        stderr_tail = result.stderr.decode("utf-8", errors="replace")[-500:]
        raise AudioConversionError(f"ffmpeg failed converting {input_path.name}: {stderr_tail}")

    logger.info("Converted %s -> %s (%d bytes)", input_path.name, output_path.name, output_path.stat().st_size)
    return output_path
