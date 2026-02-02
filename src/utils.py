"""File handling utilities for PharmaTranscribe AI."""

import os
from pathlib import Path

SUPPORTED_EXTENSIONS = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".mpeg": "audio/mpeg",
    ".webm": "audio/webm",
}


def get_mime_type(file_path: str) -> str:
    """Map file extension to MIME type.

    Args:
        file_path: Path to the audio file

    Returns:
        MIME type string for the file

    Raises:
        ValueError: If file extension is not supported
    """
    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(SUPPORTED_EXTENSIONS.keys())
        raise ValueError(f"Unsupported file type: {ext}. Supported types: {supported}")
    return SUPPORTED_EXTENSIONS[ext]


def validate_audio_file(file_path: str, max_size_mb: int = 200) -> bool:
    """Check file exists, has valid extension, and is under size limit.

    Args:
        file_path: Path to the audio file
        max_size_mb: Maximum file size in megabytes (default: 200)

    Returns:
        True if file is valid

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file extension is not supported or file exceeds size limit
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Validate extension (will raise ValueError if unsupported)
    get_mime_type(file_path)

    # Check file size
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValueError(f"File size ({size_mb:.1f}MB) exceeds limit ({max_size_mb}MB)")

    return True
