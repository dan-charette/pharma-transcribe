"""Durable on-disk storage for recordings and transcripts.

Everything the user creates (recordings, transcripts) is persisted here the
moment it exists, so a crash of the browser tab, the Streamlit server, or
the machine cannot lose it. All writes are atomic (temp file + os.replace).
"""

import hashlib
import os
import tempfile
from datetime import datetime
from pathlib import Path

from src.logging_config import get_logger

BASE_DIR = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = BASE_DIR / "recordings"
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"

_FINGERPRINT_SAMPLE = 65536

logger = get_logger("storage")


def new_session_stem(now: datetime | None = None) -> str:
    """Return a unique filename stem like 'session_20260731_141530'."""
    now = now or datetime.now()
    return now.strftime("session_%Y%m%d_%H%M%S")


def recording_fingerprint(data: bytes) -> str:
    """Cheap, stable identity for a recording.

    Hashes length + first/last 64KB so identity checks stay O(1)-ish even
    for multi-hundred-MB recordings re-checked on every Streamlit rerun.
    """
    h = hashlib.sha256()
    h.update(str(len(data)).encode("ascii"))
    h.update(data[:_FINGERPRINT_SAMPLE])
    h.update(data[-_FINGERPRINT_SAMPLE:])
    return h.hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes to path atomically (temp file in same dir + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def save_recording(
    data: bytes,
    fmt: str,
    stem: str | None = None,
    directory: Path | None = None,
) -> Path:
    """Persist raw recording bytes to disk atomically.

    Args:
        data: Compressed audio bytes (e.g. WebM/MP4 from the recorder).
        fmt: File extension without dot (e.g. "webm", "mp4").
        stem: Filename stem; generated from the current time if omitted.
        directory: Target directory (default: RECORDINGS_DIR).

    Returns:
        Path of the saved file.
    """
    directory = directory or RECORDINGS_DIR
    stem = stem or new_session_stem()
    path = directory / f"{stem}.{fmt}"
    _atomic_write_bytes(path, data)
    logger.info("Saved recording: %s (%d bytes)", path, len(data))
    return path


def save_transcript(
    text: str,
    stem: str,
    partial: bool = False,
    directory: Path | None = None,
) -> Path:
    """Persist transcript text to disk atomically.

    Partial checkpoints go to <stem>.partial.txt. The final save goes to
    <stem>.txt and removes the partial checkpoint if one exists.

    Returns:
        Path of the saved file.
    """
    directory = directory or TRANSCRIPTS_DIR
    suffix = ".partial.txt" if partial else ".txt"
    path = directory / f"{stem}{suffix}"
    _atomic_write_bytes(path, text.encode("utf-8"))
    if not partial:
        partial_path = directory / f"{stem}.partial.txt"
        try:
            partial_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove partial transcript: %s", partial_path)
    logger.info("Saved transcript: %s (%d chars, partial=%s)", path, len(text), partial)
    return path


def _list_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix != ".part"]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def list_recordings(directory: Path | None = None) -> list[Path]:
    """List saved recordings, newest first."""
    return _list_files(directory or RECORDINGS_DIR)


def list_transcripts(directory: Path | None = None) -> list[Path]:
    """List saved transcripts (including partials), newest first."""
    return _list_files(directory or TRANSCRIPTS_DIR)
