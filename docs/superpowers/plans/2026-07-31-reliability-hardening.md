# Reliability Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PharmaTranscribe AI crash-proof for long (60+ minute) recordings: every recording and transcript is persisted to disk the moment it exists, memory blowups are eliminated, everything is logged, and crashed browser recordings are recoverable.

**Architecture:** The Streamlit app (`app.py`) currently holds recordings and transcripts only in memory/session state, and re-runs an in-memory pydub→ffmpeg conversion (decompressing 60 min of audio to ~300MB of PCM) on every rerun — the likely OOM crash. We fix this with: (1) a `src/storage.py` module that durably persists recordings/transcripts with atomic writes, (2) a file-to-file ffmpeg conversion that runs once per recording in constant memory, (3) rotating-file logging, (4) retry/backoff in the Gemini client, (5) incremental transcript checkpointing during streaming, and (6) browser-side IndexedDB checkpointing of recorder chunks with a crash-recovery UI.

**Tech Stack:** Python 3.13, Streamlit 1.52 (incl. `streamlit.testing.v1.AppTest`), pydub, ffmpeg (CLI, at `/opt/homebrew/bin/ffmpeg`), google-genai SDK, fpdf2, pytest. Frontend: vanilla JS Streamlit custom component (MediaRecorder, IndexedDB).

## Global Constraints

- Python interpreter for ALL commands: `/Users/danielcharette/Git/Transcriber/venv/bin/python` (the worktree has no venv of its own). Run tests from the repo root as: `/Users/danielcharette/Git/Transcriber/venv/bin/python -m pytest tests/ -q`
- No new entries in `requirements.txt` — use stdlib + already-installed packages only.
- Never log the API key. Never log audio bytes or transcript text content in log lines — log sizes, paths, durations, and states only.
- All persisted artifacts (recordings, transcripts) must be written atomically: write to a temp file in the same directory, then `os.replace` to the final path.
- Existing public interfaces must keep working: `AudioRecording`, `convert_audio_to_mp3`, `convert_wav_to_mp3`, `save_recording_as_mp3`, `get_mime_type`, `build_transcription_prompt`, and all `src/gemini_client.py` function signatures. The pre-existing test suite (67 currently-passing tests) must remain green after every task.
- Two tests in `tests/test_prompts.py` fail BEFORE this plan starts (`test_build_prompt_empty_keywords`, `test_build_prompt_contains_timestamp_instructions` — stale assertions). Task 5 fixes them. Tasks 1–4 must not touch them; from Task 5 onward the whole suite must be green.
- Commit at the end of every task with a descriptive message.
- Log file lines use the logger name `transcriber`; log directory is `logs/` at repo root; recordings in `recordings/`; transcripts in `transcripts/` — all three git-ignored (Task 1 adds them).

---

### Task 1: Logging infrastructure

**Files:**
- Create: `src/logging_config.py`
- Create: `tests/test_logging_config.py`
- Modify: `.gitignore` (append artifact directories)

**Interfaces:**
- Produces: `setup_logging(level: int = logging.INFO) -> logging.Logger` — idempotent (safe to call on every Streamlit rerun; never duplicates handlers), returns the `"transcriber"` logger writing to `logs/transcriber.log` (rotating, 5MB × 3 backups) and to stderr.
- Produces: `get_logger(name: str = "") -> logging.Logger` — returns `"transcriber"` or `"transcriber.<name>"` child logger.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_logging_config.py`:

```python
"""Tests for centralized logging configuration."""

import logging

from src import logging_config
from src.logging_config import get_logger, setup_logging


class TestSetupLogging:
    def test_returns_transcriber_logger(self, tmp_path, monkeypatch):
        monkeypatch.setattr(logging_config, "LOG_DIR", tmp_path)
        monkeypatch.setattr(logging_config, "LOG_FILE", tmp_path / "transcriber.log")
        logger = setup_logging()
        try:
            assert logger.name == "transcriber"
        finally:
            for h in list(logger.handlers):
                logger.removeHandler(h)
                h.close()

    def test_idempotent_no_duplicate_handlers(self, tmp_path, monkeypatch):
        monkeypatch.setattr(logging_config, "LOG_DIR", tmp_path)
        monkeypatch.setattr(logging_config, "LOG_FILE", tmp_path / "transcriber.log")
        logger = setup_logging()
        try:
            count_first = len(logger.handlers)
            logger_again = setup_logging()
            assert logger_again is logger
            assert len(logger.handlers) == count_first
        finally:
            for h in list(logger.handlers):
                logger.removeHandler(h)
                h.close()

    def test_writes_to_log_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(logging_config, "LOG_DIR", tmp_path)
        monkeypatch.setattr(logging_config, "LOG_FILE", tmp_path / "transcriber.log")
        logger = setup_logging()
        try:
            logger.info("hello from test")
            for h in logger.handlers:
                h.flush()
            assert (tmp_path / "transcriber.log").exists()
            content = (tmp_path / "transcriber.log").read_text(encoding="utf-8")
            assert "hello from test" in content
            assert "INFO" in content
        finally:
            for h in list(logger.handlers):
                logger.removeHandler(h)
                h.close()

    def test_creates_log_directory(self, tmp_path, monkeypatch):
        log_dir = tmp_path / "nested" / "logs"
        monkeypatch.setattr(logging_config, "LOG_DIR", log_dir)
        monkeypatch.setattr(logging_config, "LOG_FILE", log_dir / "transcriber.log")
        logger = setup_logging()
        try:
            assert log_dir.exists()
        finally:
            for h in list(logger.handlers):
                logger.removeHandler(h)
                h.close()


class TestGetLogger:
    def test_default_returns_root_transcriber(self):
        assert get_logger().name == "transcriber"

    def test_named_returns_child(self):
        assert get_logger("storage").name == "transcriber.storage"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/danielcharette/Git/Transcriber/venv/bin/python -m pytest tests/test_logging_config.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'src.logging_config'`

- [ ] **Step 3: Write the implementation**

Create `src/logging_config.py`:

```python
"""Centralized logging configuration for PharmaTranscribe AI.

setup_logging() is idempotent so it can be called at the top of app.py,
which Streamlit re-executes on every rerun, without duplicating handlers.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "transcriber.log"

_LOGGER_NAME = "transcriber"
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the application logger.

    Safe to call repeatedly (e.g., on every Streamlit rerun): handlers are
    only attached the first time.

    Args:
        level: Logging level for the logger (default: INFO)

    Returns:
        The configured "transcriber" logger.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_FORMAT)

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def get_logger(name: str = "") -> logging.Logger:
    """Return the app logger or a named child of it.

    Args:
        name: Optional child name (e.g., "storage" -> "transcriber.storage")

    Returns:
        Logger instance.
    """
    if name:
        return logging.getLogger(f"{_LOGGER_NAME}.{name}")
    return logging.getLogger(_LOGGER_NAME)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/danielcharette/Git/Transcriber/venv/bin/python -m pytest tests/test_logging_config.py -v`
Expected: 6 PASS

- [ ] **Step 5: Append artifact directories to `.gitignore`**

Append this block to the end of `.gitignore`:

```
# App artifacts (auto-saved recordings, transcripts, logs)
recordings/
transcripts/
logs/
```

- [ ] **Step 6: Run the full suite (regression check)**

Run: `/Users/danielcharette/Git/Transcriber/venv/bin/python -m pytest tests/ -q`
Expected: 73 passed, 2 failed — the ONLY failures are the two pre-existing `tests/test_prompts.py` failures named in Global Constraints.

- [ ] **Step 7: Commit**

```bash
git add src/logging_config.py tests/test_logging_config.py .gitignore
git commit -m "Add rotating-file logging infrastructure"
```

---

### Task 2: Durable storage module

**Files:**
- Create: `src/storage.py`
- Create: `tests/test_storage.py`

**Interfaces:**
- Produces: `new_session_stem(now: datetime | None = None) -> str` — returns e.g. `"session_20260731_141530"`.
- Produces: `recording_fingerprint(data: bytes) -> str` — cheap stable content identity (hex digest).
- Produces: `save_recording(data: bytes, fmt: str, stem: str | None = None, directory: Path | None = None) -> Path` — atomic write to `recordings/<stem>.<fmt>`, returns final path.
- Produces: `save_transcript(text: str, stem: str, partial: bool = False, directory: Path | None = None) -> Path` — atomic write; partial goes to `<stem>.partial.txt`, final goes to `<stem>.txt` AND removes the partial file if present.
- Produces: `list_recordings(directory: Path | None = None) -> list[Path]`, `list_transcripts(directory: Path | None = None) -> list[Path]` — newest-first by mtime; return `[]` when the directory does not exist; never include `.part` temp files.
- Produces: module constants `RECORDINGS_DIR`, `TRANSCRIPTS_DIR` (repo-root `recordings/`, `transcripts/`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_storage.py`:

```python
"""Tests for durable on-disk storage of recordings and transcripts."""

from datetime import datetime

import pytest

from src.storage import (
    list_recordings,
    list_transcripts,
    new_session_stem,
    recording_fingerprint,
    save_recording,
    save_transcript,
)


class TestNewSessionStem:
    def test_format(self):
        stem = new_session_stem(datetime(2026, 7, 31, 14, 15, 30))
        assert stem == "session_20260731_141530"

    def test_defaults_to_now(self):
        stem = new_session_stem()
        assert stem.startswith("session_")
        assert len(stem) == len("session_20260731_141530")


class TestRecordingFingerprint:
    def test_stable_for_same_data(self):
        data = b"abc" * 100000
        assert recording_fingerprint(data) == recording_fingerprint(data)

    def test_differs_for_different_data(self):
        assert recording_fingerprint(b"aaaa") != recording_fingerprint(b"bbbb")

    def test_differs_for_different_length_same_prefix(self):
        base = b"x" * 200000
        assert recording_fingerprint(base) != recording_fingerprint(base + b"y")

    def test_handles_small_payloads(self):
        assert recording_fingerprint(b"") != recording_fingerprint(b"a")


class TestSaveRecording:
    def test_saves_bytes_to_expected_path(self, tmp_path):
        path = save_recording(b"audio-bytes", "webm", stem="session_x", directory=tmp_path)
        assert path == tmp_path / "session_x.webm"
        assert path.read_bytes() == b"audio-bytes"

    def test_generates_stem_when_missing(self, tmp_path):
        path = save_recording(b"data", "mp4", directory=tmp_path)
        assert path.name.startswith("session_")
        assert path.suffix == ".mp4"
        assert path.read_bytes() == b"data"

    def test_creates_directory(self, tmp_path):
        target = tmp_path / "does" / "not" / "exist"
        path = save_recording(b"data", "webm", stem="s", directory=target)
        assert path.exists()

    def test_no_leftover_temp_files(self, tmp_path):
        save_recording(b"data", "webm", stem="s", directory=tmp_path)
        leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".part"]
        assert leftovers == []


class TestSaveTranscript:
    def test_final_transcript_path_and_content(self, tmp_path):
        path = save_transcript("hello transcript", "session_x", directory=tmp_path)
        assert path == tmp_path / "session_x.txt"
        assert path.read_text(encoding="utf-8") == "hello transcript"

    def test_partial_transcript_uses_partial_suffix(self, tmp_path):
        path = save_transcript("partial text", "session_x", partial=True, directory=tmp_path)
        assert path == tmp_path / "session_x.partial.txt"
        assert path.read_text(encoding="utf-8") == "partial text"

    def test_final_save_removes_partial(self, tmp_path):
        save_transcript("partial text", "session_x", partial=True, directory=tmp_path)
        save_transcript("final text", "session_x", directory=tmp_path)
        assert not (tmp_path / "session_x.partial.txt").exists()
        assert (tmp_path / "session_x.txt").read_text(encoding="utf-8") == "final text"

    def test_partial_overwrites_previous_partial(self, tmp_path):
        save_transcript("v1", "session_x", partial=True, directory=tmp_path)
        path = save_transcript("v1 v2", "session_x", partial=True, directory=tmp_path)
        assert path.read_text(encoding="utf-8") == "v1 v2"


class TestListing:
    def test_missing_directory_returns_empty(self, tmp_path):
        assert list_recordings(directory=tmp_path / "nope") == []
        assert list_transcripts(directory=tmp_path / "nope") == []

    def test_newest_first(self, tmp_path):
        import os
        import time

        a = save_recording(b"a", "webm", stem="older", directory=tmp_path)
        b = save_recording(b"b", "webm", stem="newer", directory=tmp_path)
        now = time.time()
        os.utime(a, (now - 100, now - 100))
        os.utime(b, (now, now))
        listed = list_recordings(directory=tmp_path)
        assert [p.name for p in listed] == ["newer.webm", "older.webm"]

    def test_excludes_part_files(self, tmp_path):
        save_recording(b"a", "webm", stem="real", directory=tmp_path)
        (tmp_path / "junk.part").write_bytes(b"tmp")
        listed = list_recordings(directory=tmp_path)
        assert [p.name for p in listed] == ["real.webm"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/danielcharette/Git/Transcriber/venv/bin/python -m pytest tests/test_storage.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'src.storage'`

- [ ] **Step 3: Write the implementation**

Create `src/storage.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/danielcharette/Git/Transcriber/venv/bin/python -m pytest tests/test_storage.py -v`
Expected: all PASS (17 tests)

- [ ] **Step 5: Run the full suite (regression check)**

Run: `/Users/danielcharette/Git/Transcriber/venv/bin/python -m pytest tests/ -q`
Expected: only the two pre-existing `tests/test_prompts.py` failures.

- [ ] **Step 6: Commit**

```bash
git add src/storage.py tests/test_storage.py
git commit -m "Add durable storage module with atomic writes"
```

---

### Task 3: Constant-memory file-based MP3 conversion

**Files:**
- Modify: `src/audio_recorder.py` (add one function; do NOT change existing functions)
- Modify: `tests/test_audio_recorder.py` (append a new test class)

**Interfaces:**
- Produces: `convert_file_to_mp3(input_path: Path | str, output_path: Path | str | None = None, bitrate: str = "192k", timeout: int = 600) -> Path` — converts on disk via the ffmpeg CLI (no in-memory PCM), returns the output path. Default output: input path with `.mp3` suffix. Raises `AudioConversionError` on missing ffmpeg, non-zero exit, or timeout.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_audio_recorder.py`:

```python
import subprocess
import wave

from src.audio_recorder import convert_file_to_mp3


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
```

Note: `pytest` and `AudioConversionError` are already imported at the top of this test file; add any missing imports (`subprocess`, `wave`, `convert_file_to_mp3`) to the existing import block at the top of the file rather than mid-file.

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/danielcharette/Git/Transcriber/venv/bin/python -m pytest tests/test_audio_recorder.py -v -k ConvertFileToMp3`
Expected: FAIL/ERROR with `ImportError: cannot import name 'convert_file_to_mp3'`

- [ ] **Step 3: Write the implementation**

Add to `src/audio_recorder.py` (top of file: add `import subprocess` and `from src.logging_config import get_logger`; add module-level `logger = get_logger("audio")` after imports). Then append:

```python
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
```

(`Path` is already imported in this module.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/danielcharette/Git/Transcriber/venv/bin/python -m pytest tests/test_audio_recorder.py -v`
Expected: all PASS (existing tests + 5 new)

- [ ] **Step 5: Run the full suite (regression check)**

Run: `/Users/danielcharette/Git/Transcriber/venv/bin/python -m pytest tests/ -q`
Expected: only the two pre-existing `tests/test_prompts.py` failures.

- [ ] **Step 6: Commit**

```bash
git add src/audio_recorder.py tests/test_audio_recorder.py
git commit -m "Add constant-memory file-based MP3 conversion via ffmpeg CLI"
```

---

### Task 4: Gemini client hardening (retries + tolerant polling + logging)

**Files:**
- Modify: `src/gemini_client.py`
- Modify: `tests/test_gemini.py` (append new test classes)

**Interfaces:**
- Produces: `call_with_retries(fn, attempts: int = 3, base_delay: float = 2.0, description: str = "operation")` — calls `fn()` and retries on retriable errors (google-genai `ServerError`, `ConnectionError`, `TimeoutError`, `OSError`) with exponential backoff (`base_delay * 2**attempt_index` seconds); re-raises the last error after exhausting attempts; non-retriable errors propagate immediately.
- Existing signatures UNCHANGED: `get_client`, `upload_audio`, `wait_for_active`, `transcribe`, `delete_file` keep their exact current signatures and return types (app.py and existing tests depend on them).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gemini.py`:

```python
from src.gemini_client import call_with_retries
from src import gemini_client as gemini_client_module


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
```

Note: `pytest`, `MagicMock`, and `wait_for_active` are already imported at the top of `tests/test_gemini.py` — check the existing import block and only add what's missing (`call_with_retries` and the module alias import).

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/danielcharette/Git/Transcriber/venv/bin/python -m pytest tests/test_gemini.py -v -k "CallWithRetries or Transient"`
Expected: FAIL/ERROR with `ImportError: cannot import name 'call_with_retries'`

- [ ] **Step 3: Write the implementation**

Modify `src/gemini_client.py`:

1. Move `import time` from inside `wait_for_active` to the module top; add `from src.logging_config import get_logger`; add module-level `logger = get_logger("gemini")`.
2. Add after the exception class:

```python
RETRIABLE_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)


def _is_retriable(exc: Exception) -> bool:
    """Retriable = transient network/server trouble, not caller mistakes."""
    if isinstance(exc, genai_errors.ServerError):
        return True
    # TimeoutError raised by our own wait_for_active must not be swallowed
    # by retry wrappers; plain socket timeouts are OSError subclasses anyway.
    return isinstance(exc, RETRIABLE_EXCEPTIONS)


def call_with_retries(fn, attempts: int = 3, base_delay: float = 2.0, description: str = "operation"):
    """Call fn(), retrying transient failures with exponential backoff.

    Args:
        fn: Zero-argument callable.
        attempts: Total attempts including the first (default: 3).
        base_delay: First backoff delay in seconds; doubles each retry.
        description: Human-readable label for log lines.

    Returns:
        Whatever fn() returns.

    Raises:
        The last exception if all attempts fail, or immediately for
        non-retriable errors.
    """
    last_exc = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if not _is_retriable(exc):
                raise
            last_exc = exc
            if attempt < attempts - 1:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "%s failed (attempt %d/%d): %s -- retrying in %.1fs",
                    description, attempt + 1, attempts, exc, delay,
                )
                time.sleep(delay)
    logger.error("%s failed after %d attempts: %s", description, attempts, last_exc)
    raise last_exc
```

This requires `from google.genai import errors as genai_errors` at the top (add it — the module currently imports only `genai` and `types`).

3. Wrap the upload inside `upload_audio` (signature unchanged):

```python
def upload_audio(client: genai.Client, file_path: str, mime_type: str) -> types.File:
    """Upload file to Gemini File API with retries on transient failures."""
    import os

    size = os.path.getsize(file_path)
    logger.info("Uploading %s (%d bytes, %s)", file_path, size, mime_type)
    result = call_with_retries(
        lambda: client.files.upload(file=file_path, config={"mime_type": mime_type}),
        description="Gemini file upload",
    )
    logger.info("Upload complete: %s", result.name)
    return result
```

(Move `import os` to module top rather than inside the function.)

4. Make the polling loop in `wait_for_active` tolerate transient errors — replace the loop body so a retriable exception from `client.files.get` logs a warning and continues polling (still subject to the overall timeout), while keeping the exact current behavior for `ACTIVE`/`FAILED`/timeout:

```python
    while True:
        try:
            current_file = client.files.get(name=file.name)
        except Exception as exc:
            if not _is_retriable(exc):
                raise
            logger.warning("Transient error polling file state (%s); will retry", exc)
            current_file = None

        if current_file is not None:
            if current_file.state == "ACTIVE":
                logger.info("File active: %s", file.name)
                return current_file

            if current_file.state == "FAILED":
                raise FileProcessingError(
                    f"File processing failed: {file.name}. The file may be corrupted or in an unsupported format."
                )

        elapsed = time.time() - start_time
        if elapsed >= timeout_seconds:
            state = current_file.state if current_file is not None else "UNKNOWN"
            raise TimeoutError(
                f"File did not become ACTIVE within {timeout_seconds} seconds. "
                f"Current state: {state}"
            )

        time.sleep(poll_interval)
```

5. Add logging to `transcribe` (log start with model name and file name before creating the stream) and to `delete_file` (in the `except` branch: `logger.warning("Failed to delete Gemini file %s", file_name)` before returning False). Signatures unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/danielcharette/Git/Transcriber/venv/bin/python -m pytest tests/test_gemini.py -v`
Expected: all PASS (existing + 5 new)

- [ ] **Step 5: Run the full suite (regression check)**

Run: `/Users/danielcharette/Git/Transcriber/venv/bin/python -m pytest tests/ -q`
Expected: only the two pre-existing `tests/test_prompts.py` failures.

- [ ] **Step 6: Commit**

```bash
git add src/gemini_client.py tests/test_gemini.py
git commit -m "Harden Gemini client: retries, tolerant polling, logging"
```

---

### Task 5: Fix stale prompt tests

**Files:**
- Modify: `tests/test_prompts.py` (two test methods only)

**Interfaces:** none (test-only change; `src/prompts.py` is NOT modified).

Context: the prompt template was rewritten at some point; two tests still assert the old prompt's literal text (`"TIMESTAMPS"` section header). The template's actual content (see `src/prompts.py`) includes `"TRANSCRIPTION INSTRUCTIONS"`, the `[MM:SS]` format instruction, and `[00:00]` in the example. The tests must be updated to assert real current behavior — do not change the prompt itself.

- [ ] **Step 1: Update the two failing tests**

In `tests/test_prompts.py`, replace `test_build_prompt_empty_keywords` with:

```python
    def test_build_prompt_empty_keywords(self):
        """Test that empty keywords produce valid prompt."""
        result = build_transcription_prompt("")

        # Should still contain instructions
        assert "TRANSCRIPTION INSTRUCTIONS" in result
        assert "(No specific terminology provided)" in result
```

and replace `test_build_prompt_contains_timestamp_instructions` with:

```python
    def test_build_prompt_contains_timestamp_instructions(self):
        """Test that prompt includes timestamp formatting instructions."""
        result = build_transcription_prompt("test")

        assert "[MM:SS]" in result
        assert "[00:00]" in result  # Example format
```

- [ ] **Step 2: Run the prompt tests**

Run: `/Users/danielcharette/Git/Transcriber/venv/bin/python -m pytest tests/test_prompts.py -v`
Expected: all PASS

- [ ] **Step 3: Run the full suite — must now be fully green**

Run: `/Users/danielcharette/Git/Transcriber/venv/bin/python -m pytest tests/ -q`
Expected: 0 failures. From this task on, the whole suite must stay green.

- [ ] **Step 4: Commit**

```bash
git add tests/test_prompts.py
git commit -m "Fix stale prompt tests to assert current template content"
```

---

### Task 6: Wire durability into app.py

**Files:**
- Rewrite: `app.py` (full contents below)
- Create: `tests/test_app_smoke.py`

**Interfaces:**
- Consumes: `setup_logging` (Task 1); `save_recording`, `save_transcript`, `list_recordings`, `list_transcripts`, `new_session_stem`, `recording_fingerprint` (Task 2); `convert_file_to_mp3`, `AudioConversionError` (Task 3); hardened `upload_audio` / `wait_for_active` / `transcribe` / `delete_file` (Task 4 — same signatures as before).

Behavior changes bundled here (all reliability-motivated):
1. Recordings are saved to `recordings/` the moment they arrive, once per unique recording (fingerprint-guarded), and the UI shows the on-disk path.
2. MP3 conversion happens ONCE per recording, file-to-file via `convert_file_to_mp3` (no per-rerun pydub decode — this was the OOM crash source). Failed conversion is remembered (`None` marker) so it doesn't retry on every rerun.
3. Transcription of a recording reuses the already-saved file (no second temp copy); only uploads create (and later delete) a temp file.
4. The streaming transcript is checkpointed to `transcripts/<stem>.partial.txt` every ~2000 new characters, the final transcript is saved to `<stem>.txt`, and ANY failure path preserves whatever text already streamed (with the path shown to the user).
5. The transcript is displayed outside the collapsed status container after completion (preserves an improvement the user made locally).
6. PDF bytes are generated once per transcript and cached in session state (previously rebuilt on every rerun).
7. Every stage and every error is logged; a "Saved sessions" expander lists on-disk recordings/transcripts so users can always find their data after a crash/restart.
8. The `MemoryError` handler is gone (its cause is fixed); generic handler covers the rest.

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_app_smoke.py`:

```python
"""Smoke test: the full Streamlit app script executes without raising.

Uses Streamlit's AppTest harness. Custom components return None under
AppTest, so this exercises the script path without a recording present —
which is exactly the path every rerun takes.
"""

from streamlit.testing.v1 import AppTest


def test_app_runs_without_exception(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    at = AppTest.from_file("app.py", default_timeout=30).run()
    assert not at.exception


def test_app_shows_title(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    at = AppTest.from_file("app.py", default_timeout=30).run()
    assert at.title[0].value == "PharmaTranscribe AI"
```

- [ ] **Step 2: Run smoke tests against the current app.py — they should already pass**

Run: `/Users/danielcharette/Git/Transcriber/venv/bin/python -m pytest tests/test_app_smoke.py -v`
Expected: PASS (this establishes the harness works BEFORE the rewrite; the rewrite must keep it passing). If these fail here, STOP and report — the harness assumption is wrong.

- [ ] **Step 3: Rewrite app.py**

Replace the entire contents of `app.py` with:

```python
"""PharmaTranscribe AI - Streamlit application entry point."""

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from google.api_core import exceptions as google_exceptions
from google.genai import errors as genai_errors

from src.audio_recorder import AudioConversionError, convert_file_to_mp3
from src.components.audio_recorder import audio_recorder
from src.gemini_client import (
    FileProcessingError,
    delete_file,
    get_client,
    transcribe,
    upload_audio,
    wait_for_active,
)
from src.logging_config import setup_logging
from src.prompts import build_transcription_prompt
from src.storage import (
    list_recordings,
    list_transcripts,
    new_session_stem,
    recording_fingerprint,
    save_recording,
    save_transcript,
)
from src.utils import get_mime_type

logger = setup_logging()

# Load environment variables
load_dotenv()

# How many characters of new transcript text trigger a partial checkpoint
TRANSCRIPT_CHECKPOINT_CHARS = 2000


def build_transcript_pdf(transcript_text: str) -> bytes:
    """Render the transcript into PDF bytes."""
    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "PharmaTranscribe AI - Transcript", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(10)

    pdf.set_font("Helvetica", "", 10)
    safe_text = transcript_text.encode("latin-1", errors="replace").decode("latin-1")
    pdf.multi_cell(0, 5, safe_text)

    return bytes(pdf.output())


# Page configuration
st.set_page_config(page_title="PharmaTranscribe AI", page_icon="💊", layout="wide")

st.title("PharmaTranscribe AI")
st.markdown("Transcribe pharmaceutical earnings calls with domain-aware accuracy.")

# --- Sidebar: API Key Handling ---
with st.sidebar:
    st.header("Configuration")

    # Check for API key in environment first
    env_api_key = os.getenv("GOOGLE_API_KEY")

    if env_api_key:
        st.success("API key loaded from environment")
        api_key = env_api_key
    else:
        api_key = st.text_input(
            "Google API Key",
            type="password",
            placeholder="Enter your Gemini API key",
            help="Get your API key from https://makersuite.google.com/app/apikey",
        )
        if not api_key:
            st.warning("Please enter your API key to continue")

# --- Main Content ---
if not api_key:
    st.info("Please configure your Google API key in the sidebar to get started.")
    st.stop()

# Store API key in session state
st.session_state["api_key"] = api_key

# --- Input Section ---
st.header("Audio Input")

# Create tabs for input methods
tab_upload, tab_record = st.tabs(["Upload File", "Record Audio"])

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload Earnings Call Audio",
        type=["mp3", "wav", "m4a", "mpeg", "webm"],
        help="Supported formats: MP3, WAV, M4A, MPEG, WebM (up to 200MB)",
    )

with tab_record:
    st.markdown("Record audio directly from your microphone.")
    st.caption("Speaker audio capture enabled (echo cancellation disabled)")

    audio_recording = audio_recorder(key="audio_recorder")

    if audio_recording:
        audio_bytes = audio_recording.getvalue()
        fingerprint = recording_fingerprint(audio_bytes)

        # Persist each new recording to disk immediately so no crash can lose it
        if st.session_state.get("recording_fingerprint") != fingerprint:
            stem = new_session_stem()
            try:
                saved_path = save_recording(audio_bytes, audio_recording.format, stem=stem)
            except OSError as e:
                logger.exception("Failed to save recording to disk")
                st.error(f"Could not save recording to disk: {e}")
                saved_path = None
            if saved_path:
                st.session_state["recording_fingerprint"] = fingerprint
                st.session_state["recording_path"] = str(saved_path)
                st.session_state["session_stem"] = stem
                st.session_state.pop("mp3_path", None)

        recording_path = st.session_state.get("recording_path")
        if recording_path:
            size_mb = len(audio_bytes) / (1024 * 1024)
            st.success(f"Recording saved to `{recording_path}` ({size_mb:.1f} MB)")

            # Convert to MP3 once per recording; remember failure so we don't
            # retry (and re-spike memory/CPU) on every rerun.
            if "mp3_path" not in st.session_state:
                try:
                    with st.spinner("Preparing MP3..."):
                        mp3_path = convert_file_to_mp3(Path(recording_path))
                    st.session_state["mp3_path"] = str(mp3_path)
                except AudioConversionError as e:
                    st.session_state["mp3_path"] = None
                    logger.error("MP3 conversion failed: %s", e)

            mp3_path = st.session_state.get("mp3_path")
            if mp3_path and Path(mp3_path).exists():
                st.download_button(
                    label="Download Recording as MP3",
                    data=Path(mp3_path).read_bytes(),
                    file_name=Path(mp3_path).name,
                    mime="audio/mpeg",
                )
            elif mp3_path is None and "mp3_path" in st.session_state:
                st.warning(
                    "Could not prepare MP3 download; the original recording "
                    f"is safe at `{recording_path}`."
                )

# Determine which audio source to use
audio_source = None
audio_source_type = None

if uploaded_file:
    audio_source = uploaded_file
    audio_source_type = "upload"
elif audio_recording:
    audio_source = audio_recording
    audio_source_type = "recording"

st.header("Context Keywords")

keywords = st.text_area(
    "Enter domain-specific terminology",
    placeholder="Enter drug names, tickers, separated by commas (e.g., Keytruda, VRTX, pembrolizumab)",
    help="These terms will be used to improve transcription accuracy for pharmaceutical terminology",
)

# --- Transcription ---
if st.button("Transcribe", type="primary", disabled=audio_source is None):
    if audio_source is None:
        st.error("Please upload an audio file or record audio first.")
        st.stop()

    # Initialize variables for cleanup
    client = None
    gemini_file = None
    temp_file_path = None
    owns_temp_file = False
    completed = False
    chunks = []

    # Recordings reuse their save-time stem so the transcript pairs with the
    # audio file; uploads get a fresh stem per transcription run.
    if audio_source_type == "recording":
        stem = st.session_state.get("session_stem") or new_session_stem()
        st.session_state["session_stem"] = stem
    else:
        stem = new_session_stem()

    try:
        with st.status("Processing...", expanded=True) as status:
            # Prepare audio file based on source type
            if audio_source_type == "upload":
                status.update(label="Preparing uploaded file...")
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
                ) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    temp_file_path = tmp.name
                owns_temp_file = True
                logger.info(
                    "Upload received: %s (%d bytes)", uploaded_file.name, uploaded_file.size
                )
            elif audio_source_type == "recording":
                status.update(label="Preparing recording...")
                # The recording is already durably on disk -- reuse it directly
                temp_file_path = st.session_state["recording_path"]

            # Get MIME type and initialize client
            mime_type = get_mime_type(temp_file_path)
            client = get_client(api_key)

            # Upload to Gemini
            status.update(label="Uploading audio to Gemini...")
            gemini_file = upload_audio(client, temp_file_path, mime_type)

            # Wait for processing
            status.update(label="Processing audio (this may take a few minutes)...")
            gemini_file = wait_for_active(client, gemini_file)

            # Build prompt and transcribe
            status.update(label="Generating transcript...")
            prompt = build_transcription_prompt(keywords)

            # Stream transcription, checkpointing partial text to disk
            transcript_container = st.empty()
            chars_at_last_save = 0

            for chunk in transcribe(client, gemini_file, prompt):
                chunks.append(chunk)
                text_so_far = "".join(chunks)
                transcript_container.markdown(text_so_far)
                if len(text_so_far) - chars_at_last_save >= TRANSCRIPT_CHECKPOINT_CHARS:
                    save_transcript(text_so_far, stem, partial=True)
                    chars_at_last_save = len(text_so_far)

            full_text = "".join(chunks)
            transcript_path = save_transcript(full_text, stem)
            completed = True
            logger.info("Transcription complete: %d chars", len(full_text))

            status.update(label="Complete!", state="complete")

        # Store transcript in session state for download
        st.session_state["transcript"] = full_text
        st.session_state.pop("transcript_pdf", None)

        st.success(f"Transcription complete! Saved to `{transcript_path}`")

        # Display transcript outside the status container so it's visible after status collapses
        if full_text:
            st.subheader("Transcript")
            st.markdown(full_text)

    except (google_exceptions.InvalidArgument, genai_errors.ClientError):
        logger.exception("Transcription failed: invalid key or request")
        st.error("Invalid API key or request. Please check your Google API key and try again.")
    except (google_exceptions.ResourceExhausted, genai_errors.APIError) as e:
        logger.exception("Transcription failed: API error")
        if "rate" in str(e).lower() or "quota" in str(e).lower():
            st.error("Rate limit exceeded. Please wait a few minutes and try again.")
        else:
            st.error(f"API error: {e}")
    except TimeoutError:
        logger.exception("Transcription failed: processing timed out")
        st.error("Audio processing timed out. Try a smaller file or try again later.")
    except FileProcessingError as e:
        logger.exception("Transcription failed: file processing error")
        st.error(f"Could not process audio file. {e}")
    except Exception as e:
        logger.exception("Transcription failed: unexpected error")
        st.error(f"An unexpected error occurred: {e}")

    finally:
        # Preserve any partial transcript before anything else
        if chunks and not completed:
            try:
                partial_path = save_transcript("".join(chunks), stem, partial=True)
                logger.warning("Preserved partial transcript at %s", partial_path)
                st.warning(f"A partial transcript was saved to `{partial_path}`.")
            except OSError:
                logger.exception("Failed to preserve partial transcript")

        # Cleanup: Delete Gemini file
        if gemini_file and client:
            delete_file(client, gemini_file.name)

        # Cleanup: Delete local temp file (uploads only -- never the saved recording)
        if owns_temp_file and temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

# --- Download Buttons ---
if "transcript" in st.session_state and st.session_state["transcript"]:
    st.divider()
    st.subheader("Download Transcript")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="Download as TXT",
            data=st.session_state["transcript"],
            file_name="transcript.txt",
            mime="text/plain",
        )

    with col2:
        # Generate the PDF once per transcript, not on every rerun
        if "transcript_pdf" not in st.session_state:
            st.session_state["transcript_pdf"] = build_transcript_pdf(
                st.session_state["transcript"]
            )

        st.download_button(
            label="Download as PDF",
            data=st.session_state["transcript_pdf"],
            file_name="transcript.pdf",
            mime="application/pdf",
        )

# --- Saved Sessions ---
st.divider()
with st.expander("Saved sessions on disk"):
    st.caption(
        "Recordings and transcripts are saved automatically and survive "
        "crashes and restarts."
    )
    recordings = list_recordings()[:10]
    transcripts = list_transcripts()[:10]

    if not recordings and not transcripts:
        st.write("No saved sessions yet.")

    if recordings:
        st.markdown("**Recordings**")
        for rec in recordings:
            rec_mb = rec.stat().st_size / (1024 * 1024)
            st.markdown(f"- `{rec}` ({rec_mb:.1f} MB)")

    if transcripts:
        st.markdown("**Transcripts**")
        for i, txt in enumerate(transcripts):
            entry_col, btn_col = st.columns([4, 1])
            entry_col.markdown(f"- `{txt.name}`")
            btn_col.download_button(
                "Download",
                data=txt.read_text(encoding="utf-8"),
                file_name=txt.name,
                key=f"session_dl_{i}",
            )
```

- [ ] **Step 4: Run the smoke tests against the rewritten app**

Run: `/Users/danielcharette/Git/Transcriber/venv/bin/python -m pytest tests/test_app_smoke.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full suite**

Run: `/Users/danielcharette/Git/Transcriber/venv/bin/python -m pytest tests/ -q`
Expected: 0 failures.

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_app_smoke.py
git commit -m "Wire durable storage, checkpointing, and logging into app"
```

---

### Task 7: Frontend recorder hardening (memory + crash recovery)

**Files:**
- Modify: `src/components/audio_recorder/frontend/index.html` (script + one new HTML block + two style rules)
- Modify: `tests/test_audio_recording_component.py` (append a guard-test class)

**Interfaces:**
- The component's value contract with Python is UNCHANGED: it still sends `JSON.stringify({data: <base64>, format: 'webm'|'mp4', mimeType: <string>})` via `setComponentValue`. `src/components/audio_recorder/__init__.py` is NOT modified.

Changes (all inside `index.html`):
1. Replace the manual chunked `String.fromCharCode` base64 loop with `FileReader.readAsDataURL` (async, no giant intermediate strings — the old loop held ~4 copies of an hour-long recording on the main thread).
2. Change `mediaRecorder.start(100)` to `mediaRecorder.start(1000)` (36,000 chunk blobs/hour → 3,600).
3. Checkpoint every recorded chunk into IndexedDB as it arrives, so a tab crash mid-recording loses at most ~1 second of audio.
4. On component load, if checkpointed chunks exist, show a recovery banner with "Recover" (reassembles the blob and sends it to Python through the normal path) and "Discard" buttons.
5. Checkpoints are cleared when a NEW recording starts or on Discard — deliberately NOT right after sending, so a backend crash during send still leaves the browser-side copy.
6. Handle `mediaRecorder.onerror` by stopping and salvaging already-captured chunks.

- [ ] **Step 1: Write the failing guard tests**

Append to `tests/test_audio_recording_component.py`:

```python
from pathlib import Path

COMPONENT_HTML_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "components" / "audio_recorder" / "frontend" / "index.html"
)


class TestFrontendHardening:
    """Guard tests pinning the reliability-critical parts of the frontend."""

    @pytest.fixture(scope="class")
    def html(self):
        return COMPONENT_HTML_PATH.read_text(encoding="utf-8")

    def test_uses_filereader_for_base64(self, html):
        assert "readAsDataURL" in html

    def test_no_manual_base64_loop(self, html):
        assert "String.fromCharCode.apply" not in html

    def test_uses_indexeddb_checkpointing(self, html):
        assert "indexedDB.open" in html

    def test_one_second_timeslice(self, html):
        assert "mediaRecorder.start(1000)" in html
        assert "mediaRecorder.start(100)" not in html.replace("mediaRecorder.start(1000)", "")

    def test_recovery_banner_present(self, html):
        assert 'id="recoveryBanner"' in html
        assert 'id="recoverBtn"' in html
        assert 'id="discardBtn"' in html

    def test_recorder_error_handler_present(self, html):
        assert "mediaRecorder.onerror" in html
```

(`pytest` is already imported at the top of this file.)

- [ ] **Step 2: Run guard tests to verify they fail**

Run: `/Users/danielcharette/Git/Transcriber/venv/bin/python -m pytest tests/test_audio_recording_component.py -v -k FrontendHardening`
Expected: all 6 FAIL against the current HTML (it still has the manual base64 loop and none of the new markers).

- [ ] **Step 3: Modify index.html**

3a. Add these style rules inside the existing `<style>` block (after the `.duration-warning.visible` rule):

```css
        .recovery-banner {
            background: #e7f3ff;
            border: 1px solid #4b9fff;
            border-radius: 4px;
            padding: 8px 12px;
            font-size: 13px;
            color: #1a5fa8;
            display: none;
        }

        .recovery-banner button {
            margin-left: 8px;
            padding: 4px 10px;
            border-radius: 4px;
            border: 1px solid #4b9fff;
            background: white;
            cursor: pointer;
            font-size: 13px;
        }
```

3b. Add this HTML block right after the `durationWarning` div:

```html
        <div id="recoveryBanner" class="recovery-banner">
            <span id="recoveryText"></span>
            <button id="recoverBtn">Recover</button>
            <button id="discardBtn">Discard</button>
        </div>
```

3c. In the `<script>` section, add DOM lookups next to the existing ones:

```javascript
        const recoveryBanner = document.getElementById('recoveryBanner');
        const recoveryText = document.getElementById('recoveryText');
        const recoverBtn = document.getElementById('recoverBtn');
        const discardBtn = document.getElementById('discardBtn');
```

3d. Add IndexedDB helpers and the base64 helper (place after the `formatTime`/`updateTimer` functions):

```javascript
        // --- IndexedDB checkpointing --------------------------------------
        // Every recorded chunk is checkpointed so a tab crash mid-recording
        // loses at most ~1 second of audio. Checkpoints are cleared when a
        // new recording starts or the user discards them -- deliberately not
        // right after sending, so a backend crash during send still leaves
        // the browser-side copy intact.
        const DB_NAME = 'transcriber-recorder';
        const CHUNK_STORE = 'chunks';
        let chunkSeq = 0;

        function openDb() {
            return new Promise((resolve, reject) => {
                const req = indexedDB.open(DB_NAME, 1);
                req.onupgradeneeded = () => {
                    const db = req.result;
                    if (!db.objectStoreNames.contains(CHUNK_STORE)) {
                        db.createObjectStore(CHUNK_STORE, { keyPath: 'seq' });
                    }
                };
                req.onsuccess = () => resolve(req.result);
                req.onerror = () => reject(req.error);
            });
        }

        async function idbPutChunk(seq, blob, mimeType) {
            const db = await openDb();
            return new Promise((resolve, reject) => {
                const tx = db.transaction(CHUNK_STORE, 'readwrite');
                tx.objectStore(CHUNK_STORE).put({ seq: seq, blob: blob, mimeType: mimeType });
                tx.oncomplete = () => { db.close(); resolve(); };
                tx.onerror = () => { db.close(); reject(tx.error); };
            });
        }

        async function idbLoadChunks() {
            const db = await openDb();
            return new Promise((resolve, reject) => {
                const tx = db.transaction(CHUNK_STORE, 'readonly');
                const req = tx.objectStore(CHUNK_STORE).getAll();
                req.onsuccess = () => {
                    db.close();
                    resolve(req.result.sort((a, b) => a.seq - b.seq));
                };
                req.onerror = () => { db.close(); reject(req.error); };
            });
        }

        async function idbClearChunks() {
            const db = await openDb();
            return new Promise((resolve, reject) => {
                const tx = db.transaction(CHUNK_STORE, 'readwrite');
                tx.objectStore(CHUNK_STORE).clear();
                tx.oncomplete = () => { db.close(); resolve(); };
                tx.onerror = () => { db.close(); reject(tx.error); };
            });
        }

        // Efficient base64 without giant intermediate strings
        function blobToBase64(blob) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onerror = () => reject(reader.error);
                reader.onload = () => {
                    const dataUrl = reader.result;
                    resolve(dataUrl.substring(dataUrl.indexOf(',') + 1));
                };
                reader.readAsDataURL(blob);
            });
        }
```

3e. Replace the ENTIRE existing `processRecording` function with:

```javascript
        // Process recording and send compressed audio directly to Streamlit
        async function processRecording(blob, mimeTypeOverride) {
            try {
                status.textContent = 'Processing...';

                const mimeType = mimeTypeOverride
                    || (mediaRecorder && mediaRecorder.mimeType)
                    || blob.type
                    || 'audio/webm';
                let format = 'webm';
                if (mimeType.includes('mp4') || mimeType.includes('m4a')) {
                    format = 'mp4';
                }

                // Read blob directly (NO DECODING - keep compressed!)
                const base64 = await blobToBase64(blob);

                // Send JSON with format metadata
                setComponentValue(JSON.stringify({
                    data: base64,
                    format: format,
                    mimeType: mimeType
                }));

                // Update UI
                const sizeKB = (blob.size / 1024).toFixed(1);
                status.className = 'status success';
                status.textContent = `Recording captured (${sizeKB} KB)`;

                // Preview using original blob
                audioPreview.src = URL.createObjectURL(blob);
                audioContainer.style.display = 'block';

            } catch (err) {
                console.error('Error processing recording:', err);
                errorDiv.textContent = 'Error processing recording: ' + err.message
                    + ' (your audio is still checkpointed locally -- reload to recover)';
                errorDiv.style.display = 'block';
            }
        }
```

3f. In `startRecording`, immediately after `durationWarning.classList.remove('visible');` add:

```javascript
                recoveryBanner.style.display = 'none';
                chunkSeq = 0;
                try {
                    await idbClearChunks();
                } catch (e) {
                    console.error('Could not clear old checkpoints:', e);
                }
```

3g. Replace the `mediaRecorder.ondataavailable` assignment with:

```javascript
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                        chunkSeq += 1;
                        idbPutChunk(chunkSeq, event.data, mediaRecorder.mimeType)
                            .catch((err) => console.error('Checkpoint write failed:', err));
                    }
                };
```

3h. After the `mediaRecorder.onstop` assignment, add an error handler:

```javascript
                mediaRecorder.onerror = (event) => {
                    console.error('MediaRecorder error:', event.error || event);
                    errorDiv.textContent = 'Recording error -- salvaging captured audio.';
                    errorDiv.style.display = 'block';
                    stopRecording();
                };
```

3i. Replace `mediaRecorder.start(100); // Collect data every 100ms` with:

```javascript
                mediaRecorder.start(1000); // Collect data every 1s (checkpointed to IndexedDB)
```

3j. Add the recovery check and call it from `init()`:

```javascript
        // Offer recovery of chunks left behind by a crashed/closed tab
        async function checkRecovery() {
            try {
                const records = await idbLoadChunks();
                if (records.length === 0) {
                    return;
                }
                const totalBytes = records.reduce((sum, r) => sum + r.blob.size, 0);
                const mimeType = records[0].mimeType || 'audio/webm';
                const totalMB = (totalBytes / (1024 * 1024)).toFixed(1);
                recoveryText.textContent =
                    `Found an unsent recording checkpoint (${totalMB} MB).`;
                recoveryBanner.style.display = 'block';
                setFrameHeight(220);

                recoverBtn.onclick = async () => {
                    recoveryBanner.style.display = 'none';
                    const blob = new Blob(records.map((r) => r.blob), { type: mimeType });
                    await processRecording(blob, mimeType);
                };
                discardBtn.onclick = async () => {
                    try {
                        await idbClearChunks();
                    } catch (e) {
                        console.error('Could not discard checkpoints:', e);
                    }
                    recoveryBanner.style.display = 'none';
                };
            } catch (err) {
                console.error('Recovery check failed:', err);
            }
        }

        // Initialize component
        function init() {
            setFrameHeight(150);
            setComponentReady();
            checkRecovery();
        }
```

(Replace the existing `init` definition with the version above.)

- [ ] **Step 4: Run guard tests to verify they pass**

Run: `/Users/danielcharette/Git/Transcriber/venv/bin/python -m pytest tests/test_audio_recording_component.py -v`
Expected: all PASS (existing + 6 new)

- [ ] **Step 5: Sanity-check the HTML loads**

Run: `/Users/danielcharette/Git/Transcriber/venv/bin/python -c "from pathlib import Path; html = Path('src/components/audio_recorder/frontend/index.html').read_text(); assert html.count('function processRecording') == 1; assert html.count('function init') == 1; print('OK')"`
Expected: `OK`

- [ ] **Step 6: Run the full suite**

Run: `/Users/danielcharette/Git/Transcriber/venv/bin/python -m pytest tests/ -q`
Expected: 0 failures.

- [ ] **Step 7: Commit**

```bash
git add src/components/audio_recorder/frontend/index.html tests/test_audio_recording_component.py
git commit -m "Harden recorder frontend: IndexedDB checkpoints, crash recovery, efficient base64"
```

---

### Task 8: Document the reliability features

**Files:**
- Modify: `README.md`

**Interfaces:** none.

- [ ] **Step 1: Update the Features list**

In `README.md`, replace the current Features bullet list with:

```markdown
- Upload audio files up to 200MB (MP3, WAV, M4A, MPEG)
- Record audio directly from your microphone with **speaker audio capture** (captures audio playing through your speakers, perfect for transcribing earnings calls from another browser tab)
- **Crash-proof by design**: recordings are checkpointed in the browser every second and saved to disk the moment they arrive; transcripts are checkpointed to disk while they stream
- Inject domain-specific keywords for improved accuracy
- Real-time streaming transcription display
- Download transcripts as text or PDF files
```

- [ ] **Step 2: Add a Reliability section**

Insert this section into `README.md` between the `## Usage` and `## Testing` sections:

```markdown
## Reliability & Recovery

Long recordings (60+ minutes) are protected at every stage:

- **While recording**: every ~1 second of audio is checkpointed to the browser's IndexedDB. If the tab crashes or is closed mid-recording, reopening the app shows a "Recover" banner that restores the checkpoint.
- **On stop**: the compressed recording is sent to the server and immediately saved to `recordings/session_<timestamp>.<ext>`. The MP3 download is produced once, on disk, via ffmpeg.
- **While transcribing**: the streaming transcript is checkpointed to `transcripts/session_<timestamp>.partial.txt` as it grows; the final transcript is saved to `transcripts/session_<timestamp>.txt`. If transcription fails midway, the partial file is preserved and its path is shown in the UI.
- **After a crash or restart**: the "Saved sessions on disk" panel at the bottom of the app lists all recordings and transcripts found on disk.
- **Logs**: application logs rotate in `logs/transcriber.log` — check them when reporting issues.
```

- [ ] **Step 3: Run the full suite (unchanged, but confirms nothing broke)**

Run: `/Users/danielcharette/Git/Transcriber/venv/bin/python -m pytest tests/ -q`
Expected: 0 failures.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Document reliability and crash-recovery features"
```
