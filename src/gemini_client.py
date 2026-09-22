"""Gemini API wrapper for PharmaTranscribe AI using google-genai SDK."""

import os
import time
from typing import Callable, Iterator, Optional

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from src.logging_config import get_logger

logger = get_logger("gemini")


class FileProcessingError(Exception):
    """Raised when file processing fails on Gemini side."""
    pass


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


def get_client(api_key: str) -> genai.Client:
    """Initialize and return Gemini client.

    Args:
        api_key: Google API key for Gemini

    Returns:
        Configured Client instance
    """
    return genai.Client(api_key=api_key)


def upload_audio(client: genai.Client, file_path: str, mime_type: str) -> types.File:
    """Upload file to Gemini File API with retries on transient failures.

    Args:
        client: Gemini client instance
        file_path: Path to the audio file
        mime_type: MIME type of the file

    Returns:
        File object with .name and .state attributes
    """
    size = os.path.getsize(file_path)
    logger.info("Uploading %s (%d bytes, %s)", file_path, size, mime_type)
    result = call_with_retries(
        lambda: client.files.upload(file=file_path, config={"mime_type": mime_type}),
        description="Gemini file upload",
    )
    logger.info("Upload complete: %s", result.name)
    return result


def wait_for_active(
    client: genai.Client,
    file: types.File,
    timeout_seconds: int = 900,
    poll_interval: int = 3,
) -> types.File:
    """Wait for file to be ready for use.

    The new SDK handles file readiness automatically in most cases,
    but we keep this for explicit status checking if needed.

    Args:
        client: Gemini client instance
        file: File object returned from upload_audio
        timeout_seconds: Maximum time to wait (default: 900)
        poll_interval: Seconds between status checks (default: 3)

    Returns:
        File object ready for use

    Raises:
        TimeoutError: If not ready within timeout
        FileProcessingError: If processing failed
    """
    start_time = time.time()

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


TRANSCRIBE_MODEL = "gemini-3.5-transcribe"
FALLBACK_MODEL = "gemini-2.5-flash"
# count_tokens bills audio at ~32 tokens/s; divide to estimate duration.
COUNT_TOKENS_PER_SECOND = 32
# The input cap (98,304 tokens) allows ~65 min, but the binding limit is the
# 32,768-token output cap: word timestamps cost several tokens per word, so a
# dense 48-min earnings call (~7,000 words) truncates while 23 min fits.
TRANSCRIBE_MAX_SECONDS = 30 * 60


def _estimate_audio_seconds(client: genai.Client, file: types.File) -> float:
    count = client.models.count_tokens(model=TRANSCRIBE_MODEL, contents=[file])
    return (count.total_tokens or 0) / COUNT_TOKENS_PER_SECOND


def _parse_offset(offset: Optional[str]) -> float:
    """Parse a duration string like "24.800s" into seconds."""
    try:
        return float((offset or "0").rstrip("s"))
    except ValueError:
        return 0.0


def _format_timestamp(seconds: float) -> str:
    total = int(seconds)
    return f"[{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}]"


def _speaker_name(label: Optional[str]) -> str:
    """Map diarization labels like "spk:0" to "Speaker 1"."""
    try:
        return f"Speaker {int((label or '').rsplit(':', 1)[-1]) + 1}"
    except ValueError:
        return "Speaker"


def _transcribe_with_asr_model(client: genai.Client, file: types.File) -> list[str]:
    """Transcribe with the dedicated ASR model, returning formatted paragraphs.

    This model ignores text prompts and
    returns `audio_transcription` parts instead of text; speaker labels and
    timestamps come from the transcription config. Consecutive segments from
    the same speaker are merged into one paragraph.
    """
    response = call_with_retries(
        lambda: client.models.generate_content(
            model=TRANSCRIBE_MODEL,
            contents=[file],
            config=types.GenerateContentConfig(
                audio_transcription_config=types.AudioTranscriptionConfig(
                    diarization=True, word_timestamp=True
                ),
            ),
        ),
        attempts=2,  # a fallback model is waiting; don't linger on 503s
        description="Gemini transcribe-model request",
    )

    candidate = response.candidates[0] if response.candidates else None
    parts = candidate.content.parts if candidate and candidate.content else None
    if candidate and candidate.finish_reason not in (None, types.FinishReason.STOP):
        raise RuntimeError(f"transcribe model stopped early: {candidate.finish_reason}")

    paragraphs: list[list] = []  # [start_seconds, speaker, [texts]]
    for part in parts or []:
        segment = part.audio_transcription
        if not segment or not (segment.text or "").strip():
            continue
        speaker = _speaker_name(segment.speaker_label)
        if paragraphs and paragraphs[-1][1] == speaker:
            paragraphs[-1][2].append(segment.text.strip())
            continue
        start = _parse_offset(segment.words[0].start_offset) if segment.words else 0.0
        paragraphs.append([start, speaker, [segment.text.strip()]])

    if not paragraphs:
        raise RuntimeError("transcribe model returned no transcription")
    return [
        f"{_format_timestamp(start)} {speaker}: {' '.join(texts)}\n\n"
        for start, speaker, texts in paragraphs
    ]


def transcribe(
    client: genai.Client,
    file: types.File,
    system_prompt: str,
    on_model_selected: Optional[Callable[[str], None]] = None,
) -> Iterator[str]:
    """Generate transcript, preferring the dedicated ASR model.

    Uses gemini-3.5-transcribe when the audio fits its input cap, and falls
    back to prompt-driven gemini-2.5-flash when the audio is too long or the
    transcribe model fails for any API reason (capacity, quota, rejection).
    The transcribe model returns its result in one piece, so a fallback
    never mixes output from two models.

    Args:
        client: Gemini client instance
        file: Active Gemini File object
        system_prompt: Formatted prompt from prompts.py (used by the fallback
            model only -- the transcribe model ignores prompts)
        on_model_selected: Optional callback told which model is producing
            the transcript, before the first chunk is yielded

    Yields:
        Text chunks as they stream in
    """
    try:
        seconds = _estimate_audio_seconds(client, file)
    except Exception as exc:
        logger.warning("Could not estimate audio length (%s); using %s", exc, FALLBACK_MODEL)
        seconds = None

    if seconds is None:
        pass
    elif seconds > TRANSCRIBE_MAX_SECONDS:
        logger.info(
            "Audio ~%.0f min exceeds %s cap; using %s",
            seconds / 60, TRANSCRIBE_MODEL, FALLBACK_MODEL,
        )
    else:
        logger.info("Starting transcription with model=%s file=%s", TRANSCRIBE_MODEL, file.name)
        try:
            paragraphs = _transcribe_with_asr_model(client, file)
        except Exception as exc:
            logger.warning("%s failed (%s); falling back to %s", TRANSCRIBE_MODEL, exc, FALLBACK_MODEL)
        else:
            if on_model_selected:
                on_model_selected(TRANSCRIBE_MODEL)
            yield from paragraphs
            return

    if on_model_selected:
        on_model_selected(FALLBACK_MODEL)
    logger.info("Starting transcription with model=%s file=%s", FALLBACK_MODEL, file.name)
    response = client.models.generate_content_stream(
        model=FALLBACK_MODEL,
        contents=[system_prompt, file],
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=32768,
        ),
    )

    for chunk in response:
        if chunk.text:
            yield chunk.text


def delete_file(client: genai.Client, file_name: str) -> bool:
    """Delete file from Gemini storage.

    Args:
        client: Gemini client instance
        file_name: Name of the file to delete (e.g., "files/abc123")

    Returns:
        True if deleted, False if file not found or error occurred
    """
    try:
        client.files.delete(name=file_name)
        return True
    except Exception:
        # Silently handle errors to not interrupt user flow
        logger.warning("Failed to delete Gemini file %s", file_name)
        return False
