"""Gemini API wrapper for PharmaTranscribe AI using google-genai SDK."""

import os
import time
from typing import Iterator

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


def transcribe(
    client: genai.Client,
    file: types.File,
    system_prompt: str,
) -> Iterator[str]:
    """Generate transcript using Gemini with streaming.

    Args:
        client: Gemini client instance
        file: Active Gemini File object
        system_prompt: Formatted prompt from prompts.py

    Yields:
        Text chunks as they stream in
    """
    logger.info("Starting transcription with model=%s file=%s", "gemini-3.5-flash", file.name)
    response = client.models.generate_content_stream(
        model="gemini-3.5-flash",
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
