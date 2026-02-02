"""Gemini API wrapper for PharmaTranscribe AI using google-genai SDK."""

from typing import Iterator

from google import genai
from google.genai import types


class FileProcessingError(Exception):
    """Raised when file processing fails on Gemini side."""
    pass


def get_client(api_key: str) -> genai.Client:
    """Initialize and return Gemini client.

    Args:
        api_key: Google API key for Gemini

    Returns:
        Configured Client instance
    """
    return genai.Client(api_key=api_key)


def upload_audio(client: genai.Client, file_path: str, mime_type: str) -> types.File:
    """Upload file to Gemini File API.

    Args:
        client: Gemini client instance
        file_path: Path to the audio file
        mime_type: MIME type of the file

    Returns:
        File object with .name and .state attributes
    """
    return client.files.upload(file=file_path, config={"mime_type": mime_type})


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
    import time

    start_time = time.time()

    while True:
        current_file = client.files.get(name=file.name)

        if current_file.state == "ACTIVE":
            return current_file

        if current_file.state == "FAILED":
            raise FileProcessingError(
                f"File processing failed: {file.name}. The file may be corrupted or in an unsupported format."
            )

        elapsed = time.time() - start_time
        if elapsed >= timeout_seconds:
            raise TimeoutError(
                f"File did not become ACTIVE within {timeout_seconds} seconds. "
                f"Current state: {current_file.state}"
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
    response = client.models.generate_content_stream(
        model="gemini-2.5-flash",
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
        return False
