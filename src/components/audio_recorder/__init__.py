"""Custom audio recorder component with echo cancellation disabled.

This component allows recording audio from the microphone while capturing
speaker audio (system audio playing through speakers), which is filtered out
by default browser echo cancellation.
"""

import base64
import io
from pathlib import Path

import streamlit.components.v1 as components

# Get the path to the frontend directory
_COMPONENT_PATH = Path(__file__).parent / "frontend"

# Declare the component
_component_func = components.declare_component(
    "audio_recorder",
    path=str(_COMPONENT_PATH),
)


class AudioRecording:
    """Wrapper class to mimic Streamlit's UploadedFile interface."""

    def __init__(self, audio_bytes: bytes):
        self._bytes = audio_bytes
        self._buffer = io.BytesIO(audio_bytes)

    def getvalue(self) -> bytes:
        """Return the audio data as bytes."""
        return self._bytes

    def read(self) -> bytes:
        """Read the audio data."""
        return self._buffer.read()

    def seek(self, pos: int) -> int:
        """Seek to a position in the buffer."""
        return self._buffer.seek(pos)

    def __len__(self) -> int:
        """Return the length of the audio data."""
        return len(self._bytes)


def audio_recorder(key: str = None) -> AudioRecording | None:
    """Display an audio recorder that captures speaker audio.

    This recorder disables echo cancellation, auto gain control, and noise
    suppression to allow capturing audio playing through the computer's speakers.

    Args:
        key: Optional key for the component.

    Returns:
        AudioRecording object with WAV audio data, or None if no recording.
    """
    # Call the component
    component_value = _component_func(key=key, default=None)

    # If we have a value, decode it from base64
    if component_value is not None and isinstance(component_value, str):
        try:
            audio_bytes = base64.b64decode(component_value)
            return AudioRecording(audio_bytes)
        except Exception:
            return None

    return None
