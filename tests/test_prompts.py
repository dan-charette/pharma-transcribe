"""Unit tests for the transcription prompt."""

from src.prompts import TRANSCRIPTION_PROMPT


class TestTranscriptionPrompt:
    """Tests for the fallback model's transcription prompt."""

    def test_prompt_contains_instructions(self):
        assert "TRANSCRIPTION INSTRUCTIONS" in TRANSCRIPTION_PROMPT

    def test_prompt_contains_timestamp_instructions(self):
        assert "[HH:MM:SS]" in TRANSCRIPTION_PROMPT
        assert "[00:00:00]" in TRANSCRIPTION_PROMPT  # Example format

    def test_prompt_examples_match_speaker_naming_format(self):
        """Every example paragraph must parse with the speaker-naming PARA_RE."""
        from src.speakers import PARA_RE
        examples = [line for line in TRANSCRIPTION_PROMPT.splitlines() if line.startswith("[0")]
        assert len(examples) == 4
        assert all(PARA_RE.match(b) for b in examples)

    def test_prompt_contains_speaker_instructions(self):
        assert "Speaker 1:" in TRANSCRIPTION_PROMPT

    def test_prompt_has_no_placeholders(self):
        """No leftover format fields from the removed keywords feature."""
        assert "{" not in TRANSCRIPTION_PROMPT
        assert "TERMINOLOGY" not in TRANSCRIPTION_PROMPT
