"""Unit tests for prompt construction."""

import pytest
from src.prompts import build_transcription_prompt, SYSTEM_PROMPT_TEMPLATE


class TestBuildTranscriptionPrompt:
    """Tests for build_transcription_prompt function."""

    def test_build_prompt_with_keywords(self, sample_keywords):
        """Test that keywords are properly injected into the prompt."""
        result = build_transcription_prompt(sample_keywords)

        # Check keywords are present
        assert "Keytruda" in result
        assert "pembrolizumab" in result
        assert "VRTX" in result
        assert "Vertex Pharmaceuticals" in result
        assert "tezacaftor" in result

        # Check they're formatted as bullet points
        assert "- Keytruda" in result
        assert "- pembrolizumab" in result

    def test_build_prompt_empty_keywords(self):
        """Test that empty keywords produce valid prompt."""
        result = build_transcription_prompt("")

        # Should still contain instructions
        assert "TRANSCRIPTION INSTRUCTIONS" in result
        assert "(No specific terminology provided)" in result

    def test_build_prompt_whitespace_only_keywords(self):
        """Test that whitespace-only keywords are handled."""
        result = build_transcription_prompt("   ")

        assert "(No specific terminology provided)" in result
        assert "TRANSCRIPTION INSTRUCTIONS" in result

    def test_build_prompt_single_keyword(self):
        """Test with a single keyword."""
        result = build_transcription_prompt("Keytruda")

        assert "- Keytruda" in result
        assert "CRITICAL TERMINOLOGY LIST" in result

    def test_build_prompt_keywords_with_extra_spaces(self):
        """Test that extra spaces around keywords are trimmed."""
        result = build_transcription_prompt("  Keytruda  ,  pembrolizumab  ,  VRTX  ")

        assert "- Keytruda" in result
        assert "- pembrolizumab" in result
        assert "- VRTX" in result
        # Should not have extra spaces
        assert "-   Keytruda" not in result

    def test_build_prompt_contains_timestamp_instructions(self):
        """Test that prompt includes timestamp formatting instructions."""
        result = build_transcription_prompt("test")

        assert "[MM:SS]" in result
        assert "[00:00]" in result  # Example format

    def test_build_prompt_contains_speaker_instructions(self):
        """Test that prompt includes speaker identification instructions."""
        result = build_transcription_prompt("test")

        assert "Speaker 1:" in result or "speaker turns" in result.lower()

    def test_prompt_template_has_keywords_placeholder(self):
        """Test that the template contains the keywords placeholder."""
        assert "{keywords}" in SYSTEM_PROMPT_TEMPLATE
