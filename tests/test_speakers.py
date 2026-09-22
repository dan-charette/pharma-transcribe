"""Tests for speaker naming with a mocked Gemini client."""

import json
from unittest.mock import Mock

import pytest

from src.speakers import name_speakers, parse_paragraphs

TRANSCRIPT = (
    "[00:00:00] Speaker 1: Good morning, and welcome to the call. I'll turn it over to Jane Smith.\n\n"
    "[00:00:10] Speaker 2: Thanks. I'm Jane Smith, Chief Financial Officer.\n\n"
    "[00:00:30] Speaker 3: Hi, this is Raj Patel. Thanks for taking my question.\n\n"
    "[00:00:40] Speaker 2: Sure, Raj. Revenue grew 12% this quarter.\n\n"
)


def _client(payload):
    client = Mock()
    client.models.generate_content.return_value = Mock(text=json.dumps(payload))
    return client


def _labels(text):
    return [label for _, label, _ in parse_paragraphs(text)]


class TestParseParagraphs:
    def test_parses_timestamp_label_body(self):
        assert parse_paragraphs("[00:01:02] Jane Smith: Hello there.") == [
            ("00:01:02", "Jane Smith", "Hello there.")
        ]

    def test_untimestamped_block_continues_previous_paragraph(self):
        paras = parse_paragraphs("[00:00:00] Speaker 1: First.\n\nStill me.\n\n[00:00:09] Speaker 2: Next.")
        assert paras[0] == ("00:00:00", "Speaker 1", "First.\n\nStill me.")
        assert len(paras) == 2


class TestNameSpeakersByVoice:
    def test_maps_each_voice_label_to_a_name(self):
        client = _client({"voices": [
            {"label": "Speaker 1", "name": "Operator"},
            {"label": "Speaker 2", "name": "Jane Smith"},
            {"label": "Speaker 3", "name": "Raj Patel"},
        ]})

        result = name_speakers(client, TRANSCRIPT, by_voice=True)

        assert _labels(result) == ["Operator", "Jane Smith", "Raj Patel", "Jane Smith"]
        assert [b for _, _, b in parse_paragraphs(result)] == [b for _, _, b in parse_paragraphs(TRANSCRIPT)]

    def test_unsupported_or_empty_names_keep_the_label(self):
        client = _client({"voices": [
            {"label": "Speaker 1", "name": ""},
            {"label": "Speaker 2", "name": "Daniel Vitt"},  # never spoken
            {"label": "Speaker 3", "name": "Raj Patel"},
        ]})

        result = name_speakers(client, TRANSCRIPT, by_voice=True)

        assert _labels(result) == ["Speaker 1", "Speaker 2", "Raj Patel", "Speaker 2"]

    def test_sends_full_paragraph_text(self):
        long_body = "word " * 200 + "and now over to Sam Lee."
        client = _client({"voices": []})

        name_speakers(client, f"[00:00:00] Speaker 1: {long_body}", by_voice=True)

        prompt = client.models.generate_content.call_args.kwargs["contents"][0]
        assert "over to Sam Lee." in prompt


class TestNameSpeakersPerParagraph:
    def test_names_each_paragraph_and_keeps_missing_ones(self):
        client = _client({"speakers": [
            {"index": 0, "speaker": "Operator"},
            {"index": 1, "speaker": "Jane Smith"},
            {"index": 3, "speaker": "Jane Smith"},
        ]})

        result = name_speakers(client, TRANSCRIPT, by_voice=False)

        assert _labels(result) == ["Operator", "Jane Smith", "Speaker 3", "Jane Smith"]


def test_text_without_paragraphs_is_returned_without_api_call():
    client = Mock()

    assert name_speakers(client, "no timestamps here", by_voice=True) == "no timestamps here"
    client.models.generate_content.assert_not_called()


def test_api_errors_propagate():
    client = Mock()
    client.models.generate_content.side_effect = ValueError("bad request")

    with pytest.raises(ValueError):
        name_speakers(client, TRANSCRIPT, by_voice=True)
