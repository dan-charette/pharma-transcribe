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
