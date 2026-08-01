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
