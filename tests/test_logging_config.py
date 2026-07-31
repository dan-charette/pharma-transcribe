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
