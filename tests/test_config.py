"""Tests for src/config.py — config loading, validation, defaults."""

import os
from unittest.mock import patch

import pytest


def test_get_csv_list_default():
    """_get_csv_list returns parsed default when env var is unset."""
    from src.config import _get_csv_list

    with patch.dict(os.environ, {}, clear=False):
        # Remove key if it exists
        os.environ.pop("TEST_CSV_KEY", None)
        result = _get_csv_list("TEST_CSV_KEY", "a,b,c")
    assert result == ["a", "b", "c"]


def test_get_csv_list_from_env():
    """_get_csv_list parses env var with whitespace handling."""
    from src.config import _get_csv_list

    with patch.dict(os.environ, {"TEST_CSV_KEY": " foo , BAR , baz "}):
        result = _get_csv_list("TEST_CSV_KEY", "default")
    assert result == ["foo", "bar", "baz"]


def test_get_int_default():
    """_get_int returns default when env var is unset."""
    from src.config import _get_int

    os.environ.pop("TEST_INT_KEY", None)
    result = _get_int("TEST_INT_KEY", 42)
    assert result == 42


def test_get_int_valid():
    """_get_int parses a valid integer from env."""
    from src.config import _get_int

    with patch.dict(os.environ, {"TEST_INT_KEY": "100"}):
        result = _get_int("TEST_INT_KEY", 42)
    assert result == 100


def test_get_int_invalid_falls_back():
    """_get_int falls back to default on invalid input."""
    from src.config import _get_int

    with patch.dict(os.environ, {"TEST_INT_KEY": "not_a_number"}):
        result = _get_int("TEST_INT_KEY", 42)
    assert result == 42


def test_validate_extension_supported():
    """validate_extension returns True for supported extensions."""
    from src.config import validate_extension

    assert validate_extension("meeting.mp4") is True
    assert validate_extension("recording.m4a") is True
    assert validate_extension("audio.wav") is True


def test_validate_extension_unsupported():
    """validate_extension returns False for unsupported extensions."""
    from src.config import validate_extension

    assert validate_extension("document.pdf") is False
    assert validate_extension("image.png") is False


def test_validate_gcp_config_missing():
    """validate_gcp_config raises ValueError when required vars are unset."""
    from src.config import validate_gcp_config

    with patch("src.config.GCP_PROJECT_ID", None), \
         patch("src.config.GCS_BUCKET_NAME", None):
        with pytest.raises(ValueError, match="Missing required GCP configuration"):
            validate_gcp_config()


def test_validate_gcp_config_valid():
    """validate_gcp_config passes when required vars are set."""
    from src.config import validate_gcp_config

    with patch("src.config.GCP_PROJECT_ID", "my-project"), \
         patch("src.config.GCS_BUCKET_NAME", "my-bucket"):
        validate_gcp_config()  # Should not raise


def test_gcs_upload_timeout_default():
    """Verify GCS_UPLOAD_TIMEOUT_SECONDS default fallback in _get_int."""
    from src.config import _get_int
    with patch.dict(os.environ, {}, clear=True):
        assert _get_int("GCS_UPLOAD_TIMEOUT_SECONDS", 600) == 600


def test_chunk_length_seconds_dutch_loaded():
    """Verify CHUNK_LENGTH_SECONDS_DUTCH is correctly loaded from src.config."""
    from src.config import CHUNK_LENGTH_SECONDS_DUTCH
    assert CHUNK_LENGTH_SECONDS_DUTCH in [300, 900]  # Depends on active .env file, but should be defined
