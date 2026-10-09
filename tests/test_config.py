"""Tests for src/config.py — config loading, validation, defaults, and migration guard."""

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


def test_get_float_valid():
    """_get_float parses a valid float from env."""
    from src.config import _get_float

    with patch.dict(os.environ, {"TEST_FLOAT_KEY": "15.5"}):
        result = _get_float("TEST_FLOAT_KEY", 25.0)
    assert result == 15.5


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

    with patch("src.config.GCP_AUTH_MODE", "adc"), \
         patch("src.config.GCP_PROJECT_ID", "my-project"), \
         patch("src.config.GCS_BUCKET_NAME", "my-bucket"):
        validate_gcp_config()  # Should not raise


def test_validate_gcp_config_invalid_auth_mode():
    """validate_gcp_config raises ValueError on invalid GCP_AUTH_MODE."""
    from src.config import validate_gcp_config

    with patch("src.config.GCP_AUTH_MODE", "invalid_mode"), \
         patch("src.config.GCP_PROJECT_ID", "my-project"), \
         patch("src.config.GCS_BUCKET_NAME", "my-bucket"):
        with pytest.raises(ValueError, match="Invalid GCP_AUTH_MODE: 'invalid_mode'"):
            validate_gcp_config()


def test_validate_gcp_config_key_mode_missing_vars():
    """validate_gcp_config in service_account_key mode raises ValueError if key vars missing."""
    from src.config import validate_gcp_config

    with patch("src.config.GCP_AUTH_MODE", "service_account_key"), \
         patch("src.config.GCP_PROJECT_ID", "my-project"), \
         patch("src.config.GCS_BUCKET_NAME", "my-bucket"), \
         patch("src.config.GCP_SERVICE_ACCOUNT_KEY_FILE", None), \
         patch("src.config.GCP_EXPECTED_SERVICE_ACCOUNT", None):
        with pytest.raises(ValueError, match="GCP_SERVICE_ACCOUNT_KEY_FILE, GCP_EXPECTED_SERVICE_ACCOUNT"):
            validate_gcp_config()


def test_validate_gcp_config_key_mode_valid():
    """validate_gcp_config passes in service_account_key mode when all required vars are set."""
    from src.config import validate_gcp_config

    with patch("src.config.GCP_AUTH_MODE", "service_account_key"), \
         patch("src.config.GCP_PROJECT_ID", "my-project"), \
         patch("src.config.GCS_BUCKET_NAME", "my-bucket"), \
         patch("src.config.GCP_SERVICE_ACCOUNT_KEY_FILE", "/path/to/key.json"), \
         patch("src.config.GCP_EXPECTED_SERVICE_ACCOUNT", "sa@proj.iam.gserviceaccount.com"):
        validate_gcp_config()  # Should not raise



def test_gcs_upload_timeout_default():
    """Verify GCS_UPLOAD_TIMEOUT_SECONDS default fallback in _get_int."""
    from src.config import _get_int
    with patch.dict(os.environ, {}, clear=True):
        assert _get_int("GCS_UPLOAD_TIMEOUT_SECONDS", 600) == 600


def test_chunk_max_length_config_loaded():
    """Verify CHUNK_MAX_LENGTH_SECONDS_DUTCH and CHUNK_MAX_LENGTH_SECONDS_ENGLISH are loaded."""
    from src.config import CHUNK_MAX_LENGTH_SECONDS_DUTCH, CHUNK_MAX_LENGTH_SECONDS_ENGLISH
    assert CHUNK_MAX_LENGTH_SECONDS_DUTCH == 2700
    assert CHUNK_MAX_LENGTH_SECONDS_ENGLISH == 2700


def test_line_formatting_config_defaults():
    """Verify default values for line formatting configurations."""
    from src.config import (
        TRANSCRIPTION_MAX_CHARS_PER_LINE,
        TRANSCRIPTION_MAX_SENTENCES_PER_LINE,
        TRANSCRIPTION_CUSTOM_PROMPT,
    )
    assert isinstance(TRANSCRIPTION_MAX_CHARS_PER_LINE, int)
    assert isinstance(TRANSCRIPTION_MAX_SENTENCES_PER_LINE, int)
    assert TRANSCRIPTION_CUSTOM_PROMPT is None or isinstance(TRANSCRIPTION_CUSTOM_PROMPT, str)


@pytest.mark.parametrize(
    "legacy_key,new_key",
    [
        ("CHUNK_LENGTH_SECONDS_DUTCH", "CHUNK_MAX_LENGTH_SECONDS_DUTCH"),
        ("CHUNK_LENGTH_SECONDS_ENGLISH", "CHUNK_MAX_LENGTH_SECONDS_ENGLISH"),
        ("TRANSCRIPTION_BATCH_SIZE", "TRANSCRIPTION_CONCURRENCY"),
    ],
)
def test_migration_guard_raises_on_legacy_key(legacy_key, new_key):
    """Presence of any legacy key raises ValueError naming both old and new key."""
    from src.config import _check_migration_guard

    with patch.dict(os.environ, {legacy_key: "500"}, clear=False):
        with pytest.raises(ValueError) as exc_info:
            _check_migration_guard()
        err_msg = str(exc_info.value)
        assert legacy_key in err_msg
        assert new_key in err_msg
