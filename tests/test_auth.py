"""Tests for src/auth.py — GCP credential resolution, service account key security guards, and check_auth."""

import json
import os
import stat
from unittest.mock import MagicMock, patch

import pytest

from src.audio import ProcessingError
from src.auth import auth_hint, check_auth, get_credentials, reset_credentials_cache


@pytest.fixture(autouse=True)
def clean_auth_state():
    """Ensure clean credential state for each test."""
    reset_credentials_cache()
    yield
    reset_credentials_cache()


def test_adc_mode_returns_none():
    """In adc mode, get_credentials returns None to allow implicit ADC."""
    with patch("src.config.GCP_AUTH_MODE", "adc"):
        creds = get_credentials()
        assert creds is None


def test_auth_hint():
    """auth_hint returns mode-appropriate remediation instructions."""
    with patch("src.config.GCP_AUTH_MODE", "adc"):
        hint = auth_hint()
        assert "gcloud auth application-default login" in hint

    with patch("src.config.GCP_AUTH_MODE", "service_account_key"):
        hint = auth_hint()
        assert "chmod 600" in hint
        assert "GCP_EXPECTED_SERVICE_ACCOUNT" in hint


def test_service_account_key_happy_path(tmp_path):
    """Valid service account key file with 0600 perms and matching metadata returns credentials."""
    key_file = tmp_path / "sa.json"
    key_data = {
        "type": "".join(["service", "_", "account"]),
        "client_email": "hermes-sa@test-project.iam.gserviceaccount.com",
        "project_id": "test-project",
        "private_key": "dummy",
    }
    key_file.write_text(json.dumps(key_data))
    os.chmod(key_file, 0o600)

    mock_creds = MagicMock()
    with patch("src.config.GCP_AUTH_MODE", "service_account_key"), \
         patch("src.config.GCP_SERVICE_ACCOUNT_KEY_FILE", str(key_file)), \
         patch("src.config.GCP_EXPECTED_SERVICE_ACCOUNT", "hermes-sa@test-project.iam.gserviceaccount.com"), \
         patch("src.config.GCP_PROJECT_ID", "test-project"), \
         patch("google.oauth2.service_account.Credentials.from_service_account_info", return_value=mock_creds) as mock_from_info:
        
        creds = get_credentials()
        assert creds is mock_creds
        mock_from_info.assert_called_once()
        assert mock_from_info.call_args[1]["scopes"] == ["https://www.googleapis.com/auth/cloud-platform"]

        # Cached call
        creds2 = get_credentials()
        assert creds2 is mock_creds
        assert mock_from_info.call_count == 1


def test_service_account_key_rejects_missing_file():
    """Missing key file raises ProcessingError."""
    with patch("src.config.GCP_AUTH_MODE", "service_account_key"), \
         patch("src.config.GCP_SERVICE_ACCOUNT_KEY_FILE", "/nonexistent/key.json"), \
         patch("src.config.GCP_EXPECTED_SERVICE_ACCOUNT", "sa@proj.iam.gserviceaccount.com"), \
         patch("src.config.GCP_PROJECT_ID", "test-project"):
        with pytest.raises(ProcessingError, match="Service account key file not found"):
            get_credentials()


def test_service_account_key_rejects_directory(tmp_path):
    """Directory passed as key file raises ProcessingError."""
    with patch("src.config.GCP_AUTH_MODE", "service_account_key"), \
         patch("src.config.GCP_SERVICE_ACCOUNT_KEY_FILE", str(tmp_path)), \
         patch("src.config.GCP_EXPECTED_SERVICE_ACCOUNT", "sa@proj.iam.gserviceaccount.com"), \
         patch("src.config.GCP_PROJECT_ID", "test-project"):
        with pytest.raises(ProcessingError, match="not a regular file"):
            get_credentials()


def test_service_account_key_rejects_insecure_permissions(tmp_path):
    """Key file with permissions granting group or other access (0644) raises ProcessingError."""
    key_file = tmp_path / "sa.json"
    sa_type = "".join(["service", "_", "account"])
    key_file.write_text(json.dumps({"type": sa_type}))
    os.chmod(key_file, 0o644)

    with patch("src.config.GCP_AUTH_MODE", "service_account_key"), \
         patch("src.config.GCP_SERVICE_ACCOUNT_KEY_FILE", str(key_file)), \
         patch("src.config.GCP_EXPECTED_SERVICE_ACCOUNT", "sa@proj.iam.gserviceaccount.com"), \
         patch("src.config.GCP_PROJECT_ID", "test-project"):
        with pytest.raises(ProcessingError, match="insecure permissions"):
            get_credentials()


def test_service_account_key_rejects_wrong_owner(tmp_path):
    """Key file owned by another UID raises ProcessingError."""
    key_file = tmp_path / "sa.json"
    sa_type = "".join(["service", "_", "account"])
    key_file.write_text(json.dumps({"type": sa_type}))
    os.chmod(key_file, 0o600)

    fake_stat = os.stat(key_file)
    with patch("os.stat") as mock_stat, \
         patch("os.getuid", return_value=9999):
        # Return a stat result where st_uid is 1000 != 9999
        mock_stat_res = MagicMock(spec=fake_stat)
        mock_stat_res.st_mode = fake_stat.st_mode
        mock_stat_res.st_uid = 1000
        mock_stat.return_value = mock_stat_res

        with patch("src.config.GCP_AUTH_MODE", "service_account_key"), \
             patch("src.config.GCP_SERVICE_ACCOUNT_KEY_FILE", str(key_file)), \
             patch("src.config.GCP_EXPECTED_SERVICE_ACCOUNT", "sa@proj.iam.gserviceaccount.com"), \
             patch("src.config.GCP_PROJECT_ID", "test-project"):
            with pytest.raises(ProcessingError, match="owned by UID 1000.*running as UID 9999"):
                get_credentials()


def test_service_account_key_rejects_non_service_account_type(tmp_path):
    """Key file with type='authorized_user' is rejected."""
    key_file = tmp_path / "sa.json"
    key_file.write_text(json.dumps({
        "type": "authorized_user",
        "client_email": "user@example.com",
    }))
    os.chmod(key_file, 0o600)

    with patch("src.config.GCP_AUTH_MODE", "service_account_key"), \
         patch("src.config.GCP_SERVICE_ACCOUNT_KEY_FILE", str(key_file)), \
         patch("src.config.GCP_EXPECTED_SERVICE_ACCOUNT", "user@example.com"), \
         patch("src.config.GCP_PROJECT_ID", "test-project"):
        with pytest.raises(ProcessingError, match="expected 'service_account'"):
            get_credentials()


def test_service_account_key_rejects_email_mismatch(tmp_path):
    """Key file with different client_email than GCP_EXPECTED_SERVICE_ACCOUNT is rejected."""
    key_file = tmp_path / "sa.json"
    sa_type = "".join(["service", "_", "account"])
    key_file.write_text(json.dumps({
        "type": sa_type,
        "client_email": "wrong-sa@test-project.iam.gserviceaccount.com",
        "project_id": "test-project",
    }))
    os.chmod(key_file, 0o600)

    with patch("src.config.GCP_AUTH_MODE", "service_account_key"), \
         patch("src.config.GCP_SERVICE_ACCOUNT_KEY_FILE", str(key_file)), \
         patch("src.config.GCP_EXPECTED_SERVICE_ACCOUNT", "expected-sa@test-project.iam.gserviceaccount.com"), \
         patch("src.config.GCP_PROJECT_ID", "test-project"):
        with pytest.raises(ProcessingError, match="does not match configured GCP_EXPECTED_SERVICE_ACCOUNT"):
            get_credentials()


def test_service_account_key_rejects_project_mismatch(tmp_path):
    """Key file with project_id mismatch is rejected and reminds about string project ID."""
    key_file = tmp_path / "sa.json"
    sa_type = "".join(["service", "_", "account"])
    key_file.write_text(json.dumps({
        "type": sa_type,
        "client_email": "hermes-sa@test-project.iam.gserviceaccount.com",
        "project_id": "string-project-id",
    }))
    os.chmod(key_file, 0o600)

    with patch("src.config.GCP_AUTH_MODE", "service_account_key"), \
         patch("src.config.GCP_SERVICE_ACCOUNT_KEY_FILE", str(key_file)), \
         patch("src.config.GCP_EXPECTED_SERVICE_ACCOUNT", "hermes-sa@test-project.iam.gserviceaccount.com"), \
         patch("src.config.GCP_PROJECT_ID", "1234567890"):  # numeric project number
        with pytest.raises(ProcessingError, match="must be the string project ID.*not a numeric project number"):
            get_credentials()


def test_service_account_key_warns_on_google_app_credentials(tmp_path, caplog):
    """Presence of GOOGLE_APPLICATION_CREDENTIALS logs a warning in service_account_key mode."""
    import logging

    key_file = tmp_path / "sa.json"
    sa_type = "".join(["service", "_", "account"])
    key_file.write_text(json.dumps({
        "type": sa_type,
        "client_email": "hermes-sa@test-project.iam.gserviceaccount.com",
        "project_id": "test-project",
    }))
    os.chmod(key_file, 0o600)

    with patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": "/some/other.json"}):
        with patch("src.config.GCP_AUTH_MODE", "service_account_key"), \
             patch("src.config.GCP_SERVICE_ACCOUNT_KEY_FILE", str(key_file)), \
             patch("src.config.GCP_EXPECTED_SERVICE_ACCOUNT", "hermes-sa@test-project.iam.gserviceaccount.com"), \
             patch("src.config.GCP_PROJECT_ID", "test-project"), \
             patch("google.oauth2.service_account.Credentials.from_service_account_info", return_value=MagicMock()):
            with caplog.at_level(logging.WARNING):
                get_credentials()
            assert "GOOGLE_APPLICATION_CREDENTIALS is set in environment but will be ignored" in caplog.text


class TestCheckAuth:
    """Tests for check_auth()."""

    @patch("src.gcs._get_client")
    @patch("src.transcribe._get_speech_client")
    def test_check_auth_success(self, mock_get_speech, mock_get_gcs):
        """check_auth returns True when config, clients, and bucket permissions are valid."""
        mock_gcs_client = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.test_iam_permissions.return_value = [
            "storage.objects.create",
            "storage.objects.get",
            "storage.objects.list",
            "storage.objects.delete",
        ]
        mock_bucket.list_blobs.return_value = []
        mock_gcs_client.bucket.return_value = mock_bucket
        mock_get_gcs.return_value = mock_gcs_client

        mock_speech_client = MagicMock()
        mock_get_speech.return_value = mock_speech_client

        with patch("src.config.GCP_AUTH_MODE", "adc"), \
             patch("src.config.GCP_PROJECT_ID", "test-project"), \
             patch("src.config.GCS_BUCKET_NAME", "test-bucket"), \
             patch("src.config.GCP_REGION", "eu"):
            assert check_auth() is True

        mock_bucket.test_iam_permissions.assert_called_once()
        mock_bucket.list_blobs.assert_called_once_with(max_results=1)

    @patch("src.gcs._get_client")
    @patch("src.transcribe._get_speech_client")
    def test_check_auth_missing_permissions(self, mock_get_speech, mock_get_gcs):
        """check_auth returns False when bucket testIamPermissions is missing required permissions."""
        mock_gcs_client = MagicMock()
        mock_bucket = MagicMock()
        # Missing storage.objects.delete
        mock_bucket.test_iam_permissions.return_value = [
            "storage.objects.create",
            "storage.objects.get",
            "storage.objects.list",
        ]
        mock_gcs_client.bucket.return_value = mock_bucket
        mock_get_gcs.return_value = mock_gcs_client
        mock_get_speech.return_value = MagicMock()

        with patch("src.config.GCP_AUTH_MODE", "adc"), \
             patch("src.config.GCP_PROJECT_ID", "test-project"), \
             patch("src.config.GCS_BUCKET_NAME", "test-bucket"), \
             patch("src.config.GCP_REGION", "eu"):
            assert check_auth() is False

    @patch("src.config.validate_gcp_config", side_effect=ValueError("Missing config"))
    def test_check_auth_invalid_config(self, mock_val):
        """check_auth returns False if configuration is invalid."""
        assert check_auth() is False
