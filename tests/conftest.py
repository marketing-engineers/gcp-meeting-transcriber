import os
import pytest

# Set fallback environment variables for test execution when .env is not present
os.environ.setdefault("GCP_PROJECT_ID", "test-project")
os.environ.setdefault("GCS_BUCKET_NAME", "test-bucket")


@pytest.fixture(autouse=True)
def reset_gcp_caches():
    """Resets global cached clients and credentials before and after each test."""
    from src.auth import reset_credentials_cache
    from src.gcs import reset_client_cache

    reset_credentials_cache()
    reset_client_cache()
    yield
    reset_credentials_cache()
    reset_client_cache()

