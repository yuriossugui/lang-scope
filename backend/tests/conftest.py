"""Pytest configuration and shared fixtures."""

import os

# Set environment variables before importing the application so settings load
# without requiring a real .env or external credentials.
os.environ.setdefault("GOOGLE_API_KEY", "test-key")
os.environ.setdefault("EMBEDDINGS_SERVICE_URL", "http://localhost:8001")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
