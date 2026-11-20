"""Tests for the embeddings HTTP client (mocking the Node.js service)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import Settings
from app.services.embeddings_service import EmbeddingsService


@pytest.fixture
def settings() -> Settings:
    return Settings(GOOGLE_API_KEY="test-key")


@pytest.mark.asyncio
async def test_embed_texts_success(settings):
    service = EmbeddingsService(settings)
    fake_embeddings = [[0.1, 0.2], [0.3, 0.4]]

    mock_response = MagicMock()
    mock_response.json.return_value = {"embeddings": fake_embeddings}
    mock_response.raise_for_status.return_value = None

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = False

    with patch(
        "app.services.embeddings_service.httpx.AsyncClient",
        return_value=mock_client,
    ):
        result = await service.embed_texts(["texto um", "texto dois"])

    assert result == fake_embeddings
    mock_client.post.assert_called_once()
    args, kwargs = mock_client.post.call_args
    assert kwargs["json"]["texts"] == ["texto um", "texto dois"]


@pytest.mark.asyncio
async def test_embed_texts_empty(settings):
    service = EmbeddingsService(settings)
    assert await service.embed_texts([]) == []
