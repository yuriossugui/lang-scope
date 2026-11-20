"""Client for the Node.js + Transformers.js embeddings microservice."""

import logging
from typing import List

import httpx

from app.core.exceptions import EmbeddingsServiceError
from app.config import Settings

logger = logging.getLogger("app")


class EmbeddingsService:
    """Calls the embeddings microservice to generate vectors for texts."""

    def __init__(self, settings: Settings) -> None:
        self.base_url = settings.EMBEDDINGS_SERVICE_URL.rstrip("/")
        self.model = settings.EMBEDDINGS_MODEL

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts via HTTP."""
        if not texts:
            return []
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/embed",
                    json={"texts": texts, "model": self.model},
                )
                response.raise_for_status()
                payload = response.json()
                embeddings = payload["embeddings"]
                if len(embeddings) != len(texts):
                    raise EmbeddingsServiceError(
                        f"Expected {len(texts)} embeddings, got {len(embeddings)}"
                    )
                return embeddings
        except httpx.HTTPError as exc:
            logger.error("Embeddings service request failed: %s", exc)
            raise EmbeddingsServiceError(str(exc))
