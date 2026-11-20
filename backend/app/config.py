"""Application configuration loaded from environment variables."""

from functools import lru_cache
from typing import Set

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    All values can be overridden through environment variables or a ``.env`` file.
    """

    APP_NAME: str = "Intelligent Document Query API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Google Gemini (LLM)
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.1-flash-lite"

    # Embeddings microservice (Node.js + Transformers.js)
    EMBEDDINGS_SERVICE_URL: str = "http://localhost:8001"
    EMBEDDINGS_MODEL: str = "Xenova/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # Vector store
    VECTOR_STORE_DIR: str = "./data/vector_store"
    COLLECTION_NAME: str = "documents"

    # Uploads
    UPLOAD_DIR: str = "./data/uploads"
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: Set[str] = {".ofx", ".pdf", ".csv"}

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
