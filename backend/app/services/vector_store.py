"""Vector store abstraction backed by ChromaDB.

Stores document chunks, their embeddings and metadata. Metadata filtering makes it
possible to scope retrieval to specific documents and to delete every vector
belonging to a document (avoiding orphan vectors).
"""

import logging
from typing import List, Optional

import chromadb

from app.core.exceptions import VectorStoreError
from app.config import Settings
from app.models.document import DocumentChunk

logger = logging.getLogger("app")


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        try:
            self.client = chromadb.PersistentClient(path=settings.VECTOR_STORE_DIR)
            self.collection = self.client.get_or_create_collection(
                name=settings.COLLECTION_NAME
            )
        except Exception as exc:  # pragma: no cover - depends on chromadb availability
            logger.exception("Failed to initialize ChromaDB")
            raise VectorStoreError(str(exc))

    def add_chunks(
        self,
        document_id: str,
        original_filename: str,
        file_format: str,
        imported_at: str,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> None:
        """Persist chunks, embeddings and metadata for a document."""
        if not chunks:
            return
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = []
        for chunk in chunks:
            metadata = dict(chunk.metadata)
            metadata.update(
                {
                    "document_id": document_id,
                    "original_filename": original_filename,
                    "file_format": file_format,
                    "imported_at": imported_at,
                    "chunk_id": chunk.chunk_id,
                }
            )
            metadatas.append(metadata)
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as exc:
            logger.exception("Failed to add chunks to vector store")
            raise VectorStoreError(str(exc))

    def query(
        self,
        query_embedding: List[float],
        document_ids: Optional[List[str]] = None,
        n_results: int = 5,
    ) -> dict:
        """Retrieve the most similar chunks, optionally restricted to documents."""
        where = None
        if document_ids:
            where = {"document_id": {"$in": document_ids}}
        try:
            return self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
            )
        except Exception as exc:
            logger.exception("Vector store query failed")
            raise VectorStoreError(str(exc))

    def delete_document(self, document_id: str) -> int:
        """Remove every chunk/vector belonging to a document. Returns count deleted."""
        try:
            result = self.collection.get(where={"document_id": document_id})
            ids = result.get("ids", []) or []
            if ids:
                self.collection.delete(ids=ids)
            return len(ids)
        except Exception as exc:
            logger.exception("Failed to delete document vectors")
            raise VectorStoreError(str(exc))
