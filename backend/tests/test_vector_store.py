"""Tests for the ChromaDB vector store wrapper (mocking chromadb)."""

from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings
from app.models.document import DocumentChunk
from app.services.vector_store import VectorStore


@pytest.fixture
def store():
    with patch("app.services.vector_store.chromadb") as mock_chromadb:
        fake_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = fake_collection
        mock_chromadb.PersistentClient.return_value = mock_client
        yield VectorStore(Settings()), fake_collection


def test_add_chunks(store):
    vector_store, collection = store
    chunks = [
        DocumentChunk(content="chunk one", metadata={"page": 1}),
        DocumentChunk(content="chunk two", metadata={"page": 2}),
    ]
    vector_store.add_chunks(
        document_id="doc-1",
        original_filename="a.pdf",
        file_format="pdf",
        imported_at="2024-01-01T00:00:00+00:00",
        chunks=chunks,
        embeddings=[[0.1] * 384, [0.2] * 384],
    )
    collection.add.assert_called_once()
    call_kwargs = collection.add.call_args.kwargs
    assert len(call_kwargs["ids"]) == 2
    assert call_kwargs["documents"] == ["chunk one", "chunk two"]
    assert all(m["document_id"] == "doc-1" for m in call_kwargs["metadatas"])
    assert all(m["chunk_id"] for m in call_kwargs["metadatas"])


def test_query(store):
    vector_store, collection = store
    collection.query.return_value = {
        "ids": [["c1"]],
        "documents": [["hello world"]],
        "metadatas": [[{"document_id": "doc-1", "page": 1}]],
    }
    result = vector_store.query([0.1] * 384, document_ids=["doc-1"], n_results=5)
    assert result["ids"][0][0] == "c1"
    collection.query.assert_called_once()
    kwargs = collection.query.call_args.kwargs
    assert kwargs["where"] == {"document_id": {"$in": ["doc-1"]}}


def test_delete_document(store):
    vector_store, collection = store
    collection.get.return_value = {"ids": ["c1", "c2", "c3"]}
    deleted = vector_store.delete_document("doc-1")
    assert deleted == 3
    collection.delete.assert_called_once_with(ids=["c1", "c2", "c3"])
