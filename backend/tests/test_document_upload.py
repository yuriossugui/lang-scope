"""Tests for document upload endpoint (mocking the document service)."""

import io
from unittest.mock import AsyncMock, patch

from app.core.exceptions import (
    DocumentAlreadyExistsError,
    DocumentNotFoundError,
    UnsupportedFileTypeError,
)
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def _fake_upload_result():
    return {
        "document_id": "doc_abc123",
        "original_filename": "extrato.pdf",
        "file_format": "pdf",
        "imported_at": "2024-01-01T00:00:00+00:00",
        "chunk_count": 3,
        "message": "Document imported successfully",
    }


def test_upload_success():
    with patch(
        "app.api.v1.endpoints.documents.DocumentService"
    ) as service_cls:
        service_cls.return_value.import_document = AsyncMock(
            return_value=_fake_upload_result()
        )
        files = {"file": ("extrato.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")}
        response = client.post("/documents/upload", files=files)

    assert response.status_code == 201
    body = response.json()
    assert body["document_id"] == "doc_abc123"
    assert body["original_filename"] == "extrato.pdf"
    assert body["chunk_count"] == 3


def test_upload_unsupported_extension():
    with patch(
        "app.api.v1.endpoints.documents.DocumentService"
    ) as service_cls:
        service_cls.return_value.import_document = AsyncMock(
            side_effect=UnsupportedFileTypeError(".txt")
        )
        files = {"file": ("notas.txt", io.BytesIO(b"hello world"), "text/plain")}
        response = client.post("/documents/upload", files=files)

    assert response.status_code == 400
    assert response.json()["error"] == "Unsupported file type: .txt"


def test_upload_already_exists():
    with patch(
        "app.api.v1.endpoints.documents.DocumentService"
    ) as service_cls:
        service_cls.return_value.import_document = AsyncMock(
            side_effect=DocumentAlreadyExistsError("doc_abc123")
        )
        files = {"file": ("extrato.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")}
        response = client.post("/documents/upload", files=files)

    assert response.status_code == 409


def test_list_documents():
    with patch(
        "app.api.v1.endpoints.documents.DocumentService"
    ) as service_cls:
        service_cls.return_value.list_documents.return_value = [
            {
                "document_id": "doc_1",
                "original_filename": "a.pdf",
                "file_format": "pdf",
                "imported_at": "2024-01-01T00:00:00+00:00",
                "chunk_count": 2,
            }
        ]
        response = client.get("/documents/")

    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["document_id"] == "doc_1"


def test_get_document():
    with patch(
        "app.api.v1.endpoints.documents.DocumentService"
    ) as service_cls:
        service_cls.return_value.get_document.return_value = {
            "document_id": "doc_1",
            "original_filename": "a.pdf",
            "file_format": "pdf",
            "imported_at": "2024-01-01T00:00:00+00:00",
            "chunk_count": 2,
            "metadata": {},
        }
        response = client.get("/documents/doc_1")

    assert response.status_code == 200
    assert response.json()["document_id"] == "doc_1"


def test_get_document_not_found():
    with patch(
        "app.api.v1.endpoints.documents.DocumentService"
    ) as service_cls:
        service_cls.return_value.get_document.side_effect = DocumentNotFoundError("missing")
        response = client.get("/documents/missing")

    assert response.status_code == 404


def test_delete_document():
    with patch(
        "app.api.v1.endpoints.documents.DocumentService"
    ) as service_cls:
        service_cls.return_value.delete_document.return_value = {
            "document_id": "doc_1",
            "deleted_chunks": 5,
            "message": "Document and its embeddings deleted successfully",
        }
        response = client.delete("/documents/doc_1")

    assert response.status_code == 200
    assert response.json()["deleted_chunks"] == 5


def test_delete_document_not_found():
    with patch(
        "app.api.v1.endpoints.documents.DocumentService"
    ) as service_cls:
        service_cls.return_value.delete_document.side_effect = DocumentNotFoundError("missing")
        response = client.delete("/documents/missing")

    assert response.status_code == 404
