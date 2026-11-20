"""Pydantic schemas for document endpoints."""

from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: str
    original_filename: str
    file_format: str
    imported_at: datetime
    chunk_count: int
    message: str = "Document imported successfully"


class DocumentListItem(BaseModel):
    document_id: str
    original_filename: str
    file_format: str
    imported_at: datetime
    chunk_count: int


class DocumentDetail(BaseModel):
    document_id: str
    original_filename: str
    file_format: str
    imported_at: datetime
    chunk_count: int
    metadata: Dict[str, Any]


class DocumentDeleteResponse(BaseModel):
    document_id: str
    deleted_chunks: int
    message: str = "Document and its embeddings deleted successfully"


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[Dict[str, Any]] = None
