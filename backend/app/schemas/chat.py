"""Pydantic schemas for chat endpoints."""

from typing import List, Optional
from pydantic import BaseModel, Field


class Source(BaseModel):
    document_id: str
    document_name: str
    chunk_id: str
    content: str
    reference: Optional[str] = None


class ChatAskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural language question")
    document_ids: Optional[List[str]] = Field(
        None, description="Optional list of document IDs to restrict the search"
    )


class ChatAskResponse(BaseModel):
    answer: str
    sources: List[Source] = []
    documents_consulted: List[str] = []
