"""Domain models for documents and chunks."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass
class DocumentChunk:
    """A piece of text extracted from a document, ready for embedding."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = field(default_factory=lambda: __import__("uuid").uuid4().hex)


@dataclass
class DocumentInfo:
    """Metadata describing an imported document."""

    document_id: str
    original_filename: str
    file_format: str
    imported_at: datetime
    file_hash: str
    chunk_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
