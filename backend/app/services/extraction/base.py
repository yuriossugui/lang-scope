"""Abstract base class for document extractors."""

from abc import ABC, abstractmethod
from typing import List

from app.models.document import DocumentChunk


class DocumentExtractor(ABC):
    """Extracts textual chunks from a document file."""

    @abstractmethod
    def extract(self, file_path: str) -> List[DocumentChunk]:
        """Return a list of chunks extracted from the file at ``file_path``."""
