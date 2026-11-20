"""PDF document extractor using pypdf."""

import logging
from typing import List

from pypdf import PdfReader

from app.services.extraction.base import DocumentExtractor
from app.models.document import DocumentChunk

logger = logging.getLogger("app")


class PDFExtractor(DocumentExtractor):
    def extract(self, file_path: str) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        try:
            reader = PdfReader(file_path)
        except Exception as exc:
            logger.exception("Failed to read PDF %s", file_path)
            raise
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                logger.warning("Failed to extract text from page %d", page_number)
                text = ""
            if text.strip():
                chunks.append(
                    DocumentChunk(
                        content=text,
                        metadata={
                            "page": page_number,
                            "source_type": "page",
                        },
                    )
                )
        return chunks
