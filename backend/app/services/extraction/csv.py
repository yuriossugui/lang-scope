"""CSV document extractor using the Python standard library."""

import csv
import logging
from typing import List

from app.services.extraction.base import DocumentExtractor
from app.models.document import DocumentChunk

logger = logging.getLogger("app")


class CSVExtractor(DocumentExtractor):
    def extract(self, file_path: str) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        try:
            with open(file_path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for line_number, row in enumerate(reader, start=2):
                    content = " | ".join(
                        f"{key}: {value}"
                        for key, value in row.items()
                        if value is not None and str(value).strip()
                    )
                    if content.strip():
                        chunks.append(
                            DocumentChunk(
                                content=content,
                                metadata={
                                    "line": line_number,
                                    "record": line_number - 1,
                                    "source_type": "line",
                                },
                            )
                        )
        except Exception as exc:
            logger.exception("Failed to read CSV %s", file_path)
            raise
        return chunks
