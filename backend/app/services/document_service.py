"""Document service: orchestration of import, list, get and delete."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import UploadFile

from app.core.exceptions import (
    DocumentAlreadyExistsError,
    DocumentNotFoundError,
    DocumentProcessingError,
    InvalidFileError,
    UnsupportedFileTypeError,
)
from app.core.logging import logger
from app.config import Settings
from app.models.document import DocumentChunk
from app.services.extraction.base import DocumentExtractor
from app.services.extraction.csv import CSVExtractor
from app.services.extraction.ofx import OFXExtractor
from app.services.extraction.pdf import PDFExtractor
from app.services.embeddings_service import EmbeddingsService
from app.services.vector_store import VectorStore

_EXTENSION_MAP: Dict[str, type[DocumentExtractor]] = {
    ".pdf": PDFExtractor,
    ".csv": CSVExtractor,
    ".ofx": OFXExtractor,
}


class DocumentService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.vector_store_dir = Path(settings.VECTOR_STORE_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.vector_store_dir / "documents_registry.json"
        self.embeddings_service = EmbeddingsService(settings)
        self.vector_store = VectorStore(settings)

    def _load_registry(self) -> Dict[str, Any]:
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as handle:
                    return json.load(handle)
            except json.JSONDecodeError:
                logger.warning("Corrupted documents registry, resetting")
                return {}
        return {}

    def _save_registry(self, registry: Dict[str, Any]) -> None:
        with open(self.registry_path, "w", encoding="utf-8") as handle:
            json.dump(registry, handle, ensure_ascii=False, indent=2, default=str)

    def _get_extractor(self, extension: str) -> DocumentExtractor:
        ext = extension.lower()
        if ext not in _EXTENSION_MAP:
            raise UnsupportedFileTypeError(ext)
        return _EXTENSION_MAP[ext]()

    async def import_document(self, file: UploadFile) -> Dict[str, Any]:
        filename = file.filename or "unknown"
        extension = Path(filename).suffix.lower()
        if not extension:
            raise InvalidFileError("File has no extension")
        if extension not in self.settings.ALLOWED_EXTENSIONS:
            raise UnsupportedFileTypeError(extension)

        contents = await file.read()
        size_mb = len(contents) / (1024 * 1024)
        if size_mb > self.settings.MAX_FILE_SIZE_MB:
            raise InvalidFileError(
                f"File size {size_mb:.2f}MB exceeds limit of "
                f"{self.settings.MAX_FILE_SIZE_MB}MB"
            )

        file_hash = hashlib.sha256(contents).hexdigest()
        registry = self._load_registry()
        for document in registry.values():
            if document.get("file_hash") == file_hash:
                raise DocumentAlreadyExistsError(document["document_id"])

        document_id = (
            f"doc_{hashlib.sha256((filename + file_hash).encode()).hexdigest()[:16]}"
        )
        imported_at = datetime.now(timezone.utc)

        stored_filename = f"{document_id}_{filename}"
        stored_path = self.upload_dir / stored_filename
        with open(stored_path, "wb") as handle:
            handle.write(contents)

        try:
            extractor = self._get_extractor(extension)
            chunks = extractor.extract(str(stored_path))
        except UnsupportedFileTypeError:
            raise
        except Exception as exc:
            logger.exception("Extraction failed for %s", filename)
            raise DocumentProcessingError(str(exc))

        if not chunks:
            raise DocumentProcessingError("No content could be extracted from the file")

        for index, chunk in enumerate(chunks, start=1):
            chunk.metadata.setdefault("chunk_index", index)
            chunk.metadata["imported_at"] = imported_at.isoformat()

        texts = [chunk.content for chunk in chunks]
        try:
            embeddings = await self.embeddings_service.embed_texts(texts)
        except Exception as exc:
            logger.exception("Embedding generation failed for %s", filename)
            raise DocumentProcessingError(f"Embedding generation failed: {exc}")

        imported_at_iso = imported_at.isoformat()
        self.vector_store.add_chunks(
            document_id=document_id,
            original_filename=filename,
            file_format=extension.lstrip("."),
            imported_at=imported_at_iso,
            chunks=chunks,
            embeddings=embeddings,
        )

        registry[document_id] = {
            "document_id": document_id,
            "original_filename": filename,
            "file_format": extension.lstrip("."),
            "imported_at": imported_at_iso,
            "file_hash": file_hash,
            "chunk_count": len(chunks),
            "stored_filename": stored_filename,
        }
        self._save_registry(registry)

        return {
            "document_id": document_id,
            "original_filename": filename,
            "file_format": extension.lstrip("."),
            "imported_at": imported_at_iso,
            "chunk_count": len(chunks),
            "message": "Document imported successfully",
        }

    def list_documents(self) -> List[Dict[str, Any]]:
        registry = self._load_registry()
        return [
            {
                "document_id": d["document_id"],
                "original_filename": d["original_filename"],
                "file_format": d["file_format"],
                "imported_at": d["imported_at"],
                "chunk_count": d["chunk_count"],
            }
            for d in registry.values()
        ]

    def get_document(self, document_id: str) -> Dict[str, Any]:
        registry = self._load_registry()
        if document_id not in registry:
            raise DocumentNotFoundError(document_id)
        d = registry[document_id]
        return {
            "document_id": d["document_id"],
            "original_filename": d["original_filename"],
            "file_format": d["file_format"],
            "imported_at": d["imported_at"],
            "chunk_count": d["chunk_count"],
            "metadata": {
                "file_hash": d.get("file_hash"),
                "stored_filename": d.get("stored_filename"),
            },
        }

    def delete_document(self, document_id: str) -> Dict[str, Any]:
        registry = self._load_registry()
        if document_id not in registry:
            raise DocumentNotFoundError(document_id)
        document = registry[document_id]

        deleted_chunks = self.vector_store.delete_document(document_id)

        stored_path = self.upload_dir / document.get("stored_filename", "")
        if stored_path.exists():
            stored_path.unlink()

        del registry[document_id]
        self._save_registry(registry)

        return {
            "document_id": document_id,
            "deleted_chunks": deleted_chunks,
            "message": "Document and its embeddings deleted successfully",
        }
