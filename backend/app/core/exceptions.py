"""Centralized application exceptions."""

from typing import Any, Dict, Optional


class AppException(Exception):
    """Base application exception carrying an HTTP status code and details."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class DocumentNotFoundError(AppException):
    def __init__(self, document_id: str) -> None:
        super().__init__(
            message=f"Document not found: {document_id}",
            status_code=404,
            details={"document_id": document_id},
        )


class InvalidFileError(AppException):
    def __init__(self, reason: str) -> None:
        super().__init__(
            message="Invalid file",
            status_code=400,
            details={"reason": reason},
        )


class UnsupportedFileTypeError(AppException):
    def __init__(self, extension: str) -> None:
        super().__init__(
            message=f"Unsupported file type: {extension}",
            status_code=400,
            details={"extension": extension},
        )


class DocumentProcessingError(AppException):
    def __init__(self, message: str) -> None:
        super().__init__(
            message=f"Document processing error: {message}",
            status_code=422,
        )


class DocumentAlreadyExistsError(AppException):
    def __init__(self, document_id: str) -> None:
        super().__init__(
            message=f"Document already imported: {document_id}",
            status_code=409,
            details={"document_id": document_id},
        )


class EmbeddingsServiceError(AppException):
    def __init__(self, message: str) -> None:
        super().__init__(
            message=f"Embeddings service error: {message}",
            status_code=502,
        )


class VectorStoreError(AppException):
    def __init__(self, message: str) -> None:
        super().__init__(
            message=f"Vector store error: {message}",
            status_code=500,
        )


class LLMError(AppException):
    def __init__(self, message: str) -> None:
        super().__init__(
            message=f"LLM error: {message}",
            status_code=502,
        )
