"""Document endpoints."""

from typing import List

from fastapi import APIRouter, UploadFile, File, Depends, status
from app.config import get_settings, Settings
from app.core.exceptions import (
    DocumentAlreadyExistsError,
    DocumentNotFoundError,
    DocumentProcessingError,
    InvalidFileError,
    UnsupportedFileTypeError,
)
from app.services.document_service import DocumentService
from app.schemas.document import (
    DocumentDeleteResponse,
    DocumentDetail,
    DocumentListItem,
    DocumentUploadResponse,
    ErrorResponse,
)

router = APIRouter()


def _build_service(settings: Settings) -> DocumentService:
    return DocumentService(settings)


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def upload_document(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
) -> DocumentUploadResponse:
    """Import and process a document (.ofx, .pdf, .csv)."""
    service = _build_service(settings)
    result = await service.import_document(file)
    return DocumentUploadResponse(**result)


@router.get(
    "/",
    response_model=List[DocumentListItem],
    status_code=status.HTTP_200_OK,
)
def list_documents(
    settings: Settings = Depends(get_settings),
) -> List[DocumentListItem]:
    """List all imported documents."""
    service = _build_service(settings)
    return [DocumentListItem(**item) for item in service.list_documents()]


@router.get(
    "/{document_id}",
    response_model=DocumentDetail,
    status_code=status.HTTP_200_OK,
    responses={404: {"model": ErrorResponse}},
)
def get_document(
    document_id: str,
    settings: Settings = Depends(get_settings),
) -> DocumentDetail:
    """Retrieve metadata of a single imported document."""
    service = _build_service(settings)
    return DocumentDetail(**service.get_document(document_id))


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    status_code=status.HTTP_200_OK,
    responses={404: {"model": ErrorResponse}},
)
def delete_document(
    document_id: str,
    settings: Settings = Depends(get_settings),
) -> DocumentDeleteResponse:
    """Delete a document and all its associated embeddings."""
    service = _build_service(settings)
    return DocumentDeleteResponse(**service.delete_document(document_id))
