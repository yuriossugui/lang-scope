"""Chat endpoints."""

from fastapi import APIRouter, Depends, status
from app.config import get_settings, Settings
from app.services.langgraph_flow import run_rag
from app.schemas.chat import ChatAskRequest, ChatAskResponse, Source

router = APIRouter()


@router.post(
    "/ask",
    response_model=ChatAskResponse,
    status_code=status.HTTP_200_OK,
)
def ask(
    request: ChatAskRequest,
    settings: Settings = Depends(get_settings),
) -> ChatAskResponse:
    """Ask a natural language question over imported documents via LangGraph RAG."""
    result = run_rag(request.question, request.document_ids, settings)
    sources = [
        Source(
            document_id=item.get("document_id", ""),
            document_name=item.get("document_name", ""),
            chunk_id=item.get("chunk_id", ""),
            content=item.get("content", ""),
            reference=item.get("reference"),
        )
        for item in result.get("sources", [])
    ]
    documents_consulted = [
        item.get("document_id")
        for item in result.get("sources", [])
        if item.get("document_id")
    ]
    return ChatAskResponse(
        answer=result.get("answer", ""),
        sources=sources,
        documents_consulted=list(dict.fromkeys(documents_consulted)),
    )
