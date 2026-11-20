"""FastAPI application entry point."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.exceptions import AppException
from app.core.logging import setup_logging
from app.api.v1.endpoints import documents, chat

setup_logging()

app = FastAPI(
    title="Intelligent Document Query API",
    description=(
        "RAG-based REST API for importing documents and querying their content "
        "in natural language using LangChain, LangGraph and Google Gemini."
    ),
    version="1.0.0",
)


@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "detail": exc.details},
    )


app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])


@app.get("/health", tags=["health"])
async def health() -> dict:
    return {"status": "ok"}
