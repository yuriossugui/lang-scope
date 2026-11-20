"""Tests for the LangGraph RAG flow (mocking vector store, embeddings, LLM)."""

from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings
from app.services.langgraph_flow import run_rag


@pytest.fixture
def settings() -> Settings:
    return Settings(GOOGLE_API_KEY="test-key")


def _patch_services():
    vs = patch("app.services.langgraph_flow.VectorStore").start()
    emb = patch("app.services.langgraph_flow.TransformersJsEmbeddings").start()
    llm = patch("app.services.langgraph_flow.LLMService").start()
    return vs, emb, llm


def test_run_rag_with_context(settings):
    vs, emb, llm = _patch_services()
    try:
        vs.return_value.query.return_value = {
            "ids": [["c1"]],
            "documents": [["O saldo da conta e 1000 reais"]],
            "metadatas": [[
                {"document_id": "d1", "original_filename": "extrato.pdf",
                 "chunk_id": "c1", "page": 3}
            ]],
        }
        emb.return_value.embed_query.return_value = [0.1] * 384
        llm.return_value.generate.return_value = "O saldo e 1000 reais."

        result = run_rag("Qual o saldo?", None, settings)

        assert "O saldo e 1000 reais" in result["answer"]
        assert len(result["sources"]) == 1
        assert result["sources"][0]["document_name"] == "extrato.pdf"
        assert result["sources"][0]["reference"] == "3"
        vs.return_value.query.assert_called_once()
        llm.return_value.generate.assert_called_once()
    finally:
        patch.stopall()


def test_run_rag_no_context(settings):
    vs, emb, llm = _patch_services()
    try:
        vs.return_value.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]]}
        emb.return_value.embed_query.return_value = [0.1] * 384

        result = run_rag("Qualquer coisa", None, settings)

        assert "dados suficientes" in result["answer"]
        assert result["sources"] == []
        llm.return_value.generate.assert_not_called()
    finally:
        patch.stopall()


def test_run_rag_restricted_to_documents(settings):
    vs, emb, llm = _patch_services()
    try:
        vs.return_value.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
        }
        emb.return_value.embed_query.return_value = [0.1] * 384
        llm.return_value.generate.return_value = "resposta"

        run_rag("X", ["d1"], settings)

        kwargs = vs.return_value.query.call_args.kwargs
        assert kwargs["document_ids"] == ["d1"]
    finally:
        patch.stopall()
