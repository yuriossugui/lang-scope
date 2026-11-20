"""Tests for the chat endpoint (mocking the LangGraph RAG flow)."""

from unittest.mock import patch

from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_ask_success():
    with patch("app.api.v1.endpoints.chat.run_rag") as rag:
        rag.return_value = {
            "answer": "O saldo e 1000 reais, conforme o extrato.",
            "sources": [
                {
                    "document_id": "d1",
                    "document_name": "extrato.pdf",
                    "chunk_id": "c1",
                    "content": "saldo 1000",
                    "reference": "pagina 3",
                }
            ],
        }
        response = client.post(
            "/chat/ask",
            json={"question": "Qual o saldo?", "document_ids": ["d1"]},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "O saldo e 1000 reais, conforme o extrato."
    assert len(body["sources"]) == 1
    assert body["sources"][0]["document_name"] == "extrato.pdf"
    assert body["documents_consulted"] == ["d1"]
    rag.assert_called_once()
    args, _ = rag.call_args
    assert args[0] == "Qual o saldo?"
    assert args[1] == ["d1"]


def test_ask_without_document_ids():
    with patch("app.api.v1.endpoints.chat.run_rag") as rag:
        rag.return_value = {"answer": "resposta", "sources": []}
        response = client.post("/chat/ask", json={"question": "Qual o saldo?"})

    assert response.status_code == 200
    args, _ = rag.call_args
    assert args[0] == "Qual o saldo?"
    assert args[1] is None


def test_ask_empty_question_rejected():
    response = client.post("/chat/ask", json={"question": ""})
    assert response.status_code == 422
