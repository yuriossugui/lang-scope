"""LangGraph orchestration of the RAG query flow.

Graph: retrieve (vector search) -> generate (LLM with context) -> END.
"""

import logging
from typing import List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from app.core.logging import logger
from app.services.vector_store import VectorStore
from app.services.langchain_embeddings import TransformersJsEmbeddings
from app.services.llm import LLMService
from app.config import Settings


class RAGState(TypedDict, total=False):
    question: str
    document_ids: Optional[List[str]]
    query_embedding: List[float]
    context: List[dict]
    answer: str
    sources: List[dict]


def build_rag_graph(settings: Settings):
    vector_store = VectorStore(settings)
    embeddings = TransformersJsEmbeddings(settings)
    llm = LLMService(settings)

    def retrieve_node(state: RAGState) -> RAGState:
        question = state["question"]
        document_ids = state.get("document_ids")
        query_embedding = embeddings.embed_query(question)
        results = vector_store.query(
            query_embedding, document_ids=document_ids, n_results=5
        )
        ids = (results.get("ids") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        context = []
        for position, chunk_id in enumerate(ids):
            metadata = metadatas[position] if position < len(metadatas) else {}
            reference = None
            for key in ("page", "line", "record", "chunk_index"):
                if metadata.get(key) is not None:
                    reference = str(metadata.get(key))
                    break
            context.append(
                {
                    "chunk_id": metadata.get("chunk_id", chunk_id),
                    "document_id": metadata.get("document_id"),
                    "document_name": metadata.get("original_filename"),
                    "content": documents[position] if position < len(documents) else "",
                    "reference": reference,
                    "metadata": metadata,
                }
            )
        return {**state, "query_embedding": query_embedding, "context": context}

    def generate_node(state: RAGState) -> RAGState:
        question = state["question"]
        context = state.get("context", []) or []
        if not context:
            return {
                **state,
                "answer": (
                    "Não há dados suficientes nos documentos importados para responder "
                    "à pergunta."
                ),
                "sources": [],
            }

        context_text = "\n\n".join(
            f"[Fonte: {item.get('document_name')} - "
            f"{item.get('reference', '')}]\n{item.get('content', '')}"
            for item in context
        )
        prompt = (
            "Você é um assistente que responde perguntas usando EXCLUSIVAMENTE o "
            "contexto fornecido.\n"
            "Regras:\n"
            "1. Responda somente com informações presentes no contexto.\n"
            "2. Não invente informações que não estejam no contexto.\n"
            "3. Se não houver dados suficientes, diga claramente que não é possível "
            "responder.\n"
            "4. Identifique os documentos utilizados como fonte e indique página, "
            "linha ou registro quando disponível.\n\n"
            f"Contexto:\n{context_text}\n\n"
            f"Pergunta: {question}\n\nResposta:"
        )
        answer = llm.generate(prompt)
        sources = [
            {
                "document_id": item.get("document_id"),
                "document_name": item.get("document_name"),
                "chunk_id": item.get("chunk_id"),
                "content": item.get("content"),
                "reference": item.get("reference"),
            }
            for item in context
        ]
        return {**state, "answer": answer, "sources": sources}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def run_rag(question: str, document_ids: Optional[List[str]], settings: Settings) -> dict:
    """Execute the RAG flow and return the answer plus sources."""
    rag_app = build_rag_graph(settings)
    final_state = rag_app.invoke({"question": question, "document_ids": document_ids})
    return {
        "answer": final_state.get("answer", ""),
        "sources": final_state.get("sources", []),
    }
