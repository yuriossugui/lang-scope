# API de Consulta Inteligente a Documentos (RAG + LangChain + LangGraph)

API REST em **Python + FastAPI** que permite importar documentos e realizar perguntas em linguagem natural sobre o conteúdo deles, utilizando uma arquitetura **RAG (Retrieval-Augmented Generation)** com **LangChain**, **LangGraph** e o modelo **Google Gemini**.

## Sumário

- [Objetivo](#objetivo)
- [Arquitetura da solução](#arquitetura-da-solução)
- [Tecnologias utilizadas](#tecnologias-utilizadas)
- [Decisões técnicas](#decisões-técnicas)
- [Estrutura de diretórios](#estrutura-de-diretórios)
- [Pré-requisitos](#pré-requisitos)
- [Configuração das variáveis de ambiente](#configuração-das-variáveis-de-ambiente)
- [Instalação das dependências](#instalação-das-dependências)
- [Como executar o backend Python](#como-executar-o-backend-python)
- [Como executar o serviço de embeddings (JavaScript)](#como-executar-o-serviço-de-embeddings-javascript)
- [Execução com Docker Compose](#execução-com-docker-compose)
- [Exemplos de requisições e respostas](#exemplos-de-requisições-e-respostas)
- [Como executar os testes](#como-executar-os-testes)
- [Persistência e gerenciamento](#persistência-e-gerenciamento)

## Objetivo

Permitir importar arquivos (`.ofx`, `.pdf`, `.csv`) e, a partir deles, responder perguntas em linguagem natural com base exclusivamente no conteúdo importado, citando as fontes (documento, página/linha/registro) quando disponíveis.

## Arquitetura da solução

A solução separa responsabilidades em camadas (Clean Architecture simplificado):

1. **API (FastAPI)** — rotas REST, validação de entrada/saída com Pydantic, tratamento centralizado de exceções.
2. **Serviço de documentos** — orquestração da importação: validação de arquivo, extração de conteúdo, geração de embeddings, persistência no vector store e registro de metadados.
3. **Extratores de documento** — um extrator por formato (PDF, CSV, OFX), facilmente extensível para novos formatos.
4. **Serviço de embeddings (Node.js + Transformers.js)** — microserviço HTTP independente que gera vetores localmente com o modelo `Xenova/all-MiniLM-L6-v2` do Hugging Face.
5. **Vector Store (ChromaDB)** — armazena vetores, textos e metadados; permite busca por similaridade, filtragem por documento e exclusão completa de vetores de um documento (sem dados órfãos).
6. **LangChain** — adaptador `Embeddings` que consome os vetores gerados pelo serviço Node.js; integração com o LLM via `ChatGoogleGenerativeAI`.
7. **LangGraph** — orquestra o fluxo de consulta: **recuperação de contexto → geração da resposta**.

```
Cliente
  │ HTTP (FastAPI)
  ▼
+---------------------+      HTTP        +-----------------------------+
|   Backend Python    | <───────────────► | Embeddings Service (Node)   |
|  - Rotas / Pydantic |      /embed      |  - Transformers.js          |
|  - Serviço docs     |                  |  - all-MiniLM-L6-v2         |
|  - Vector Store     |                  +-----------------------------+
|  - LangChain/LangGraph
|  - LLM (Gemini)
+---------------------+        △        +-----------------------------+
                              │
                    ChromaDB (disco local)
```

## Tecnologias utilizadas

| Camada | Tecnologia |
|--------|------------|
| Linguagem backend | Python 3 |
| API REST | FastAPI |
| Validação | Pydantic |
| Processamento de documentos | LangChain, pypdf, ofxparse, csv (stdlib) |
| Orquestração do agente | LangGraph |
| LLM | Google Gemini (`ChatGoogleGenerativeAI`) |
| Embeddings | Transformers.js (`@xenova/transformers`, `Xenova/all-MiniLM-L6-v2`) via serviço Node.js/Express |
| Vector Store | ChromaDB (persistência local) |
| HTTP client | httpx |
| Testes | pytest, pytest-asyncio |
| Containerização | Docker, Docker Compose |

## Decisões técnicas

### Integração com Transformers.js (obrigatório ser JavaScript)
Transformers.js é uma biblioteca do ecossistema Node.js. Para usá-la sem substituí-la por embeddings em Python, foi criado um **microserviço Node.js/Express independente** (`embeddings-service/`) que expõe `POST /embed`. O backend Python consome esse serviço por HTTP. O LangChain consome os embeddings por meio de uma classe customizada `TransformersJsEmbeddings` (subclasse de `langchain_core.embeddings.Embeddings`) que faz chamadas síncronas HTTP ao microserviço.

### Modelo de embeddings
`Xenova/all-MiniLM-L6-v2` (384 dimensões): modelo gratuito, open source, leve, rápido e com boa qualidade para busca semântica em português/inglês. Executado localmente, sem custo e sem chave de API.

### Vector Store
ChromaDB: persistência em disco, suporte a metadados, filtragem por `document_id` (essencial para consulta restrita e exclusão sem dados órfãos) e API Python simples. Não requer servidor externo separado.

### LLM
`ChatGoogleGenerativeAI` do `langchain_google_genai`, conforme exigido pelo PRD. A chave `GOOGLE_API_KEY` é lida de variável de ambiente e nunca exposta nas respostas.

### Evitar duplicação
Antes de importar, é calculado o SHA-256 do conteúdo do arquivo. Se um documento com o mesmo hash já estiver registrado, a importação é rejeitada com `409`.

### Exclusão sem dados órfãos
Ao excluir um documento, todos os seus chunks/vetores são removidos do ChromaDB (filtro por `document_id`), o arquivo armazenado é apagado e o registro é removido do catálogo.

## Estrutura de diretórios

```
lang-chain-v2/
├── README.md
├── prd.md
├── .gitignore
├── docker-compose.yml
├── backend/
│   ├── .env.example
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI app + exception handler
│   │   ├── config.py               # Settings (pydantic-settings, .env)
│   │   ├── core/
│   │   │   ├── exceptions.py       # Exceções centralizadas
│   │   │   └── logging.py
│   │   ├── api/
│   │   │   ├── deps.py
│   │   │   └── v1/
│   │   │       └── endpoints/
│   │   │           ├── documents.py  # POST /upload, GET /, GET /{id}, DELETE /{id}
│   │   │           └── chat.py       # POST /ask
│   │   ├── models/
│   │   │   └── document.py         # DocumentChunk, DocumentInfo
│   │   ├── schemas/
│   │   │   ├── document.py         # Pydantic schemas de documentos
│   │   │   └── chat.py             # Pydantic schemas de chat
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── document_service.py # Orquestração import/list/get/delete
│   │   │   ├── embeddings_service.py # Cliente HTTP do serviço Node.js
│   │   │   ├── langchain_embeddings.py # Adaptador LangChain Embeddings
│   │   │   ├── vector_store.py     # ChromaDB wrapper
│   │   │   ├── llm.py              # ChatGoogleGenerativeAI wrapper
│   │   │   ├── langgraph_flow.py   # Grafo RAG (retrieve → generate)
│   │   │   └── extraction/
│   │   │       ├── base.py
│   │   │       ├── pdf.py          # pypdf
│   │   │       ├── csv.py          # csv (stdlib)
│   │   │       └── ofx.py          # ofxparse + fallback regex
│   │   └── tests/
│   │       ├── conftest.py
│   │       ├── test_extraction.py
│   │       ├── test_embeddings_service.py
│   │       ├── test_vector_store.py
│   │       ├── test_langgraph_flow.py
│   │       ├── test_document_upload.py
│   │       └── test_chat.py
├── embeddings-service/
│   ├── package.json
│   ├── Dockerfile
│   └── src/
│       ├── index.js            # Express app
│       └── embeddings.js       # pipeline Transformers.js + /embed
└── data/                       # gerado em runtime (uploads + vector store)
```

## Pré-requisitos

- Python 3.10+ (testado com Python 3.14)
- Node.js 18+ e npm
- Chave de API do Google Gemini (https://aistudio.google.com/app/apikey)
- (Opcional) Docker e Docker Compose

## Configuração das variáveis de ambiente

Copie o arquivo de exemplo e preencha os valores:

```bash
cp backend/.env.example backend/.env
```

Edite `backend/.env`, preenchendo ao menos:

```
GOOGLE_API_KEY=sua-chave-do-google-aqui
```

Variáveis disponíveis:

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `GOOGLE_API_KEY` | (obrigatório) | Chave da API Google Gemini |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Modelo do Gemini |
| `EMBEDDINGS_SERVICE_URL` | `http://localhost:8001` | URL do serviço de embeddings Node.js |
| `EMBEDDINGS_MODEL` | `Xenova/all-MiniLM-L6-v2` | Modelo Transformers.js |
| `EMBEDDING_DIMENSION` | `384` | Dimensão dos vetores |
| `VECTOR_STORE_DIR` | `./data/vector_store` | Pasta de persistência ChromaDB |
| `COLLECTION_NAME` | `documents` | Nome da coleção |
| `UPLOAD_DIR` | `./data/uploads` | Pasta de arquivos enviados |
| `MAX_FILE_SIZE_MB` | `10` | Tamanho máximo do arquivo |
| `DEBUG` | `false` | Modo depuração |

## Instalação das dependências

Backend Python:

```bash
cd backend
pip install -r requirements.txt
```

Serviço de embeddings (Node.js):

```bash
cd embeddings-service
npm install
```

## Como executar o backend Python

```bash
cd backend
# garanta que o .env esteja configurado
uvicorn app.main:app --reload --port 8000
```

A API fica disponível em `http://localhost:8000` e a documentação interativa em `http://localhost:8000/docs`.

## Como executar o serviço de embeddings (JavaScript)

O backend depende deste serviço para gerar embeddings. Ele deve estar rodando antes de importar documentos.

```bash
cd embeddings-service
npm start   # ou: node src/index.js
```

O serviço fica disponível em `http://localhost:8001` (`GET /health` e `POST /embed`).

Na primeira execução, o Transformers.js baixa o modelo `Xenova/all-MiniLM-L6-v2` (pode demorar alguns segundos).

## Execução com Docker Compose

Sobe backend e serviço de embeddings juntos:

```bash
docker compose up --build
```

Backend: `http://localhost:8000`
Embeddings: `http://localhost:8001`

Não esqueça de criar o `.env` com a `GOOGLE_API_KEY` antes de subir.

## Exemplos de requisições e respostas

### 1. Importar documento

```http
POST /documents/upload
Content-Type: multipart/form-data

file: <arquivo.extrato.csv>
```

Resposta (201):

```json
{
  "document_id": "doc_6bff5c870923cedc",
  "original_filename": "extrato.csv",
  "file_format": "csv",
  "imported_at": "2026-11-19T13:06:03.327591Z",
  "chunk_count": 2,
  "message": "Document imported successfully"
}
```

Arquivo com extensão não suportada → `400`. Conteúdo duplicado (mesmo hash) → `409`.

### 2. Listar documentos

```http
GET /documents
```

Resposta (200):

```json
[
  {
    "document_id": "doc_6bff5c870923cedc",
    "original_filename": "extrato.csv",
    "file_format": "csv",
    "imported_at": "2026-11-19T13:06:03.327591Z",
    "chunk_count": 2
  }
]
```

### 3. Consultar um documento

```http
GET /documents/doc_6bff5c870923cedc
```

### 4. Excluir documento e seus embeddings

```http
DELETE /documents/doc_6bff5c870923cedc
```

Resposta (200):

```json
{
  "document_id": "doc_6bff5c870923cedc",
  "deleted_chunks": 2,
  "message": "Document and its embeddings deleted successfully"
}
```

### 5. Fazer uma pergunta (RAG via LangGraph)

```http
POST /chat/ask
Content-Type: application/json

{
  "question": "Qual o saldo da conta?",
  "document_ids": ["doc_6bff5c870923cedc"]
}
```

Resposta (200):

```json
{
  "answer": "O saldo é 1000 reais, conforme o extrato.",
  "sources": [
    {
      "document_id": "doc_6bff5c870923cedc",
      "document_name": "extrato.csv",
      "chunk_id": "c1",
      "content": "2024-01-12 | Transferencia recebida | 2000.00",
      "reference": "linha 3"
    }
  ],
  "documents_consulted": ["doc_6bff5c870923cedc"]
}
```

Quando não há dados suficientes, a resposta informa que não é possível responder com base nos documentos importados.

## Como executar os testes

```bash
cd backend
pip install -r requirements.txt   # inclui pytest e pytest-asyncio
python -m pytest tests/ -v
```

Os testes usam **mocks** para serviços externos (embeddings Node.js, ChromaDB e LLM Google), não exigindo chamadas reais à API do Google nem ao serviço de embeddings durante a suíte automatizada.

## Persistência e gerenciamento

- Os documentos importados são registrados em `data/vector_store/documents_registry.json` e os arquivos originais ficam em `data/uploads/`.
- Os vetores e metadados são persistidos pelo ChromaDB em `data/vector_store/`.
- `GET /documents` lista todos os documentos com metadados.
- `GET /documents/{id}` consulta um documento individualmente.
- `DELETE /documents/{id}` remove o documento, seu arquivo e **todos** os vetores associados (sem dados órfãos).
- A deduplicação por hash SHA-256 evita reprocessamento de conteúdos já importados.
