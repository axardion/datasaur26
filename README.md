# Datasaur 2026 | Qazcode Challenge

## Medical Diagnosis Assistant: Symptoms → ICD-10

An AI-powered clinical decision support system that converts patient symptoms into structured diagnoses with ICD-10 codes, built on Kazakhstan clinical protocols.

---

## Three RAG implementations

This repo contains **three** implementations of symptom → ICD-10 diagnosis, each in its own `src_*` directory:

| Implementation    | Directory         | Retrieval / indexing | LLM usage | Best for                          |
|-------------------|-------------------|----------------------|-----------|------------------------------------|
| **Direct RAG**    | `src_direct/`     | TF-IDF (sklearn), in-memory | DiReCT-style: premises + symptoms → diagnoses | Quick local runs, minimal setup |
| **Advanced RAG**  | `src_advanced_rag/` | BM25 + FAISS (E5), RRF fusion, protocol-first ICD ranking | Optional query expansion & rerank | Higher accuracy, offline index in `artifacts/` |
| **GraphRAG**      | `src_graph_rag/`  | Microsoft GraphRAG (entities, communities, local search) | Local search answer + completion | Rich knowledge graph, **Docker submission** |

### 1. Direct RAG (`src_direct/`)

- **Retrieval:** TF-IDF over protocol chunks (char ngrams, ~1200 chars per chunk). No separate index build.
- **Flow:** Load protocols from `CORPUS_DIR` → chunk → retrieve top-k by cosine similarity → diversify by protocol → send premises + symptoms to LLM (DiReCT-style) → parse top-N diagnoses.
- **Key files:** `direct_rag_server.py`, `rag.py`, `prompts.py`, `llm_client.py`.
- **Run:** `uv run uvicorn src_direct.direct_rag_server:app --host 127.0.0.1 --port 8000` (set `OPENAI_API_KEY`, optional `CORPUS_DIR`, `DIAGNOSIS_MODEL`).

### 2. Advanced RAG (`src_advanced_rag/`)

- **Retrieval:** Hybrid BM25 + vector (FAISS with sentence-transformers, e.g. E5). Results merged with **Reciprocal Rank Fusion (RRF)**. Ranking is **protocol-first**: aggregate chunk scores per protocol, then output ICDs in a strict order (primary → dotted block → dotted meta → block → meta → family).
- **Flow:** Offline **preprocess** builds `artifacts/` (BM25 chunks, FAISS index, ICD index). Server loads artifacts; for each query optionally expands queries and/or LLM-reranks; ICDRanker returns top diagnoses. Can fall back to protocol-first or corpus ICDs without LLM.
- **Key files:** `server.py`, `retrieval.py` (HybridRetriever), `icd_ranker.py`, `fusion.py`, `preprocess.py`, `llm_client.py`.
- **Run:** Build artifacts with `python preprocess.py --corpus <corpus.jsonl>`, then `uv run uvicorn src_advanced_rag.server:app --host 127.0.0.1 --port 8000` (env: `USE_QUERY_EXPANSION`, `USE_LLM_RERANK`).

### 3. GraphRAG (`src_graph_rag/`)

- **Retrieval:** Microsoft **GraphRAG**: knowledge graph (entities, communities, community reports) + **local search**. Embeddings via local sentence-transformers (e.g. multilingual-e5-large); no embedding API at inference.
- **Flow:** One-time **index** with `graphrag index --config src_graph_rag/settings.yaml` (produces `output/`, uses `data/graphrag_input/`, `prompts/`). Server loads index; for each symptom query runs GraphRAG `local_search` with a fixed response schema → LLM returns top-3 diagnoses with ICD-10 and explanations.
- **Key files:** `diagnose_server.py`, `settings.yaml`, `local_embedding.py`, `export_graphrag_documents.py`.
- **Run:** After building the index, `uv run uvicorn src_graph_rag.diagnose_server:app --host 127.0.0.1 --port 8000`. This is the **Docker submission** (see below).

---

## How to use the Dockerfile (GraphRAG submission)

This project uses a **GraphRAG** diagnosis server (`src_graph_rag`). The Dockerfile builds an image that serves on **port 8080**.

### 1. Prerequisites

Build the GraphRAG index locally so `output/`, `data/graphrag_input/`, and `prompts/` exist:

```bash
uv run graphrag index --config src_graph_rag/settings.yaml
```

### 2. Build

```bash
docker build -t submission .
```

### 3. Run

Server listens on port 8080. Pass your API key for the completion model (LLM):

```bash
docker run -p 8080:8080 -e GRAPHRAG_API_KEY="your-key" submission
```

### 4. Optional env at run

| Variable           | Description                            | Default                |
|--------------------|----------------------------------------|------------------------|
| `GRAPHRAG_API_KEY` | API key for completion model (LLM)     | *(required)*           |
| `OPENAI_BASE_URL`  | LLM API base URL                       | `https://hub.qazcode.ai` |

Example with custom base URL:

```bash
docker run -p 8080:8080 -e GRAPHRAG_API_KEY="..." -e OPENAI_BASE_URL="https://api.openai.com/v1" submission
```

### 5. Test

```bash
curl -X POST http://localhost:8080/diagnose -H "Content-Type: application/json" -d '{"symptoms": "headache, fever"}'
```

Or open **http://localhost:8080/docs** for Swagger UI.

---

## Challenge Overview

Participants will build an MVP product where users input symptoms as free text and receive:

- **Top-N probable diagnoses** ranked by likelihood
- **ICD-10 codes** for each diagnosis
- **Brief clinical explanations** based on official Kazakhstan protocols

The solution **must** run **using GPT-OSS** — no external LLM API calls allowed. Refer to `notebooks/llm_api_examples.ipynb`

---
## Data Sources

### Kazakhstan Clinical Protocols
Official clinical guidelines serving as the primary knowledge base for diagnoses and diagnostic criteria.[[corpus.zip](https://github.com/user-attachments/files/25365231/corpus.zip)]

Data Format

```json
{"protocol_id": "p_d57148b2d4", "source_file": "HELLP-СИНДРОМ.pdf", "title": "Одобрен", "icd_codes": ["O00", "O99"], "text": "Одобрен Объединенной комиссией по качеству медицинских услуг Министерства здравоохранения Республики Казахстан от «13» января 2023 года Протокол №177 КЛИНИЧЕСКИЙ ПРОТОКОЛ ДИАГНОСТИКИ И ЛЕЧЕНИЯ HELLP-СИНДРОМ I. ВВОДНАЯ ЧАСТЬ 1.1 Код(ы) МКБ-10: Код МКБ-10 O00-O99 Беременность, роды и послеродовой период О14.2 HELLP-синдром 1.2 Дата разработки/пересмотра протокола: 2022 год. ..."}

```

---

## Evaluation

### Metrics
- **Primary metrics:** Accuracy@1, Recall@3, Latency
- **Test set:**: Dataset with cases (`data/test_set`), use `query` and `gt` fields.
- **Holdout set:** Private test cases (not included in this repository)

### Product Evaluation
Working demo interface: user inputs symptoms → system returns diagnoses with ICD-10 codes;

---
## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/dair-mus/qazcode-nu.git
cd qazcode-nu
```

### 2. Set up the environment
We kindly ask you to use `uv` as your Python package manager.

Make sure that `uv` is installed. Refer to [uv documentation](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv venv
source .venv/bin/activate
uv sync
```

### 3. Running validation

Pick one of the three RAG servers (see **Three RAG implementations** above), then run the evaluator against it.

**Option A — Direct RAG** (TF-IDF + DiReCT-style LLM):

```bash
export OPENAI_API_KEY="your-key"
export CORPUS_DIR=./data/test_set
export DIAGNOSIS_MODEL=gpt-4o
uv run uvicorn src_direct.direct_rag_server:app --host 127.0.0.1 --port 8000
```

**Option B — Advanced RAG** (BM25 + FAISS, protocol-first ranking; requires prebuilt `artifacts/`):

```bash
# First: python -m src_advanced_rag.preprocess --corpus <corpus.jsonl>
uv run uvicorn src_advanced_rag.server:app --host 127.0.0.1 --port 8000
```

**Option C — GraphRAG** (submission; requires prebuilt `output/` index):

```bash
export GRAPHRAG_API_KEY="your-key"
# First: uv run graphrag index --config src_graph_rag/settings.yaml
uv run uvicorn src_graph_rag.diagnose_server:app --host 127.0.0.1 --port 8000
```

**Mock server** (random ICD-10, no LLM):

```bash
uv run uvicorn src_direct.mock_server:app --host 127.0.0.1 --port 8000
```

Then run the validation pipeline in a separate terminal:
```bash
uv run python evaluate.py -e http://127.0.0.1:8000/diagnose -d ./data/test_set -n <your_team_name>
```
`-e`: endpoint (POST request) that will accept the symptoms

`-d`: path to the directory with protocols

`-n`: name of your team (please avoid special symbols)

By default, the evalutaion results will be output to `data/evals`.

### Docker
The **Dockerfile** builds the **GraphRAG** server (`src_graph_rag`), which serves on **port 8080**. See **How to use the Dockerfile (GraphRAG submission)** at the top of this README for build/run steps. For a mock server locally (no Docker), use `src_direct.mock_server:app` or `src_advanced_rag.mock_server:app` as above. 

### Submission Checklist

- [ ] Everything packed into a single project (application, models, vector DB, indexes)
- [ ] Image builds successfully: `docker build -t submission .`
- [ ] Container starts and serves on port 8080: `docker run -p 8080:8080 submission`
- [ ] Web UI accepts free-text symptoms input
- [ ] Endpoint for POST requests accepts free-text symptoms
- [ ] Returns top-N diagnoses with ICD-10 codes
- [ ] No external network calls during inference
- [ ] README with build and run instructions

### How to Submit

1. Provide a Git repository with `Dockerfile`
2. Submit the link via [submission form](https://docs.google.com/forms/d/e/1FAIpQLSe8qg6LsgJroHf9u_MVDBLPqD8S_W6MrphAteRqG-c4cqhQDw/viewform)
3. We will pull, build, and run your container on the private holdout set
---

### Repo structure
- `data/evals`: evaluation results directory
- `data/examples/response.json`: example of a JSON response from your project endpoint
- `data/test_set`: use these to evaluate your solution (protocol JSONs with `text`, `icd_codes`, `query`, `gt`).
- `data/graphrag_input/`: input documents for GraphRAG index (used by `src_graph_rag`).
- `notebooks/llm_api_examples.ipynb`: shows how to make a request to GPT-OSS.
- **`src_direct/`** — Direct RAG (DiReCT-style): `direct_rag_server.py`, `rag.py`, `prompts.py`, `llm_client.py`, `mock_server.py`
- **`src_advanced_rag/`** — Hybrid RAG: `server.py`, `retrieval.py`, `icd_ranker.py`, `fusion.py`, `preprocess.py`, `llm_client.py`, `mock_server.py`; prebuilt index in `artifacts/`
- **`src_graph_rag/`** — GraphRAG (submission): `diagnose_server.py`, `settings.yaml`, `local_embedding.py`, `export_graphrag_documents.py`; index in `output/`, prompts in `prompts/`
- `evaluate.py`: runs the given dataset through the provided endpoint
- `pyproject.toml`, `uv.lock`: project dependencies
- `Dockerfile`: builds the GraphRAG server image (port 8080)
