# Datasaur 2026 | Qazcode Challenge

## Medical Diagnosis Assistant: Symptoms → ICD-10

An AI-powered clinical decision support system that converts patient symptoms into structured diagnoses with ICD-10 codes, built on Kazakhstan clinical protocols.

---

## How to use the Dockerfile for Juri

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

**DIRECT RAG (DiReCT-style)** — retrieval-augmented diagnosis using protocol premises:

```bash
# Set GPT-OSS API key (hub.qazcode.ai)
export OPENAI_API_KEY="your-key"

# Optional: corpus dir (default: data/test_set), model name
export CORPUS_DIR=./data/test_set
export DIAGNOSIS_MODEL=gpt-4o

uv run uvicorn src.direct_rag_server:app --host 127.0.0.1 --port 8000
```

This server loads Kazakhstan protocols from `data/test_set`, indexes them with TF-IDF, and for each symptom query retrieves relevant protocol excerpts (premises). The LLM is prompted DiReCT-style to output top-N diagnoses with ICD-10 codes and explanations.

**Mock server** (random ICD-10, no LLM):

```bash
uv run uvicorn src.mock_server:app --host 127.0.0.1 --port 8000
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
We prepared a Dockerfile to run our mock server example.
```bash
docker build -t mock-server .
docker run -p 8000:8000 mock-server
```
Then run the validation as shown above.

Feel free to use the mock-server FastAPI template and Dockerfile structure to build your own project around.

Remember to adjust the CMD in Dockerfile for your real Python server instead of `src.mock_server:app` before submission. 

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
- `notebooks/llm_api_examples.ipynb`: shows how to make a request to GPT-OSS.
- `src/`: solution source code
  - `direct_rag_server.py`: **DIRECT RAG** server (DiReCT-style: RAG + LLM diagnosis)
  - `rag.py`: corpus loading, chunking, TF-IDF retrieval
  - `prompts.py`: DiReCT-style prompts (premises + symptoms → diagnoses)
  - `llm_client.py`: GPT-OSS client for diagnosis
  - `mock_server.py`: example mock endpoint (random ICD-10)
- `evaluate.py`: runs the given dataset through the provided endpoint.
- `pyproject.toml`: describes dependencies of the project.
- `uv.lock`: stores the exact dependency versions, autogenerated by uv.
- `Dockerfile`: contains build instructions for a Docker image.
