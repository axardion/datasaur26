"""
DIRECT RAG diagnostic server (DiReCT-style).

Uses retrieval-augmented generation:
1. Load Kazakhstan clinical protocols from corpus (e.g. data/test_set).
2. On each request: retrieve relevant protocol chunks (premises) by symptoms.
3. Call LLM with DiReCT-style prompt (premises + symptoms) → top-N diagnoses with ICD-10.

Run:
  uv run uvicorn src.direct_rag_server:app --host 127.0.0.1 --port 8000

Environment:
  OPENAI_API_KEY       - API key for GPT-OSS (hub.qazcode.ai)
  OPENAI_BASE_URL      - Optional, default https://hub.qazcode.ai
  CORPUS_DIR           - Path to protocol JSONs (default: data/test_set)
  DIAGNOSIS_MODEL      - Model name (default: gpt-4o)
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from src.rag import DirectRAGIndex, ProtocolChunk
from src.llm_client import diagnose_with_rag


# Default corpus: test set (each file has text + icd_codes)
CORPUS_DIR = Path(os.environ.get("CORPUS_DIR", "data/test_set"))
TOP_K_RETRIEVE = 12
TOP_N_DIAGNOSES = 5

# Global RAG index (loaded at startup)
_rag_index: Optional[DirectRAGIndex] = None


def get_rag_index() -> DirectRAGIndex:
    global _rag_index
    if _rag_index is None:
        if not CORPUS_DIR.exists():
            raise FileNotFoundError(f"Corpus directory not found: {CORPUS_DIR}")
        _rag_index = DirectRAGIndex(CORPUS_DIR)
    return _rag_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🏥 DIRECT RAG Diagnostic Server (DiReCT-style)")
    print("=" * 50)
    print(f"Corpus: {CORPUS_DIR}")
    try:
        idx = get_rag_index()
        print(f"Loaded {len(idx.chunks)} chunks from {len(idx.protocols)} protocols")
    except Exception as e:
        print(f"Warning: RAG index failed to load: {e}")
    print("Endpoint: POST /diagnose")
    print("Docs: /docs")
    print("=" * 50)
    yield


app = FastAPI(title="DIRECT RAG Diagnostic Server", lifespan=lifespan)


class DiagnoseRequest(BaseModel):
    symptoms: Optional[str] = ""


class Diagnosis(BaseModel):
    rank: int
    diagnosis: str
    icd10_code: str
    explanation: str


class DiagnoseResponse(BaseModel):
    diagnoses: list[Diagnosis]


def chunks_to_premises(chunks: list[ProtocolChunk], max_chars: int = 6000) -> list[str]:
    """Turn retrieved chunks into premise strings, respecting total length."""
    premises: list[str] = []
    total = 0
    for c in chunks:
        if total + len(c.text) > max_chars:
            break
        premises.append(c.text[:2000])  # cap single chunk
        total += len(premises[-1])
    return premises


@app.post("/diagnose", response_model=DiagnoseResponse)
async def handle_diagnose(request: DiagnoseRequest) -> DiagnoseResponse:
    """Run DIRECT RAG: retrieve premises from protocols, then LLM diagnosis."""
    symptoms = (request.symptoms or "").strip()
    if not symptoms:
        return DiagnoseResponse(diagnoses=[])

    try:
        rag = get_rag_index()
        chunks = rag.retrieve(symptoms, top_k=TOP_K_RETRIEVE)
        premises = chunks_to_premises(chunks)
    except Exception:
        premises = []

    if not premises:
        # No corpus or retrieval failed: still call LLM without premises
        premises = ["No protocol excerpts available. Use your clinical knowledge and ICD-10."]

    raw = diagnose_with_rag(symptoms, premises, top_n=TOP_N_DIAGNOSES)
    diagnoses = [
        Diagnosis(
            rank=i,
            diagnosis=d.get("diagnosis", ""),
            icd10_code=d.get("icd10_code", ""),
            explanation=d.get("explanation", ""),
        )
        for i, d in enumerate(raw[:TOP_N_DIAGNOSES], 1)
    ]
    return DiagnoseResponse(diagnoses=diagnoses)
