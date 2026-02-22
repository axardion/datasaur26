import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

log = logging.getLogger(__name__)

_project_root = Path(__file__).resolve().parent.parent
_env_path = _project_root / ".env"
_env_loaded_from: str
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path, override=True)
    _env_loaded_from = str(_env_path)
else:
    _cwd_env = Path.cwd() / ".env"
    load_dotenv(dotenv_path=_cwd_env, override=True)
    _env_loaded_from = str(_cwd_env) if _cwd_env.exists() else "(none found)"
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src_direct.rag import DirectRAGIndex, ProtocolChunk, load_protocols
from src_direct.llm_client import _mask_api_key, diagnose_with_rag


DEFAULT_CORPUS_DIR = Path("data/corpus")
FALLBACK_CORPUS_DIR = Path("data/test_set")
CORPUS_DIR = Path(os.environ.get("CORPUS_DIR", str(DEFAULT_CORPUS_DIR)))
TOP_K_RETRIEVE = 12
TOP_N_DIAGNOSES = 5

_rag_index: Optional[DirectRAGIndex] = None
_corpus_dir_used: Optional[Path] = None


def _count_protocols_in_dir(path: Path) -> int:
    return len(load_protocols(path))


def get_rag_index() -> DirectRAGIndex:
    global _rag_index, _corpus_dir_used
    if _rag_index is None:
        corpus_to_use = CORPUS_DIR
        if not corpus_to_use.exists():
            if FALLBACK_CORPUS_DIR.exists():
                print(f"Warning: CORPUS_DIR {corpus_to_use} not found. Using fallback: {FALLBACK_CORPUS_DIR}")
                print("  → Unpack corpus.zip into data/corpus for the real knowledge base.")
                corpus_to_use = FALLBACK_CORPUS_DIR
            else:
                raise FileNotFoundError(
                    f"Corpus directory not found: {corpus_to_use}. "
                    f"Unpack corpus.zip into data/corpus or set CORPUS_DIR."
                )
        n = _count_protocols_in_dir(corpus_to_use)
        if n == 0:
            if corpus_to_use == CORPUS_DIR and FALLBACK_CORPUS_DIR.exists():
                print(f"Warning: No protocol JSONs in {corpus_to_use}. Using fallback: {FALLBACK_CORPUS_DIR}")
                corpus_to_use = FALLBACK_CORPUS_DIR
                n = _count_protocols_in_dir(corpus_to_use)
            if n == 0:
                raise FileNotFoundError(
                    f"No protocol JSONs (with 'text' and 'icd_codes') in {corpus_to_use}. "
                    "Unpack corpus.zip into data/corpus."
                )
        _corpus_dir_used = corpus_to_use
        _rag_index = DirectRAGIndex(corpus_to_use)
    return _rag_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("DIAGNOSIS_MODEL", "gpt-4o")
    print("\n🏥 DIRECT RAG Diagnostic Server (DiReCT-style)")
    print("=" * 50)
    print(f".env:    {_env_loaded_from}")
    print(f"api_key: {_mask_api_key(api_key)}")
    print(f"model:   {model}")
    print(f"Corpus:  {CORPUS_DIR}")
    try:
        idx = get_rag_index()
        used = _corpus_dir_used or CORPUS_DIR
        print(f"RAG corpus: {used}")
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


def diversify_chunks(
    chunks: list[ProtocolChunk],
    max_per_protocol: int = 4,
) -> list[ProtocolChunk]:
    if not chunks:
        return []
    seen: dict[str, int] = {}
    out: list[ProtocolChunk] = []
    for c in chunks:
        n = seen.get(c.protocol_id, 0)
        if n < max_per_protocol:
            seen[c.protocol_id] = n + 1
            out.append(c)
    return out


def chunks_to_premises(
    chunks: list[ProtocolChunk], max_total_chars: int = 80_000
) -> list[str]:
    premises: list[str] = []
    total = 0
    for c in chunks:
        if total >= max_total_chars:
            break
        premises.append(c.text)
        total += len(c.text)
    return premises


@app.post("/diagnose", response_model=DiagnoseResponse)
async def handle_diagnose(request: DiagnoseRequest) -> DiagnoseResponse:
    symptoms = (request.symptoms or "").strip()
    if not symptoms:
        return DiagnoseResponse(diagnoses=[])

    rag = get_rag_index()
    chunks = rag.retrieve(symptoms, top_k=TOP_K_RETRIEVE)
    chunks = diversify_chunks(chunks, max_per_protocol=4)
    premises = chunks_to_premises(chunks)
    if not premises:
        raise HTTPException(status_code=503, detail="No protocol premises retrieved")

    try:
        raw = diagnose_with_rag(symptoms, premises, top_n=TOP_N_DIAGNOSES)
    except Exception as e:
        log.exception("LLM diagnosis failed")
        raise HTTPException(
            status_code=502,
            detail=f"LLM diagnosis failed: {type(e).__name__}: {e}",
        ) from e

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
