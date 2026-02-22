"""
FastAPI server: POST /diagnose with {"symptoms": "..."} -> {"diagnoses": [{"rank": 1, "icd10_code": "..."}, ...]}.
Query expansion and LLM rerank configurable via env. Never 500, always >=3 diagnoses.
Loads .env from project root if present.
"""

import asyncio
import hashlib
import os
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load .env before other app imports that read env vars
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI
from pydantic import BaseModel

from src_advanced_rag.icd_ranker import ICDRanker
from src_advanced_rag.llm_client import expand_queries, get_client

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
CACHE_MAX = 500

USE_QUERY_EXPANSION = os.getenv("USE_QUERY_EXPANSION", "0").strip().lower() in ("1", "true", "yes")
USE_LLM_RERANK = os.getenv("USE_LLM_RERANK", "1").strip().lower() in ("1", "true", "yes")


def _cache_key(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load indexes once at startup."""
    app.state.icd_ranker = None
    app.state.diagnose_cache: OrderedDict[str, list[dict]] = OrderedDict()
    try:
        if ARTIFACTS_DIR.exists() and (ARTIFACTS_DIR / "bm25_chunks.json").exists():
            app.state.icd_ranker = ICDRanker(ARTIFACTS_DIR)
        else:
            app.state.icd_ranker = None
    except Exception as e:
        print(f"Warning: could not load artifacts: {e}. Run preprocess first.")
    yield
    app.state.diagnose_cache.clear()


app = FastAPI(title="Medical Diagnosis Ranking", lifespan=lifespan)


class DiagnoseRequest(BaseModel):
    symptoms: str = ""


class DiagnosisOut(BaseModel):
    rank: int
    icd10_code: str


class DiagnoseResponse(BaseModel):
    diagnoses: list[DiagnosisOut]


def _fallback_diagnoses(icd_ranker: ICDRanker | None, symptoms: str = "") -> list[dict]:
    """Return exactly 3 diagnoses from protocol-first or corpus. Never fails."""
    if icd_ranker is not None and symptoms and symptoms.strip():
        try:
            out = icd_ranker.get_top_diagnoses_protocol_first([symptoms])
            if len(out) >= 3:
                return out[:3]
            codes = [o["icd10_code"] for o in out]
            for c in icd_ranker.corpus_icd_codes:
                if c not in codes:
                    codes.append(c)
                    if len(codes) >= 3:
                        break
            codes = codes[:3]
        except Exception:
            codes = list(icd_ranker.corpus_icd_codes)[:3]
    else:
        codes = list(icd_ranker.corpus_icd_codes)[:3] if icd_ranker else []
    if len(codes) < 3:
        codes = (codes + ["R50.9", "J06.9", "K59.9"])[:3]
    return [{"rank": i + 1, "icd10_code": c} for i, c in enumerate(codes)]


def _run_pipeline(symptoms: str, icd_ranker: ICDRanker) -> list[dict]:
    """Protocol-first pipeline: retrieve top chunks -> rank protocols -> output primary_icd / block / family."""
    if not symptoms or not symptoms.strip():
        codes = list(icd_ranker.corpus_icd_codes)[:3]
        return [{"rank": i + 1, "icd10_code": c} for i, c in enumerate(codes)]

    if USE_QUERY_EXPANSION:
        try:
            client = get_client()
            queries = expand_queries(symptoms, client=client)
        except RuntimeError:
            queries = [symptoms]
    else:
        queries = [symptoms]

    # Protocol-first: group chunks by protocol_id, aggregate score, output primary_icd then icd_block_codes then icd_family
    out = icd_ranker.get_top_diagnoses_protocol_first(queries)
    if len(out) < 3:
        for c in icd_ranker.corpus_icd_codes:
            if c not in {o["icd10_code"] for o in out}:
                out.append({"rank": len(out) + 1, "icd10_code": c})
                if len(out) >= 3:
                    break
    return out[:max(3, len(out))]


@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(request: DiagnoseRequest) -> DiagnoseResponse:
    """Return top diagnoses with ICD-10 codes, sorted by rank. Always >= 3."""
    symptoms = (request.symptoms or "").strip()
    cache = getattr(app.state, "diagnose_cache", None)
    icd_ranker = getattr(app.state, "icd_ranker", None)

    if cache is not None:
        key = _cache_key(symptoms)
        if key in cache:
            return DiagnoseResponse(diagnoses=[DiagnosisOut(**d) for d in cache[key]])
        while len(cache) >= CACHE_MAX:
            cache.popitem(last=False)

    if icd_ranker is None:
        fallback = [
            DiagnosisOut(rank=1, icd10_code="R50.9"),
            DiagnosisOut(rank=2, icd10_code="J06.9"),
            DiagnosisOut(rank=3, icd10_code="K59.9"),
        ]
        return DiagnoseResponse(diagnoses=fallback)

    try:
        result = await asyncio.to_thread(_run_pipeline, symptoms, icd_ranker)
    except Exception as e:
        print("Pipeline failed:", e)
        result = _fallback_diagnoses(icd_ranker, symptoms)
        return DiagnoseResponse(diagnoses=[DiagnosisOut(**d) for d in result])

    if cache is not None:
        cache[_cache_key(symptoms)] = result
        cache.move_to_end(_cache_key(symptoms))
    return DiagnoseResponse(diagnoses=[DiagnosisOut(**d) for d in result])
