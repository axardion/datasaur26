"""
Diagnosis server that uses GraphRAG (local search) to return top-3 ICD-10 diagnoses.

Same contract as mock_server: POST /diagnose with {"symptoms": "..."}.
Run: uv run uvicorn src.diagnose_server:app --host 127.0.0.1 --port 8000
"""
import src_graph_rag.local_embedding  # noqa: F401 — register local embedder so config type: local works

import asyncio
import json
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Repo root and config root (settings.yaml lives in src/)
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = Path(__file__).resolve().parent  # src/ — directory containing settings.yaml
STATIC_DIR = REPO_ROOT / "static"

def _preload_models(config: Any) -> None:
    """Load embedding (and completion) models at startup so first /diagnose doesn't trigger HF download."""
    from src_graph_rag.local_embedding import _get_model as get_embedding_model

    emb_id = getattr(config.local_search, "embedding_model_id", None) or "default_embedding_model"
    emb_models = getattr(config, "embedding_models", {}) or {}
    emb_cfg = emb_models.get(emb_id)
    emb_model_name = getattr(emb_cfg, "model", None) if emb_cfg else None
    if not emb_model_name:
        emb_model_name = "lokeshch19/ModernPubMedBERT"
    get_embedding_model(emb_model_name)


# Response type passed to GraphRAG so the LLM returns structured JSON
DIAGNOSIS_RESPONSE_TYPE = """A JSON object with a single key "diagnoses" (array of exactly 3 items).
Each item must have: "rank" (1, 2, or 3), "diagnosis" (short name in Russian or English), "icd10_code" (ICD-10 code, e.g. J20.9), "explanation" (1-2 sentences).
Return only valid JSON, no markdown code fences or extra text."""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load GraphRAG config, index data, and ML models once at startup (no lazy load on first request)."""
    from graphrag_storage import create_storage
    from graphrag_storage.tables.table_provider_factory import create_table_provider
    from graphrag.config.load_config import load_config
    from graphrag.data_model.data_reader import DataReader

    print("\n🔄 Loading GraphRAG index...")
    config = load_config(CONFIG_ROOT)
    storage = create_storage(config.output_storage)
    table_provider = create_table_provider(config.table_provider, storage=storage)
    reader = DataReader(table_provider)

    app.state.config = config
    app.state.entities = await reader.entities()
    app.state.communities = await reader.communities()
    app.state.community_reports = await reader.community_reports()
    app.state.text_units = await reader.text_units()
    app.state.relationships = await reader.relationships()
    try:
        app.state.covariates = await reader.covariates()
    except Exception:
        app.state.covariates = None

    print("✅ GraphRAG index loaded.")

    # Preload embedding model so first /diagnose doesn't trigger HF download mid-request
    print("🔄 Preloading embedding model...")
    try:
        await asyncio.to_thread(_preload_models, config)
        print("✅ Models ready.")
    except Exception as e:
        print(f"⚠️ Model preload failed (will load on first request): {e}")
    print("\n🏥 Diagnosis Server (GraphRAG)")
    print("=" * 40)
    print("Endpoint: POST /diagnose")
    print('Body:     {"symptoms": "..."}')
    print("Docs:     /docs")
    print("=" * 40)
    print("\nPress Ctrl+C to stop\n")
    yield


app = FastAPI(title="Diagnosis Server (GraphRAG)", lifespan=lifespan)

# Serve UI (medical-style frontend)
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")


class DiagnoseRequest(BaseModel):
    symptoms: Optional[str] = ""


class Diagnosis(BaseModel):
    rank: int
    diagnosis: str
    icd10_code: str
    explanation: str


class DiagnoseResponse(BaseModel):
    diagnoses: list[Diagnosis]


def _parse_diagnoses_from_response(raw: str) -> list[dict[str, Any]]:
    """Extract diagnoses array from LLM response (JSON or markdown-wrapped JSON)."""
    raw = raw.strip()
    # Remove optional markdown code block
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", raw)
    if m:
        raw = m.group(1)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\"diagnoses\"[\s\S]*\}", raw)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                return []
        else:
            return []
    diagnoses = data.get("diagnoses") if isinstance(data, dict) else None
    if not isinstance(diagnoses, list) or len(diagnoses) == 0:
        return []
    out = []
    for i, d in enumerate(diagnoses[:3]):
        if not isinstance(d, dict):
            continue
        rank = d.get("rank", i + 1)
        diagnosis = (
            d.get("diagnosis")
            or d.get("diagnosis_name")
            or d.get("name")
            or "—"
        )
        icd10_code = d.get("icd10_code") or d.get("icd_code") or ""
        explanation = d.get("explanation") or ""
        if not explanation and d.get("matching_symptoms"):
            explanation = "Matching: " + ", ".join(d.get("matching_symptoms", []))
        if not explanation and d.get("confidence_score"):
            explanation = f"Confidence: {d.get('confidence_score')}"
        out.append({
            "rank": rank if isinstance(rank, int) else i + 1,
            "diagnosis": str(diagnosis),
            "icd10_code": str(icd10_code).strip(),
            "explanation": str(explanation),
        })
    return out


@app.post("/diagnose", response_model=DiagnoseResponse)
async def handle_diagnose(request: DiagnoseRequest) -> DiagnoseResponse:
    import graphrag.api as api

    symptoms = request.symptoms or ""
    if not symptoms.strip():
        return DiagnoseResponse(
            diagnoses=[
                Diagnosis(rank=1, diagnosis="—", icd10_code="", explanation="No symptoms provided."),
                Diagnosis(rank=2, diagnosis="—", icd10_code="", explanation=""),
                Diagnosis(rank=3, diagnosis="—", icd10_code="", explanation=""),
            ]
        )

    response_text, _ = await api.local_search(
        config=app.state.config,
        entities=app.state.entities,
        communities=app.state.communities,
        community_reports=app.state.community_reports,
        text_units=app.state.text_units,
        relationships=app.state.relationships,
        covariates=app.state.covariates,
        community_level=2,
        response_type=DIAGNOSIS_RESPONSE_TYPE,
        query=symptoms,
        verbose=False,
    )

    raw = response_text if isinstance(response_text, str) else json.dumps(response_text)
    parsed = _parse_diagnoses_from_response(raw)

    if len(parsed) < 3:
        for i in range(len(parsed), 3):
            parsed.append({
                "rank": i + 1,
                "diagnosis": "—",
                "icd10_code": "",
                "explanation": "",
            })

    diagnoses = [
        Diagnosis(
            rank=p.get("rank", i + 1),
            diagnosis=p.get("diagnosis", "—"),
            icd10_code=p.get("icd10_code", ""),
            explanation=p.get("explanation", ""),
        )
        for i, p in enumerate(parsed[:3])
    ]
    return DiagnoseResponse(diagnoses=diagnoses)
