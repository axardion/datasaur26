import json
import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

_project_root = Path(__file__).resolve().parent.parent
_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path, override=True)
else:
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=True)

log = logging.getLogger(__name__)


def _mask_api_key(key: str) -> str:
    if not key:
        return "(not set)"
    if len(key) <= 12:
        return "***"
    return f"{key[:7]}...{key[-4:]}"


def get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    if not base_url:
        base_url = "https://hub.qazcode.ai"
    return OpenAI(api_key=api_key, base_url=base_url)


def diagnose_with_rag(
    symptoms: str,
    premises: list[str],
    *,
    model: str | None = None,
    top_n: int = 5,
) -> list[dict]:

    from src_direct.prompts import build_diagnosis_prompt

    model = model or os.environ.get("DIAGNOSIS_MODEL", "gpt-4o")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    log.info("LLM config: api_key=%s model=%s", _mask_api_key(api_key), model)

    prompt = build_diagnosis_prompt(symptoms, premises, top_n=top_n)
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise ValueError("LLM returned empty response")

    content = re.sub(r"^```(?:json)?\s*\n?", "", content)
    content = re.sub(r"\n?```\s*$", "", content)
    content = content.strip()

    obj_match = re.search(r"\{[\s\S]*\}", content)
    if not obj_match:
        raise ValueError(
            f"No JSON object in LLM response (first 200 chars: {content[:200]!r})"
        )
    try:
        data = json.loads(obj_match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON from LLM: {e}. First 300 chars: {content[:300]!r}"
        ) from e
    if not isinstance(data, dict):
        return []
    diagnoses = data.get("diagnoses")
    if not isinstance(diagnoses, list):
        return []
    result = []
    reasoning = (data.get("reasoning") or "").strip()
    for i, item in enumerate(diagnoses[:top_n], 1):
        if not isinstance(item, dict):
            continue
        diagnosis_name = item.get("diagnosis_name") or item.get("diagnosis") or ""
        matching = item.get("matching_symptoms") or []
        expl = item.get("explanation") or ""
        if not expl and matching:
            expl = " ".join(str(x) for x in matching)
        if not expl and reasoning:
            expl = reasoning
        result.append({
            "rank": item.get("rank", i),
            "diagnosis": str(diagnosis_name),
            "icd10_code": str(item.get("icd10_code", "")),
            "explanation": str(expl),
        })
    return result
