"""
LLM client for GPT-OSS (OpenAI-compatible API at hub.qazcode.ai).

Used for DIRECT RAG diagnosis: no external LLM API except the designated GPT-OSS endpoint.
"""

import os
import json
import re
from openai import OpenAI


def get_client() -> OpenAI:
    """Build OpenAI client for GPT-OSS (hub.qazcode.ai) or default OpenAI."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    return OpenAI(api_key=api_key, base_url=base_url)


def diagnose_with_rag(
    symptoms: str,
    premises: list[str],
    *,
    model: str | None = None,
    top_n: int = 5,
) -> list[dict]:
    """
    Call LLM with DiReCT-style prompt (symptoms + retrieved premises), return list of diagnoses.

    Each item: {"rank": int, "diagnosis": str, "icd10_code": str, "explanation": str}
    """
    from src.prompts import build_diagnosis_prompt

    model = model or os.environ.get("DIAGNOSIS_MODEL", "gpt-4o")
    prompt = build_diagnosis_prompt(symptoms, premises, top_n=top_n)

    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    content = (response.choices[0].message.content or "").strip()

    # Extract JSON array (model might wrap in markdown)
    json_match = re.search(r"\[[\s\S]*\]", content)
    if json_match:
        content = json_match.group(0)
    try:
        out = json.loads(content)
        if isinstance(out, list):
            # Normalize to have rank, diagnosis, icd10_code, explanation
            result = []
            for i, item in enumerate(out[: top_n], 1):
                if isinstance(item, dict):
                    result.append({
                        "rank": item.get("rank", i),
                        "diagnosis": str(item.get("diagnosis", "")),
                        "icd10_code": str(item.get("icd10_code", "")),
                        "explanation": str(item.get("explanation", "")),
                    })
            return result
        return []
    except json.JSONDecodeError:
        return []
