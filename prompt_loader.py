from pathlib import Path

_PROMPT_PATH = Path(__file__).resolve().parent / "normalized_prompt.txt"
_TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")

TARGET_RESPONSE_FORMAT = """A JSON object with a single key "diagnoses" (array of up to 3 items).
Each item must have: "rank" (1, 2, or 3), "protocol_id" (from context), "diagnosis_name" (from context), "icd10_code" (exact from context), "confidence_score" ("High"/"Medium"/"Low"), "matching_symptoms" (array of strings).
Include a "reasoning" key with a brief justification. Output only valid JSON, no markdown."""


def build_diagnosis_prompt(context_data: str, user_query: str) -> str:
    return _TEMPLATE.replace("{context_data}", context_data).replace("{user_query}", user_query)
