def build_diagnosis_prompt(symptoms: str, premises: list[str], top_n: int = 5) -> str:
    premises_block = "\n\n---\n\n".join(
        f"[Excerpt {i+1}]\n{p}" for i, p in enumerate(premises, 1)
    )
    return f"""Suppose you are an expert clinician using official Kazakhstan clinical protocols. Think step by step.

You are given:
1) Golden standards (excerpts from clinical protocols): diagnostic criteria, symptoms, signs, and ICD-10 codes.
2) A patient's description of their symptoms.

Your task: Based only on the golden standards above, propose the top-{top_n} most probable diagnoses that match the symptoms. For each diagnosis provide:
- An ICD-10 code that appears in the golden standards (use the exact form used there, e.g. S22.0 or S22.0XXA as written).
- A short Reason explaining how the protocol criteria support this diagnosis (DiReCT-style).
- Use only ICD-10 codes that appear in the excerpts; do not invent codes.

Format your response as a JSON array only, no other text:
[
  {{"rank": 1, "diagnosis": "Название диагноза", "icd10_code": "X00.0", "explanation": "..."}},
  {{"rank": 2, "diagnosis": "...", "icd10_code": "...", "explanation": "..."}},
  ...
]

Golden standards (protocol excerpts):
{premises_block}

Patient symptoms:
{symptoms}

Respond with only the JSON array."""
