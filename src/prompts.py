"""
DiReCT-style prompts for diagnostic reasoning with retrieved premises.

DiReCT uses:
- gen_disease_diagnose(note, disease_options) for initial disease category
- gen_reasoning_initial / gen_reasoning_advanced with premise from knowledge graph

We adapt: single prompt that takes retrieved protocol excerpts (premises) and
patient symptoms, and outputs top-N diagnoses with ICD-10 codes and explanations.
"""


def build_diagnosis_prompt(symptoms: str, premises: list[str], top_n: int = 5) -> str:
    """Build prompt for diagnosis from symptoms and retrieved protocol premises (DiReCT-style)."""
    premises_block = "\n\n---\n\n".join(
        f"[Excerpt {i+1}]\n{p}" for i, p in enumerate(premises[:8], 1)
    )
    return f"""Suppose you are an expert clinician using official Kazakhstan clinical protocols. Think step by step.

You are given:
1) Excerpts from clinical protocols (diagnostic criteria, symptoms, ICD-10 codes).
2) A patient's description of their symptoms.

Your task: Propose the top-{top_n} most probable diagnoses that match the symptoms, based on the protocol excerpts. For each diagnosis provide:
- The exact ICD-10 code as it appears in the protocols (e.g. F18.8, J20.9).
- A short clinical explanation referring to the protocol criteria.

Format your response as a JSON array only, no other text:
[
  {{"rank": 1, "diagnosis": "Название диагноза", "icd10_code": "X00.0", "explanation": "..."}},
  {{"rank": 2, "diagnosis": "...", "icd10_code": "...", "explanation": "..."}},
  ...
]

Protocol excerpts (premises):
{premises_block}

Patient symptoms:
{symptoms}

Respond with only the JSON array."""
