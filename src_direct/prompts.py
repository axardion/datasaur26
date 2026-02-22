from prompt_loader import build_diagnosis_prompt as _build_prompt


def build_diagnosis_prompt(symptoms: str, premises: list[str], top_n: int = 5) -> str:
    context_data = "\n\n---\n\n".join(
        f"[Excerpt {i+1}]\n{p}" for i, p in enumerate(premises, 1)
    )
    return _build_prompt(context_data=context_data, user_query=symptoms)
