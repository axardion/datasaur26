"""
LLM client for https://hub.qazcode.ai. Model configurable via env.
Used for: (1) optional query expansion, (2) controlled rerank.
API key read from OPENAI_API_KEY (e.g. from .env).
"""

import json
import os
import re

from openai import OpenAI

BASE_URL = "https://hub.qazcode.ai"
MODEL = os.getenv("DIAGNOSIS_MODEL", "oss-120b")
RERANK_MAX_CANDIDATES = 80
RERANK_SNIPPET_CHARS = 350
RERANK_MAX_OUT = 5
RERANK_TEMP = 0.1


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key: set OPENAI_API_KEY in .env or environment")
    return OpenAI(
        base_url=BASE_URL,
        api_key=api_key,
    )


def expand_queries(symptoms: str, client: OpenAI | None = None) -> list[str]:
    """
    Multi-query expansion: 3–5 Russian variants + original. On failure returns [symptoms].
    """
    try:
        client = client or get_client()
        prompt = f"""Даны жалобы пациента (симптомы) на русском языке. Сгенерируй 5 альтернативных формулировок для поиска по медицинским протоколам:
- синонимы и бытовые эквиваленты медицинских терминов
- аббревиатуры и полные названия
- перефразирование тех же симптомов

Исходные жалобы:
{symptoms[:2000]}

Верни ТОЛЬКО JSON-массив из 5 строк, без пояснений. Пример: ["запрос 1", "запрос 2", ...]"""
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800,
        )
        text = (resp.choices[0].message.content or "").strip()
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                queries = [str(x).strip() for x in arr if x][:5]
                if queries:
                    return [symptoms] + queries
        except json.JSONDecodeError:
            pass
        match = re.search(r"\[[\s\S]*?\]", text)
        if match:
            try:
                arr = json.loads(match.group(0))
                queries = [str(x).strip() for x in arr if x][:5]
                if queries:
                    return [symptoms] + queries
            except json.JSONDecodeError:
                pass
    except Exception as e:
        print("LLM expand failed:", e)
    return [symptoms]


def rerank_diagnoses(
    symptoms: str,
    candidates: list[tuple[str, float, str, str]],
    allowed_icd: set[str],
    client: OpenAI | None = None,
) -> list[tuple[str, float]]:
    """
    Rerank up to 80 candidates. Each: icd10_code, short title, evidence <= 350 chars.
    Strict JSON list up to 5 items. Temp 0.0–0.1. On failure return pre-LLM top 3.
    """
    try:
        client = client or get_client()
        allowed_list = sorted(allowed_icd)
        lines = []
        for code, _sc, title, snippet in candidates[:RERANK_MAX_CANDIDATES]:
            snip = (snippet or "").replace("\n", " ")[:RERANK_SNIPPET_CHARS]
            lines.append(f"- {code}: {title[:120]} | {snip}")
        context = "\n".join(lines)

        prompt = f"""По жалобам пациента выбери до 5 наиболее вероятных диагнозов МКБ-10 из списка кандидатов. Разрешены ТОЛЬКО коды из списка ниже.

Жалобы:
{symptoms[:1500]}

Кандидаты (код, описание, фрагмент):
{context}

Разрешённые коды (используй только их): {", ".join(allowed_list[:150])}

Ответь СТРОГО JSON-массив до 5 элементов: [{{"icd10_code": "...", "confidence": 0.xx}}, ...] Без пояснений."""

        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=RERANK_TEMP,
            max_tokens=400,
        )
        text = (resp.choices[0].message.content or "").strip()
        parsed: list[tuple[str, float]] = []
        try:
            match = re.search(r"\[[\s\S]*\]", text)
            if match:
                arr = json.loads(match.group(0))
                for item in arr:
                    if isinstance(item, dict):
                        code = (item.get("icd10_code") or item.get("icd_code") or "").strip()
                        conf = float(item.get("confidence", 0.5))
                        if code and code in allowed_icd:
                            parsed.append((code, conf))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        seen: set[str] = set()
        unique: list[tuple[str, float]] = []
        for code, conf in parsed:
            if code not in seen:
                seen.add(code)
                unique.append((code, conf))
        for code, _sc, _t, _s in candidates:
            if code in allowed_icd and code not in seen:
                unique.append((code, 0.3))
                seen.add(code)
            if len(unique) >= RERANK_MAX_OUT:
                break
        if unique:
            return unique[:RERANK_MAX_OUT]
    except Exception as e:
        print("LLM rerank failed:", e)
    # Fallback: pre-LLM ranking top 3
    out: list[tuple[str, float]] = []
    for code, _sc, _t, _s in candidates:
        if code in allowed_icd and code not in {c for c, _ in out}:
            out.append((code, 0.5))
        if len(out) >= 5:
            break
    return out[:5]
