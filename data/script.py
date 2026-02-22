import json
import re

INPUT_PATH = "protocols_corpus.jsonl"
OUTPUT_PATH = "protocols_only_complaints_anamnesis_filtered.jsonl"

# Если True — выкидываем документы, где не нашли ни "Жалобы", ни "Анамнез"
DROP_IF_NO_COMPLAINTS_ANAMNESIS = False

# --- 1) Начало разделов (ищем в любом месте текста) ---
START_RE = re.compile(r"(?i)\b(жалоб[ыа]|анамнез)\b\s*[:\-–]?\s*")

# --- 2) Маркеры других разделов (граница, где заканчивается текущий кусок) ---
END_MARKER_RE = re.compile(
    r"(?i)\b("
    r"жалоб[ыа]|анамнез|"
    r"диагностическ\w*\s+критери\w*|"
    r"диагностик\w*|"
    r"клиническ\w*\s+картин\w*|"
    r"объективн\w*\s+(?:осмотр|оценк\w*)|"
    r"физикальн\w*\s+обследовани\w*|"
    r"лабораторн\w*|"
    r"инструментальн\w*|"
    r"дифференциал\w*|"
    r"классификац\w*|"
    r"этиолог\w*|патогенез\w*|эпидемиолог\w*|"
    r"осложнен\w*|"
    r"лечен\w*|тактик\w*|"
    r"профилактик\w*|реабилитац\w*|"
    r"прогноз\w*|"
    r"ведение\s+пациент\w*|"
    r"маршрут\w*|"
    r"критери\w*\s+госпитализац\w*"
    r")\b"
)

def extract_complaints_anamnesis(text: str) -> str:
    """Достаёт все куски, начинающиеся с 'Жалобы' или 'Анамнез' до следующего маркера раздела."""
    if not text:
        return ""

    txt = str(text)
    matches = list(START_RE.finditer(txt))
    if not matches:
        return ""

    pieces = []
    for m in matches:
        sec = m.group(1).lower()
        if not (sec.startswith("жалоб") or sec.startswith("анамнез")):
            continue

        start = m.start()
        content_start = m.end()

        nxt = END_MARKER_RE.search(txt, content_start)
        end = nxt.start() if nxt else len(txt)

        snippet = txt[start:end].strip()

        # лёгкая чистка
        snippet = re.sub(r"[ \t]+", " ", snippet)
        snippet = re.sub(r"\n{3,}", "\n\n", snippet).strip()

        # игнорим слишком короткие
        if len(snippet) >= 40:
            pieces.append(snippet)

    # дедуп похожих кусков
    uniq = []
    seen = set()
    for p in pieces:
        key = re.sub(r"\s+", " ", p.lower())[:600]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)

    return "\n\n".join(uniq).strip()

def has_non_empty_icd_codes(obj) -> bool:
    """True если icd_codes существует и это список с хотя бы 1 элементом."""
    codes = obj.get("icd_codes", None)
    return isinstance(codes, list) and len(codes) > 0


kept = 0
dropped_no_icd = 0
dropped_no_sections = 0
total = 0

with open(INPUT_PATH, "r", encoding="utf-8") as f_in, open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue
        total += 1

        obj = json.loads(line)

        # 1) Дропаем строки, где icd_codes пустой
        if not has_non_empty_icd_codes(obj):
            dropped_no_icd += 1
            continue

        # 2) Оставляем только "Жалобы" + "Анамнез"
        original_text = obj.get("text", "") or ""
        extracted = extract_complaints_anamnesis(original_text)

        if DROP_IF_NO_COMPLAINTS_ANAMNESIS and not extracted:
            dropped_no_sections += 1
            continue

        obj["complaints_anamnesis_text"] = extracted
        obj["text"] = extracted  # как ты просил: text = только эти разделы

        f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        kept += 1

print("DONE")
print(f"Total input lines: {total}")
print(f"Kept lines: {kept}")
print(f"Dropped (empty icd_codes): {dropped_no_icd}")
print(f"Dropped (no complaints/anamnesis): {dropped_no_sections}")
print(f"Output file: {OUTPUT_PATH}")