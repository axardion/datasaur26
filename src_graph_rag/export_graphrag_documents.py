import json
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = REPO_ROOT / "corpus" / "protocols_only_complaints_anamnesis_filtered.jsonl"
OUTPUT_DIR = REPO_ROOT / "data" / "graphrag_input"
OUTPUT_JSON = OUTPUT_DIR / "documents.json"

LEADING_BOILERPLATE = re.compile(
    r"^(?:Одобрен|Одобрено|Утверждено)[^К]*(?=КЛИНИЧЕСКИЙ ПРОТОКОЛ)",
    re.IGNORECASE | re.DOTALL,
)

ICD10_PATTERN = re.compile(r"^[A-Z]\d{2}(?:\.\d{1,2})?$", re.IGNORECASE)

SECTION_6_MARKERS = [
    "6. ОРГАНИЗАЦИОННЫЕ АСПЕКТЫ ПРОТОКОЛА",
    "6. ОРГАНИЗАЦИОННЫЕ АСПЕКТЫ",
    "6.1 Список разработчиков",
    "6.2 Указание на отсутствие конфликта интересов",
    "6.5 Список использованной литературы",
]

COMMISSION_BOILERPLATE = re.compile(
    r"Одобрен[^.]*\.\s*Объединенной комиссией[^.]*\.\s*",
    re.IGNORECASE,
)


def normalize_icd_codes(icd_codes: list) -> list:
    """Keep only valid-looking ICD-10 codes; normalize to uppercase for consistency."""
    if not icd_codes:
        return []
    out = []
    for c in icd_codes:
        if not isinstance(c, str):
            continue
        s = c.strip().upper()
        if ICD10_PATTERN.match(s):
            out.append(s)
    return list(dict.fromkeys(out))  # dedupe, preserve order


def clean_text(text: str) -> str:
    """Remove boilerplate and normalize whitespace so the graph/vector DB are not polluted."""
    if not text or not text.strip():
        return ""

    match = re.search(r"КЛИНИЧЕСКИЙ ПРОТОКОЛ", text, re.IGNORECASE)
    if match:
        text = text[match.start() :]

    for marker in SECTION_6_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break

    text = COMMISSION_BOILERPLATE.sub("", text)

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_document(record: dict) -> dict | None:
    """Build one GraphRAG document with cleaned text and schema fields. Returns None if text too short."""
    protocol_id = record.get("protocol_id", "")
    source_file = record.get("source_file", "")
    raw_icd = record.get("icd_codes") or []
    icd_codes = normalize_icd_codes(raw_icd)
    raw_text = record.get("text", "")

    cleaned = clean_text(raw_text)
    if icd_codes:
        icd_line = "Коды МКБ-10: " + ", ".join(icd_codes) + ".\n\n"
        cleaned = icd_line + cleaned

    return {
        "id": protocol_id,
        "text": cleaned,
        "title": source_file.replace(".pdf", "").strip('"«»'),
        "creation_date": datetime.now().strftime("%Y-%m-%dT00:00:00Z"),
        "icd_codes": icd_codes,
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Export protocols to GraphRAG documents.json")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Export only first N protocols (for quick GraphRAG index testing)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    documents = []

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if args.limit and len(documents) >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: skip line {i}: {e}", file=sys.stderr)
                continue
            doc = build_document(record)
            if not doc["text"]:
                print(f"Warning: empty text for {doc['id']} (line {i})", file=sys.stderr)
                continue
            documents.append(doc)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as out:
        json.dump(documents, out, ensure_ascii=False, indent=0)

    print(f"Exported {len(documents)} documents to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
