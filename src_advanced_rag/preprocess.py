"""
Stage 0 — Offline preprocessing.

Build:
- BM25 chunk metadata + tokenized corpus
- FAISS chunk index + chunk metadata
- ICD index (lightweight dictionary)
- ICD-doc FAISS (SAFE version: uses primary_icd/icd_family/strong codes, not noisy meta join)

Designed to work well with a cleaned corpus JSONL such as:
  clean_protocols_corpus.v2.jsonl
where each line may contain:
  clean_text_core, clean_text, meta_icd_codes, icd_block_codes, primary_icd, icd_family

CPU-friendly defaults.

Run:
  python preprocess.py --corpus ./clean_protocols_corpus.v2.jsonl

Env:
  EMBEDDER_NAME=intfloat/multilingual-e5-base   (recommended on CPU)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env from project root so EMBEDDER_NAME etc. can be set there
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Embedder: env-configurable; E5 models use "passage:" / "query:" prefix
EMBEDDER_NAME = os.getenv("EMBEDDER_NAME", "intfloat/multilingual-e5-large")

# Chunking: smaller chunks often improve accuracy; still OK for latency on CPU after cleaning
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "3200"))
OVERLAP_CHARS = int(os.getenv("OVERLAP_CHARS", "250"))
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "300"))

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

# Keywords for ICD doc enrichment (optional; kept small for CPU)
ICD_DOC_KEYWORDS = [
    "Клиника", "Диагностика", "Критерии", "Дифференциальная", "Жалобы",
    "клиника", "диагностика", "критерии", "дифференциальная", "жалобы",
]


def _is_e5_model(name: str) -> bool:
    return "e5" in name.lower()


def _normalize_codes(x: Any) -> list[str]:
    if not x:
        return []
    out: list[str] = []
    for c in x:
        s = str(c).strip().upper()
        if s:
            out.append(s)
    return out


def _tokenize_for_bm25(text: str) -> list[str]:
    """
    BM25 tokenization:
    - keep Cyrillic/Latin words, numbers
    - keep ICD-like tokens with dots
    """
    if not text:
        return []
    text = text.replace("\n", " ").lower().strip()
    tokens = re.findall(r"[a-zа-яё0-9]+(?:\.[a-zа-яё0-9]+)?", text, re.IGNORECASE)
    return [t for t in tokens if len(t) > 1 or t.isdigit()]


def clean_text(text: str) -> str:
    """
    Corpus is assumed to be cleaned offline.
    Keep newlines; just trim and normalize trivial whitespace.
    """
    if not text or not isinstance(text, str):
        return ""
    # preserve \n, normalize spaces inside lines
    lines = text.splitlines()
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in lines]
    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


def load_corpus(corpus_path: Path) -> list[dict[str, Any]]:
    """
    Load protocols from .jsonl or directory of .json files.

    Supports:
    - original corpus: {text, icd_codes, title, source_file, protocol_id}
    - cleaned corpus v2: {clean_text_core, clean_text, meta_icd_codes, icd_block_codes,
                          primary_icd, icd_family, ...}

    Returned protocol dict includes:
      protocol_id, title, source_file, text,
      icd_codes (best available), icd_codes_strong, icd_codes_meta,
      primary_icd, icd_family
    """
    protocols: list[dict[str, Any]] = []

    def _extract_protocol(obj: dict[str, Any]) -> dict[str, Any] | None:
        # pick best text
        text = (
            obj.get("clean_text_core")
            or obj.get("clean_text")
            or obj.get("text")
            or ""
        )
        text = clean_text(text)
        if not text:
            return None

        strong = _normalize_codes(obj.get("icd_block_codes"))
        meta = _normalize_codes(obj.get("meta_icd_codes") or obj.get("icd_codes"))
        icd_codes = strong if strong else meta

        return {
            "protocol_id": obj.get("protocol_id", ""),
            "title": obj.get("title", "") or "",
            "source_file": obj.get("source_file", "") or "",
            "text": text,
            # compatibility:
            "icd_codes": icd_codes,
            # richer signals (used for ICD docs):
            "icd_codes_strong": strong,
            "icd_codes_meta": meta,
            "primary_icd": str(obj.get("primary_icd") or "").strip().upper(),
            "icd_family": str(obj.get("icd_family") or "").strip().upper(),
        }

    if corpus_path.is_file() and corpus_path.suffix == ".jsonl":
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # accept either original or cleaned
                if ("text" in obj) or ("clean_text" in obj) or ("clean_text_core" in obj):
                    p = _extract_protocol(obj)
                    if p is not None and p.get("icd_codes") is not None:
                        protocols.append(p)

    elif corpus_path.is_dir():
        for jpath in corpus_path.glob("*.json"):
            with open(jpath, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if ("text" in obj) or ("clean_text" in obj) or ("clean_text_core" in obj):
                p = _extract_protocol(obj)
                if p is not None and p.get("icd_codes") is not None:
                    protocols.append(p)

    return protocols


def chunk_protocol(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Split protocol text into overlapping chunks (char-based) preserving newlines."""
    text = clean_text(protocol.get("text", ""))
    if not text:
        return []

    chunks: list[dict[str, Any]] = []
    start = 0
    chunk_id = 0
    n = len(text)

    while start < n:
        end = min(start + CHUNK_CHARS, n)
        chunk_text = text[start:end].strip()

        if len(chunk_text) >= MIN_CHUNK_CHARS:
            chunks.append({
                "protocol_id": protocol["protocol_id"],
                "title": protocol.get("title", ""),
                "source_file": protocol.get("source_file", ""),
                "primary_icd": protocol.get("primary_icd", ""),
                "icd_family": protocol.get("icd_family", ""),
                "icd_codes": list(protocol.get("icd_codes", [])),
                "icd_codes_strong": list(protocol.get("icd_codes_strong", [])),
                "icd_codes_meta": list(protocol.get("icd_codes_meta", [])),
                "chunk_id": chunk_id,
                "chunk_text": chunk_text,
            })
            chunk_id += 1

        if end >= n:
            break
        start = max(0, end - OVERLAP_CHARS)

    return chunks


def _extract_sections_near_keywords(text: str, keywords: list[str], window: int = 900) -> str:
    """Extract short windows around keyword occurrences (optional enrichment)."""
    if not text:
        return ""
    out: list[str] = []
    for kw in keywords:
        pos = 0
        while True:
            i = text.find(kw, pos)
            if i < 0:
                break
            start = max(0, i - 120)
            end = min(len(text), i + window)
            out.append(text[start:end])
            pos = i + 1
    merged = " ".join(out)
    merged = re.sub(r"\s+", " ", merged).strip()
    return merged[:4500] if merged else ""


def build_icd_index(protocols: list[dict[str, Any]]) -> dict[str, str]:
    """
    Lightweight ICD index for quick lookup:
    - key = primary_icd OR icd_family OR strong codes (fallback)
    - value = aggregated short snippets (title + head)
    This avoids poisoning by joining full texts for every meta icd_code.
    """
    icd_to_parts: dict[str, list[str]] = {}
    for p in protocols:
        title = (str(p.get("title") or "") + " " + str(p.get("source_file") or "")).strip()
        text = clean_text(p.get("text", ""))
        head = re.sub(r"\s+", " ", text[:2500]).strip()

        primary = str(p.get("primary_icd") or "").strip().upper()
        family = str(p.get("icd_family") or "").strip().upper()
        strong = [str(c).strip().upper() for c in (p.get("icd_codes_strong") or []) if str(c).strip()]

        keys: list[str] = []
        if primary:
            keys.append(primary)
        if family:
            keys.append(family)
        if not keys:
            keys = strong

        for code in keys:
            if not code:
                continue
            icd_to_parts.setdefault(code, []).append(title)
            if head:
                icd_to_parts[code].append(head)

    return {icd: " ".join(parts)[:15000] for icd, parts in icd_to_parts.items()}


def build_icd_docs(protocols: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    SAFE ICD docs for FAISS:
    - Keys: primary_icd and icd_family (plus strong codes if no primary/family)
    - Text: title + short head of cleaned text + small keyword windows
    This avoids "poisoned" ICD docs created from noisy meta icd_codes lists.
    """
    icd_to_parts: dict[str, list[str]] = {}

    for p in protocols:
        title = (str(p.get("title") or "") + " " + str(p.get("source_file") or "")).strip()
        text = clean_text(p.get("text", ""))

        primary = str(p.get("primary_icd") or "").strip().upper()
        family = str(p.get("icd_family") or "").strip().upper()
        strong = [str(c).strip().upper() for c in (p.get("icd_codes_strong") or []) if str(c).strip()]

        keys: list[str] = []
        if primary:
            keys.append(primary)
        if family:
            keys.append(family)
        if not keys:
            keys = strong

        if not keys:
            continue

        head = text[:3500]
        sections = _extract_sections_near_keywords(text, ICD_DOC_KEYWORDS)

        # compact doc text (CPU-friendly)
        base = " ".join([
            title,
            f"PRIMARY={primary}" if primary else "",
            re.sub(r"\s+", " ", head).strip(),
            sections,
        ])
        base = re.sub(r"\s+", " ", base).strip()[:12000]

        for k in keys:
            if not k:
                continue
            icd_to_parts.setdefault(k, []).append(base)

    docs = [{"icd_code": icd, "text": " ".join(parts)[:12000]} for icd, parts in icd_to_parts.items()]
    docs.sort(key=lambda d: d["icd_code"])
    return docs


def run(corpus_path: Path | None = None) -> None:
    """Load corpus, chunk, build BM25 + FAISS + ICD index + ICD-doc FAISS, persist."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    root = Path(__file__).resolve().parent.parent
    default_jsonl = root / "clean_protocols_corpus.v2.jsonl"
    if not default_jsonl.exists():
        default_jsonl = root / "protocols_corpus.jsonl"

    path = corpus_path or default_jsonl
    if not path.exists():
        raise FileNotFoundError(f"Corpus path not found: {path}")

    protocols = load_corpus(path)
    if not protocols:
        raise ValueError(f"No protocols loaded from {path}")

    all_chunks: list[dict[str, Any]] = []
    for p in protocols:
        all_chunks.extend(chunk_protocol(p))
    if not all_chunks:
        raise ValueError("No chunks produced. Check CHUNK_CHARS / input data.")

    # BM25 artifacts (store chunks + tokenized; BM25 itself rebuilt at runtime)
    tokenized = [_tokenize_for_bm25(c["chunk_text"]) for c in all_chunks]
    _ = BM25Okapi(tokenized)  # build once to validate tokenization (optional)

    with open(ARTIFACTS_DIR / "bm25_chunks.json", "w", encoding="utf-8") as f:
        json.dump({"chunks": all_chunks}, f, ensure_ascii=False, indent=0)
    with open(ARTIFACTS_DIR / "bm25_tokenized.json", "w", encoding="utf-8") as f:
        json.dump(tokenized, f, ensure_ascii=False)

    # Embedder
    model = SentenceTransformer(EMBEDDER_NAME)
    is_e5 = _is_e5_model(EMBEDDER_NAME)

    # Chunk FAISS
    if is_e5:
        chunk_texts_for_encode = ["passage: " + c["chunk_text"] for c in all_chunks]
    else:
        chunk_texts_for_encode = [c["chunk_text"] for c in all_chunks]

    # CPU-friendly batch sizes
    batch = 16 if "large" in EMBEDDER_NAME.lower() else 32
    try:
        chunk_embeddings = model.encode(
            chunk_texts_for_encode,
            batch_size=batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    except RuntimeError:
        chunk_embeddings = model.encode(
            chunk_texts_for_encode,
            batch_size=max(8, batch // 2),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    chunk_embeddings = np.array(chunk_embeddings, dtype=np.float32)
    chunk_index = faiss.IndexFlatIP(chunk_embeddings.shape[1])
    chunk_index.add(chunk_embeddings)

    faiss.write_index(chunk_index, str(ARTIFACTS_DIR / "faiss.index"))
    np.save(ARTIFACTS_DIR / "chunk_metadata.npy", np.array(all_chunks, dtype=object), allow_pickle=True)

    # ICD index (lightweight dict)
    icd_index = build_icd_index(protocols)
    with open(ARTIFACTS_DIR / "icd_index.json", "w", encoding="utf-8") as f:
        json.dump(icd_index, f, ensure_ascii=False)

    # ICD documents for retrieval (SAFE)
    icd_docs = build_icd_docs(protocols)
    with open(ARTIFACTS_DIR / "icd_docs.json", "w", encoding="utf-8") as f:
        json.dump(icd_docs, f, ensure_ascii=False)

    # ICD-doc FAISS
    if icd_docs:
        if is_e5:
            icd_texts_for_encode = ["passage: " + d["text"] for d in icd_docs]
        else:
            icd_texts_for_encode = [d["text"] for d in icd_docs]

        try:
            icd_embeddings = model.encode(
                icd_texts_for_encode,
                batch_size=batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
        except RuntimeError:
            icd_embeddings = model.encode(
                icd_texts_for_encode,
                batch_size=max(8, batch // 2),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )

        icd_embeddings = np.array(icd_embeddings, dtype=np.float32)
        icd_faiss = faiss.IndexFlatIP(icd_embeddings.shape[1])
        icd_faiss.add(icd_embeddings)
        faiss.write_index(icd_faiss, str(ARTIFACTS_DIR / "icd_faiss.index"))
    else:
        # still write an empty index list; caller can handle absence
        with open(ARTIFACTS_DIR / "icd_faiss.index.MISSING", "w", encoding="utf-8") as f:
            f.write("No icd_docs produced.\n")

    # Corpus ICD list (prefer primary/family keys; fallback to strong/meta)
    all_icd: set[str] = set()
    for p in protocols:
        primary = str(p.get("primary_icd") or "").strip().upper()
        family = str(p.get("icd_family") or "").strip().upper()
        strong = [str(c).strip().upper() for c in (p.get("icd_codes_strong") or []) if str(c).strip()]
        meta = [str(c).strip().upper() for c in (p.get("icd_codes_meta") or []) if str(c).strip()]
        if primary:
            all_icd.add(primary)
        if family:
            all_icd.add(family)
        for c in strong:
            all_icd.add(c)
        for c in meta:
            all_icd.add(c)

    with open(ARTIFACTS_DIR / "corpus_icd_codes.json", "w", encoding="utf-8") as f:
        json.dump(sorted(all_icd), f, ensure_ascii=False)

    with open(ARTIFACTS_DIR / "encoder_name.txt", "w", encoding="utf-8") as f:
        f.write(EMBEDDER_NAME)

    # Build stats file for quick sanity check
    stats = {
        "corpus_path": str(path),
        "protocols": len(protocols),
        "chunks": len(all_chunks),
        "icd_docs": len(icd_docs),
        "embedder": EMBEDDER_NAME,
        "chunk_chars": CHUNK_CHARS,
        "overlap_chars": OVERLAP_CHARS,
    }
    with open(ARTIFACTS_DIR / "preprocess_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Protocols: {len(protocols)}, Chunks: {len(all_chunks)}, ICD-docs: {len(icd_docs)}")
    print(f"Artifacts written to: {ARTIFACTS_DIR}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Path to corpus (.jsonl or dir). Prefer clean_protocols_corpus.v2.jsonl",
    )
    args = parser.parse_args()
    run(args.corpus)


if __name__ == "__main__":
    main()