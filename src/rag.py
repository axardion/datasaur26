"""
DIRECT-style RAG: load Kazakhstan clinical protocols, chunk, index with TF-IDF,
retrieve relevant premises for symptom queries.

DiReCT (Diagnostic Reasoning for Clinical Notes) uses a knowledge graph to
provide "premises" (golden standards) to the LLM. Here we use retrieved
protocol text as premises for diagnosis with ICD-10 codes.
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


@dataclass
class ProtocolChunk:
    """Single chunk of a protocol with metadata."""
    text: str
    protocol_id: str
    icd_codes: list[str]
    source_file: str
    title: str


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks (by character, trying to break at sentence boundaries)."""
    if not text or not text.strip():
        return []
    text = text.replace("\ufeff", "").strip()
    chunks = []
    # Prefer splitting on double newline (paragraph) or single newline
    parts = re.split(r"\n\s*\n", text)
    current = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(current) + len(part) + 2 <= size:
            current = f"{current}\n\n{part}".strip() if current else part
        else:
            if current:
                # Split current if still too long
                while len(current) > size:
                    chunk = current[:size]
                    last_break = max(chunk.rfind(". "), chunk.rfind(".\n"), chunk.rfind(" "))
                    if last_break > size // 2:
                        chunk = current[: last_break + 1]
                        current = current[last_break + 1 :].strip()
                    else:
                        current = current[size - overlap :].strip()
                    chunks.append(chunk)
                if current:
                    chunks.append(current)
                    current = current[-overlap:] if len(current) >= overlap else ""
            current = part
    if current:
        chunks.append(current)
    return chunks


def load_protocols_from_dir(corpus_dir: Path) -> list[dict]:
    """Load all JSON protocol files from a directory (flat or nested)."""
    protocols = []
    for path in corpus_dir.rglob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "text" in data and "icd_codes" in data:
                protocols.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    return protocols


def build_corpus(protocols: list[dict]) -> tuple[list[ProtocolChunk], list[str]]:
    """Build list of ProtocolChunk and list of chunk texts for vectorizer."""
    chunks: list[ProtocolChunk] = []
    texts: list[str] = []
    for p in protocols:
        protocol_id = p.get("protocol_id", "")
        icd_codes = p.get("icd_codes", [])
        if not isinstance(icd_codes, list):
            icd_codes = list(icd_codes) if icd_codes else []
        source_file = p.get("source_file", "")
        title = p.get("title", "")
        text = p.get("text", "")
        for chunk_str in chunk_text(text):
            pc = ProtocolChunk(
                text=chunk_str,
                protocol_id=protocol_id,
                icd_codes=icd_codes,
                source_file=source_file,
                title=title,
            )
            chunks.append(pc)
            texts.append(chunk_str)
    return chunks, texts


class DirectRAGIndex:
    """TF-IDF index over protocol chunks for DIRECT-style retrieval."""

    def __init__(self, corpus_dir: Path):
        self.corpus_dir = Path(corpus_dir)
        self.protocols = load_protocols_from_dir(self.corpus_dir)
        self.chunks, self.texts = build_corpus(self.protocols)
        self.vectorizer = TfidfVectorizer(
            max_features=50_000,
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            analyzer="char_wb",
            ngram_range=(2, 5),
        )
        self.matrix = self.vectorizer.fit_transform(self.texts) if self.texts else None

    def retrieve(self, query: str, top_k: int = 10) -> list[ProtocolChunk]:
        """Return top-k most relevant protocol chunks for the symptom query."""
        if not self.chunks or self.matrix is None:
            return []
        q_vec = self.vectorizer.transform([query])
        sim = cosine_similarity(q_vec, self.matrix).ravel()
        indices = sim.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in indices]
