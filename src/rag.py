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
    text: str
    protocol_id: str
    icd_codes: list[str]
    source_file: str
    title: str


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if not text or not text.strip():
        return []
    text = text.replace("\ufeff", "").strip()
    chunks = []
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


def load_protocols_from_jsonl(jsonl_path: Path) -> list[dict]:
    protocols = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "text" in data and "icd_codes" in data:
                    protocols.append(data)
            except json.JSONDecodeError:
                continue
    return protocols


def load_protocols_from_dir(corpus_dir: Path) -> list[dict]:
    paths = sorted(corpus_dir.rglob("*.json"), key=lambda p: str(p))
    protocols = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "text" in data and "icd_codes" in data:
                protocols.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    return protocols


def load_protocols(corpus_path: Path) -> list[dict]:
    corpus_path = Path(corpus_path)
    if corpus_path.is_file():
        if corpus_path.suffix.lower() == ".jsonl":
            return load_protocols_from_jsonl(corpus_path)
        return []
    jsonl_file = corpus_path / "protocols_corpus.jsonl"
    if jsonl_file.is_file():
        return load_protocols_from_jsonl(jsonl_file)
    return load_protocols_from_dir(corpus_path)


def build_corpus(protocols: list[dict]) -> tuple[list[ProtocolChunk], list[str]]:
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
    def __init__(self, corpus_path: Path):
        self.corpus_path = Path(corpus_path)
        self.protocols = load_protocols(self.corpus_path)
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
        if not self.chunks or self.matrix is None:
            return []
        q_vec = self.vectorizer.transform([query])
        sim = cosine_similarity(q_vec, self.matrix).ravel()
        indices = sim.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in indices]
