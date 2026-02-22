"""
Stage 2 — Hybrid retrieval: BM25 + vector (FAISS), then RRF merge.
E5 models: encode queries as "query: <text>", chunks already stored as "passage: <text>" in index.
"""

import json
import re
from pathlib import Path

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src_advanced_rag.fusion import rrf_merge_by_rank

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
TOP_K = 200
RRF_KEEP = 400


def _tokenize_for_bm25(text: str) -> list[str]:
    text = text.lower().strip()
    tokens = re.findall(r"[a-zа-яё0-9]+\.?[a-zа-яё0-9]*|[A-Z]\d+\.?\d*", text, re.IGNORECASE)
    return [t for t in tokens if len(t) > 1 or t.isdigit()]


def _is_e5_model(name: str) -> bool:
    return "e5" in name.lower()


class HybridRetriever:
    def __init__(self, artifacts_dir: Path | None = None):
        self.artifacts_dir = artifacts_dir or ARTIFACTS_DIR
        self.chunks: list[dict] = []
        self.bm25: BM25Okapi | None = None
        self.faiss_index: faiss.Index | None = None
        self.encoder: SentenceTransformer | None = None
        self._encoder_name = ""
        self._load()

    def _load(self) -> None:
        with open(self.artifacts_dir / "bm25_chunks.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = data["chunks"]
        with open(self.artifacts_dir / "bm25_tokenized.json", "r", encoding="utf-8") as f:
            tokenized = json.load(f)
        self.bm25 = BM25Okapi(tokenized)
        self.faiss_index = faiss.read_index(str(self.artifacts_dir / "faiss.index"))
        meta = np.load(self.artifacts_dir / "chunk_metadata.npy", allow_pickle=True)
        if len(meta) != len(self.chunks):
            self.chunks = list(meta)
        try:
            with open(self.artifacts_dir / "encoder_name.txt", "r", encoding="utf-8") as f:
                self._encoder_name = f.read().strip()
        except Exception:
            self._encoder_name = "intfloat/multilingual-e5-large"
        self.encoder = SentenceTransformer(self._encoder_name)

    def _encode_query(self, query: str) -> np.ndarray:
        if _is_e5_model(self._encoder_name):
            text = "query: " + query
        else:
            text = query
        q = self.encoder.encode([text], normalize_embeddings=True)
        return np.array(q, dtype=np.float32)

    def bm25_search(self, query: str, top_k: int = TOP_K) -> list[tuple[int, float]]:
        tokenized = _tokenize_for_bm25(query)
        if not tokenized:
            return []
        scores = self.bm25.get_scores(tokenized)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]

    def vector_search(self, query: str, top_k: int = TOP_K) -> list[tuple[int, float]]:
        q = self._encode_query(query)
        scores, indices = self.faiss_index.search(q, top_k)
        out: list[tuple[int, float]] = []
        for i, s in zip(indices[0], scores[0]):
            if i >= 0:
                out.append((int(i), float(s)))
        return out

    def search(self, query: str, top_k: int = TOP_K) -> list[tuple[int, float]]:
        bm25_list = self.bm25_search(query, top_k)
        vec_list = self.vector_search(query, top_k)
        list_bm25 = [str(idx) for idx, _ in bm25_list]
        list_vec = [str(idx) for idx, _ in vec_list]
        merged = rrf_merge_by_rank([list_bm25, list_vec])
        return [(int(doc_id), score) for doc_id, score in merged[:RRF_KEEP]]

    def search_multi(self, queries: list[str], top_k: int = TOP_K) -> list[tuple[int, float]]:
        all_lists: list[list[str]] = []
        for q in queries:
            bm25_list = self.bm25_search(q, top_k)
            vec_list = self.vector_search(q, top_k)
            list_bm25 = [str(idx) for idx, _ in bm25_list]
            list_vec = [str(idx) for idx, _ in vec_list]
            merged = rrf_merge_by_rank([list_bm25, list_vec])
            all_lists.append([doc_id for doc_id, _ in merged[:RRF_KEEP]])
        merged = rrf_merge_by_rank(all_lists)
        return [(int(doc_id), score) for doc_id, score in merged[:RRF_KEEP]]


class ICDDocRetriever:
    """Retrieve ICD-level documents by query (same embedder, E5 query prefix)."""

    def __init__(self, artifacts_dir: Path | None = None):
        self.artifacts_dir = artifacts_dir or ARTIFACTS_DIR
        self.icd_docs: list[dict] = []
        self.icd_faiss: faiss.Index | None = None
        self.encoder: SentenceTransformer | None = None
        self._encoder_name = ""
        self._load()

    def _load(self) -> None:
        with open(self.artifacts_dir / "icd_docs.json", "r", encoding="utf-8") as f:
            self.icd_docs = json.load(f)
        self.icd_faiss = faiss.read_index(str(self.artifacts_dir / "icd_faiss.index"))
        try:
            with open(self.artifacts_dir / "encoder_name.txt", "r", encoding="utf-8") as f:
                self._encoder_name = f.read().strip()
        except Exception:
            self._encoder_name = "intfloat/multilingual-e5-large"
        self.encoder = SentenceTransformer(self._encoder_name)

    def _encode_query(self, query: str) -> np.ndarray:
        if _is_e5_model(self._encoder_name):
            text = "query: " + query
        else:
            text = query
        q = self.encoder.encode([text], normalize_embeddings=True)
        return np.array(q, dtype=np.float32)

    def search(self, query: str, top_k: int = 200) -> list[tuple[str, float]]:
        """Return [(icd_code, score), ...] for top_k ICD docs."""
        q = self._encode_query(query)
        scores, indices = self.icd_faiss.search(q, top_k)
        out: list[tuple[str, float]] = []
        for i, s in zip(indices[0], scores[0]):
            if i >= 0 and i < len(self.icd_docs):
                out.append((self.icd_docs[i]["icd_code"], float(s)))
        return out
