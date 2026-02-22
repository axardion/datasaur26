"""
Protocol-first ICD ranking: aggregate chunk scores per protocol, then output
candidates in strict order (primary > block_dotted > meta_dotted > block_nodot > meta_nodot > family).
Dotted codes (e.g. E78.0) always outrank family (E78). Fallback: score-weighted frequency in top chunks.
"""

import json
import math
import os
from collections import defaultdict
from pathlib import Path

from src_advanced_rag.retrieval import HybridRetriever, ICDDocRetriever
from src_advanced_rag.fusion import rrf_merge_by_rank

# Default DEBUG=1 (on); set DEBUG=0 to disable
DEBUG = os.getenv("DEBUG", "1").strip().lower() in ("1", "true", "yes")

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
TOP_ICD = 80
MAX_SNIPPET_CHARS = 350
ICD_DOC_TOP_K = 200
PROTOCOL_TOP_CHUNKS = 200


def _load_icd_index(artifacts_dir: Path) -> dict[str, str]:
    with open(artifacts_dir / "icd_index.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _load_corpus_icd_codes(artifacts_dir: Path) -> set[str]:
    with open(artifacts_dir / "corpus_icd_codes.json", "r", encoding="utf-8") as f:
        return set(json.load(f))


def _protocol_scores_from_chunk_scores(
    chunk_scores: list[tuple[int, float]],
    chunks: list[dict],
    top_chunks: int = PROTOCOL_TOP_CHUNKS,
) -> list[tuple[str, float]]:
    """Group chunks by protocol_id; aggregate score per protocol (sum). Sorted by score desc."""
    protocol_score: dict[str, float] = defaultdict(float)
    for chunk_idx, score in chunk_scores[:top_chunks]:
        if chunk_idx >= len(chunks):
            continue
        pid = chunks[chunk_idx].get("protocol_id") or ""
        if pid:
            protocol_score[pid] += score
    return sorted(protocol_score.items(), key=lambda x: -x[1])


def _has_dot(code: str) -> bool:
    """True if code contains a dot (e.g. E78.0). Dotted = more specific than family (E78)."""
    return "." in str(code)


def _codes_for_protocol(chunk: dict) -> tuple[str, list[str], str, list[str]]:
    """(primary_icd, icd_codes_strong, icd_family, icd_codes_meta) from chunk."""
    primary = str(chunk.get("primary_icd") or "").strip().upper()
    block = [str(c).strip().upper() for c in (chunk.get("icd_codes_strong") or []) if str(c).strip()]
    family = str(chunk.get("icd_family") or "").strip().upper()
    meta = [str(c).strip().upper() for c in (chunk.get("icd_codes_meta") or []) if str(c).strip()]
    return primary, block, family, meta


def _ordered_candidates_for_protocol(
    chunk: dict,
    corpus_icd_codes: set[str],
) -> list[str]:
    """
    Strict candidate order for one protocol (dotted codes always before family):
    1. primary_icd (if exists and valid)
    2. icd_block_codes WITH dot
    3. meta_icd_codes WITH dot
    4. icd_block_codes without dot
    5. meta_icd_codes without dot
    6. icd_family (last resort only)
    """
    primary, block, family, meta = _codes_for_protocol(chunk)
    out: list[str] = []
    seen: set[str] = set()

    def add(c: str) -> None:
        if c and c in corpus_icd_codes and c not in seen:
            seen.add(c)
            out.append(c)

    if primary:
        add(primary)
    for code in block:
        if _has_dot(code):
            add(code)
    for code in meta:
        if _has_dot(code):
            add(code)
    for code in block:
        if not _has_dot(code):
            add(code)
    for code in meta:
        if not _has_dot(code):
            add(code)
    if family:
        add(family)
    return out


def _diagnoses_from_protocols(
    protocol_order: list[tuple[str, float]],
    chunks: list[dict],
    corpus_icd_codes: set[str],
    max_diagnoses: int = 3,
) -> list[str]:
    """Build rank1..rank3 from best protocols using strict order (dotted before family)."""
    out: list[str] = []
    seen: set[str] = set()
    protocol_id_to_chunk: dict[str, dict] = {}
    for c in chunks:
        pid = c.get("protocol_id") or ""
        if pid and pid not in protocol_id_to_chunk:
            protocol_id_to_chunk[pid] = c
    for protocol_id, score in protocol_order:
        if len(out) >= max_diagnoses:
            break
        c = protocol_id_to_chunk.get(protocol_id)
        if not c:
            continue
        candidates = _ordered_candidates_for_protocol(c, corpus_icd_codes)
        if DEBUG and candidates:
            print(f"[DEBUG] protocol={protocol_id} score={score:.4f} candidates={candidates[:10]}")
        for code in candidates:
            if code not in seen and len(out) < max_diagnoses:
                seen.add(code)
                out.append(code)
            if len(out) >= max_diagnoses:
                break
    return out[:max_diagnoses]


def _weighted_frequency_code_in_top_chunks(
    chunk_scores: list[tuple[int, float]],
    chunks: list[dict],
    corpus_icd_codes: set[str],
    exclude: set[str],
    top_n: int = 100,
) -> str | None:
    """Fallback: ICD with highest score-weighted frequency in top chunks. Only corpus codes; exclude already selected."""
    code_weight: dict[str, float] = defaultdict(float)
    for chunk_idx, score in chunk_scores[:top_n]:
        if chunk_idx >= len(chunks):
            continue
        c = chunks[chunk_idx]
        for code in c.get("icd_codes_strong") or c.get("icd_codes") or c.get("icd_codes_meta") or []:
            code = str(code).strip().upper()
            if code and code in corpus_icd_codes and code not in exclude:
                code_weight[code] += score
    if not code_weight:
        return None
    return max(code_weight.items(), key=lambda x: x[1])[0]


def aggregate_icd_scores(
    chunk_scores: list[tuple[int, float]],
    chunks: list[dict],
) -> list[tuple[str, float, str]]:
    """
    Distribute chunk RRF score to ICDs with weight 1/sqrt(num_icd_codes_in_protocol)
    to reduce dilution from protocols with many codes.
    Returns (icd_code, aggregate_score, best_snippet) for top ICDs.
    """
    icd_score: dict[str, float] = defaultdict(float)
    icd_snippet: dict[str, str] = {}
    for chunk_idx, score in chunk_scores:
        if chunk_idx >= len(chunks):
            continue
        c = chunks[chunk_idx]
        n_icd = max(1, len(c.get("icd_codes", [])))
        weight = 1.0 / math.sqrt(n_icd)
        text = (c.get("chunk_text") or "")[:MAX_SNIPPET_CHARS]
        for code in c.get("icd_codes", []):
            code = str(code).strip()
            if not code:
                continue
            icd_score[code] += score * weight
            if code not in icd_snippet or len(text) > len(icd_snippet.get(code, "")):
                icd_snippet[code] = text
    sorted_icd = sorted(icd_score.items(), key=lambda x: -x[1])[:TOP_ICD]
    return [(code, sc, icd_snippet.get(code, "")[:MAX_SNIPPET_CHARS]) for code, sc in sorted_icd]


def get_icd_short_title(icd_code: str, icd_index: dict[str, str]) -> str:
    raw = icd_index.get(icd_code, "")
    if not raw:
        return icd_code
    first = raw[:300].strip()
    return first.replace("\n", " ") if first else icd_code


class ICDRanker:
    def __init__(self, artifacts_dir: Path | None = None):
        self.artifacts_dir = artifacts_dir or ARTIFACTS_DIR
        self.retriever = HybridRetriever(self.artifacts_dir)
        self.icd_doc_retriever: ICDDocRetriever | None = None
        try:
            if (self.artifacts_dir / "icd_faiss.index").exists():
                self.icd_doc_retriever = ICDDocRetriever(self.artifacts_dir)
        except Exception:
            self.icd_doc_retriever = None
        self.icd_index = _load_icd_index(self.artifacts_dir)
        self.corpus_icd_codes = _load_corpus_icd_codes(self.artifacts_dir)

    def get_top_diagnoses_protocol_first(
        self,
        queries: list[str],
        top_chunks: int = PROTOCOL_TOP_CHUNKS,
    ) -> list[dict]:
        """
        Protocol-first: (1) top-K chunks, (2) group by protocol_id, sum scores,
        (3) sort protocols desc, (4) collect up to 3 codes in strict order per protocol
        (primary > block_dotted > meta_dotted > block_nodot > meta_nodot > family).
        Fallback: score-weighted frequency in top chunks; then emergency corpus fill.
        """
        chunk_scores = self.retriever.search_multi(queries, top_k=top_chunks)
        chunks = self.retriever.chunks
        protocol_order = _protocol_scores_from_chunk_scores(
            chunk_scores, chunks, top_chunks=top_chunks
        )
        if DEBUG and protocol_order:
            for i, (pid, sc) in enumerate(protocol_order[:5]):
                print(f"[DEBUG] protocol_rank={i+1} protocol_id={pid} score={sc:.4f}")
        codes = _diagnoses_from_protocols(
            protocol_order, chunks, self.corpus_icd_codes, max_diagnoses=3
        )
        # Fallback: weighted frequency among top chunks (only corpus codes)
        exclude = set(codes)
        while len(codes) < 3:
            next_code = _weighted_frequency_code_in_top_chunks(
                chunk_scores, chunks, self.corpus_icd_codes, exclude=exclude, top_n=100
            )
            if next_code:
                codes.append(next_code)
                exclude.add(next_code)
            else:
                break
        # Emergency: fill to 3 from corpus
        while len(codes) < 3:
            for c in self.corpus_icd_codes:
                if c not in codes:
                    codes.append(c)
                    break
            else:
                break
        codes = codes[:3]
        if DEBUG and codes:
            print(f"[DEBUG] final diagnoses={codes}")
        return [{"rank": i + 1, "icd10_code": c} for i, c in enumerate(codes)]

    def get_top_icd_candidates(
        self,
        queries: list[str],
    ) -> list[tuple[str, float, str, str]]:
        """
        Chunk retrieval + ICD-doc retrieval; merge with RRF; aggregate chunk scores with 1/sqrt(n_icd).
        Returns (icd_code, score, short_title, evidence_snippet) for top TOP_ICD.
        """
        chunk_scores = self.retriever.search_multi(queries, top_k=200)
        chunks = self.retriever.chunks
        from_chunks = aggregate_icd_scores(chunk_scores, chunks)

        # ICD-doc retrieval: top 200 by query (use first query only for ICD-doc)
        icd_doc_scores: dict[str, float] = {}
        if self.icd_doc_retriever is not None and queries:
            for q in queries[:3]:
                for code, sim in self.icd_doc_retriever.search(q, top_k=ICD_DOC_TOP_K):
                    icd_doc_scores[code] = icd_doc_scores.get(code, 0) + sim

        # Merge: RRF-style merge of chunk-derived ICD list and ICD-doc list
        chunk_icd_list = [code for code, _, _ in from_chunks]
        doc_icd_list = sorted(icd_doc_scores.keys(), key=lambda c: -icd_doc_scores[c])[:TOP_ICD]
        merged_rrf = rrf_merge_by_rank([chunk_icd_list, doc_icd_list])

        # Build (code, combined_score, title, snippet); use chunk score and snippet when available
        chunk_score_map = {code: (sc, sn) for code, sc, sn in from_chunks}
        result: list[tuple[str, float, str, str]] = []
        seen: set[str] = set()
        for code, rrf_s in merged_rrf:
            if code not in self.corpus_icd_codes or code in seen:
                continue
            seen.add(code)
            cs, snippet = chunk_score_map.get(code, (0.0, ""))
            combined = rrf_s + 0.5 * cs
            title = get_icd_short_title(code, self.icd_index)
            result.append((code, combined, title, snippet[:MAX_SNIPPET_CHARS]))
            if len(result) >= TOP_ICD:
                break
        # Backfill from chunk list if we have fewer than TOP_ICD
        for code, sc, snippet in from_chunks:
            if code in seen:
                continue
            if code not in self.corpus_icd_codes:
                continue
            seen.add(code)
            title = get_icd_short_title(code, self.icd_index)
            result.append((code, sc, title, snippet[:MAX_SNIPPET_CHARS]))
            if len(result) >= TOP_ICD:
                break
        result.sort(key=lambda x: -x[1])
        return result[:TOP_ICD]
