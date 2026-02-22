"""
Reciprocal Rank Fusion (RRF) to merge multiple ranked lists.
score = sum(1 / (k + rank)) over all lists; k=60 is standard.
"""

from collections import defaultdict

RRF_K = 60


def rrf_merge_by_rank(ranked_lists: list[list[str]]) -> list[tuple[str, float]]:
    """
    ranked_lists: list of lists of doc_ids, each list in rank order (best first).
    Returns [(doc_id, rrf_score), ...] sorted by rrf_score descending.
    """
    scores: dict[str, float] = defaultdict(float)
    for lst in ranked_lists:
        for rank, doc_id in enumerate(lst, start=1):
            scores[doc_id] += 1.0 / (RRF_K + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
