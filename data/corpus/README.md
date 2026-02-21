# RAG corpus: Kazakhstan clinical protocols

This directory is the **knowledge base for RAG** (TF-IDF index). The server builds the index from **`protocols_corpus.jsonl`** in this directory.

## Format: JSONL

Place **`protocols_corpus.jsonl`** here: one JSON object per line. Each line must have:

- `protocol_id` (string)
- `text` (string) — full protocol text
- `icd_codes` (list of strings)

Optional: `source_file`, `title`.

Example line:

```json
{"protocol_id": "p_d57148b2d4", "source_file": "HELLP-СИНДРОМ.pdf", "title": "Одобрен", "icd_codes": ["O00", "O99", "O14.2"], "text": "Одобрен Объединенной комиссией..."}
```

If `protocols_corpus.jsonl` is missing, the server looks for `*.json` files in this directory instead (same field names per file).

## If this directory is empty

The server will fall back to `data/test_set` for the RAG index and print a warning.
