
# # How to use for juric team
#
# **Prereqs:** Build the GraphRAG index locally so `output/`, `data/graphrag_input/`, and `prompts/` exist.
#
# **Build:**
#   docker build -t submission .
#
# **Run (serve on port 8080):**
#   docker run -p 8080:8080 -e GRAPHRAG_API_KEY="your-key" submission
#
# **Optional env at run:**
#   - GRAPHRAG_API_KEY — API key for the completion model (required for diagnosis).
#   - OPENAI_BASE_URL  — LLM endpoint (default in image: https://hub.qazcode.ai).
#     Example: docker run -p 8080:8080 -e GRAPHRAG_API_KEY="..." -e OPENAI_BASE_URL="https://api.openai.com/v1" submission
#
# **Test:** POST to http://localhost:8080/diagnose with body `{"symptoms": "..."}`; see /docs for Swagger.
#
# ---

FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src_graph_rag/ ./src_graph_rag/
COPY static/ ./static/

COPY data/graphrag_input/ ./data/graphrag_input/
COPY output/ ./output/
COPY prompts/ ./prompts/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV OPENAI_BASE_URL=https://hub.qazcode.ai

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "src_graph_rag.diagnose_server:app", "--host", "0.0.0.0", "--port", "8080"]
