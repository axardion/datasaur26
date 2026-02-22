"""
Local embedding using Hugging Face sentence-transformers (no external API calls).
Default: intfloat/multilingual-e5-large (1024-dim, good for Russian and 90+ languages).
Register with graphrag_llm so settings.yaml can use type: "local".
"""

from typing import TYPE_CHECKING, Any, Optional, Unpack

from graphrag_llm.embedding.embedding import LLMEmbedding
from graphrag_llm.embedding.embedding_factory import register_embedding
from graphrag_llm.metrics.noop_metrics_store import NoopMetricsStore
from graphrag_llm.types import LLMEmbeddingResponse, LLMEmbeddingUsage
from openai.types.embedding import Embedding

if TYPE_CHECKING:
    from graphrag_llm.config import ModelConfig
    from graphrag_llm.metrics import MetricsStore
    from graphrag_llm.tokenizer import Tokenizer
    from graphrag_llm.types import LLMEmbeddingArgs

# Default: multilingual E5 large (1024-dim, supports Russian and 90+ languages)
# https://huggingface.co/intfloat/multilingual-e5-large
DEFAULT_MODEL = "intfloat/multilingual-e5-large"

# E5 models expect "query: " or "passage: " prefix (https://huggingface.co/intfloat/multilingual-e5-large)
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "
E5_PASSAGE_THRESHOLD_CHARS = 80  # texts longer than this get "passage: ", else "query: "

# Cache loaded model so we only load weights once per process (avoids reload on every request)
_loaded_models: dict[str, Any] = {}


def _get_model(model_name: str):
    """Return a cached SentenceTransformer so weights load only once."""
    from sentence_transformers import SentenceTransformer
    if model_name not in _loaded_models:
        _loaded_models[model_name] = SentenceTransformer(model_name)
    return _loaded_models[model_name]


class LocalPubMedEmbedding(LLMEmbedding):
    """Local embedding via sentence-transformers (default: multilingual-e5-large). No network calls."""

    def __init__(
        self,
        *,
        model_id: str,
        model_config: "ModelConfig",
        tokenizer: "Tokenizer",
        metrics_store: Optional["MetricsStore"] = None,
        metrics_processor: Any = None,
        rate_limiter: Any = None,
        retrier: Any = None,
        cache: Any = None,
        cache_key_creator: Any = None,
        **kwargs: Any,
    ):
        self._model_id = model_id
        self._model_config = model_config
        self._tokenizer = tokenizer
        self._metrics_store = metrics_store or NoopMetricsStore()
        model_name = getattr(model_config, "model", None) or DEFAULT_MODEL
        self._model = _get_model(model_name)
        self._use_e5_prefix = "e5" in model_name.lower() or "multilingual-e5" in model_name.lower()

    def embedding(
        self, /, **kwargs: Unpack["LLMEmbeddingArgs"]
    ) -> LLMEmbeddingResponse:
        input_texts = kwargs.get("input") or []
        if not input_texts:
            return LLMEmbeddingResponse(
                object="list",
                data=[],
                model=self._model_id,
                usage=LLMEmbeddingUsage(prompt_tokens=0, total_tokens=0),
            )
        if self._use_e5_prefix:
            prefixed = []
            for t in input_texts:
                s = t if isinstance(t, str) else str(t)
                prefix = E5_PASSAGE_PREFIX if len(s) > E5_PASSAGE_THRESHOLD_CHARS else E5_QUERY_PREFIX
                prefixed.append(prefix + s)
            input_texts = prefixed
        vectors = self._model.encode(
            input_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        data = [
            Embedding(
                object="embedding",
                embedding=vec.tolist(),
                index=i,
            )
            for i, vec in enumerate(vectors)
        ]
        return LLMEmbeddingResponse(
            object="list",
            data=data,
            model=self._model_id,
            usage=LLMEmbeddingUsage(prompt_tokens=0, total_tokens=0),
        )

    async def embedding_async(
        self, /, **kwargs: Unpack["LLMEmbeddingArgs"]
    ) -> LLMEmbeddingResponse:
        import asyncio
        return await asyncio.to_thread(self.embedding, **kwargs)

    @property
    def metrics_store(self) -> "MetricsStore":
        return self._metrics_store

    @property
    def tokenizer(self) -> "Tokenizer":
        return self._tokenizer


# Register so create_embedding(type="local", ...) works. Import this module before indexing or running the server.
register_embedding(
    "local",
    LocalPubMedEmbedding,
    scope="singleton",
)
