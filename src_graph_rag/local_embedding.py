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

DEFAULT_MODEL = "lokeshch19/ModernPubMedBERT"


class LocalPubMedEmbedding(LLMEmbedding):
    """Local embedding via sentence-transformers (ModernPubMedBERT). No network calls."""

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
        from sentence_transformers import SentenceTransformer

        self._model_id = model_id
        self._model_config = model_config
        self._tokenizer = tokenizer
        self._metrics_store = metrics_store or NoopMetricsStore()
        model_name = getattr(model_config, "model", None) or DEFAULT_MODEL
        self._model = SentenceTransformer(model_name)

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


register_embedding(
    "local",
    LocalPubMedEmbedding,
    scope="singleton",
)
