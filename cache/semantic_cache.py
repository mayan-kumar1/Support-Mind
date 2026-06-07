import numpy as np
from sentence_transformers import SentenceTransformer
from logger import get_logger
from rag.pipeline import get_embedding_model

logger = get_logger(__name__)

SIMILARITY_THRESHOLD = 0.92
MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticCache:
    def __init__(self):
        self._model = None
        self._cache: list[dict] = []  # list of {embedding, query, response}
        logger.info("Semantic cache initialised  threshold: %.2f", SIMILARITY_THRESHOLD)

    @property
    def model(self) -> SentenceTransformer:
        return get_embedding_model()

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def get(self, query: str) -> str | None:
        """Check cache for a semantically similar query. Returns response or None."""
        if not self._cache:
            return None

        query_embedding = self.model.encode(query)

        best_score = 0.0
        best_response = None

        for entry in self._cache:
            score = self._cosine_similarity(query_embedding, entry["embedding"])  # type: ignore
            if score > best_score:
                best_score = score
                best_response = entry["response"]

        if best_score >= SIMILARITY_THRESHOLD:
            logger.info("Cache hit  similarity: %.4f  query: %s", best_score, query)
            return best_response

        logger.debug("Cache miss  best_similarity: %.4f  query: %s", best_score, query)
        return None

    def set(self, query: str, response: str) -> None:
        """Store a query-response pair in the cache."""
        embedding = self.model.encode(query)
        self._cache.append(
            {
                "embedding": embedding,
                "query": query,
                "response": response,
            }
        )
        logger.info(
            "Cache entry stored  query: %s  total_entries: %d", query, len(self._cache)
        )

    def size(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache = []
        logger.info("Cache cleared")


# Singleton — one cache shared across all users
semantic_cache = SemanticCache()
