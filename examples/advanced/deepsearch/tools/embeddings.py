"""Multi-provider embeddings and similarity functions."""
from __future__ import annotations
import logging
import math
import httpx
from abc import ABC, abstractmethod
from collections import Counter

logger = logging.getLogger("deepsearch")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def jaccard_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity between two texts."""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def jaccard_rank(query: str, documents: list[str]) -> list[dict]:
    """Rank documents by Jaccard similarity to query."""
    results = []
    for i, doc in enumerate(documents):
        score = jaccard_similarity(query, doc)
        results.append({"index": i, "relevance_score": score})
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        ...


class JinaEmbeddingProvider(EmbeddingProvider):
    """Jina Embeddings API."""
    def __init__(self, api_key: str, model: str = "jina-embeddings-v3", dimensions: int = 256) -> None:
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": texts,
            "dimensions": self.dimensions,
            "embedding_type": "float",
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://api.jina.ai/v1/embeddings",
                    json=payload, headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()

            # Sort by index to ensure correct order
            sorted_data = sorted(data.get("data", []), key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_data]
        except Exception as e:
            logger.warning("Jina embeddings failed: %s", e)
            return []


class LocalEmbeddingProvider(EmbeddingProvider):
    """Simple TF-IDF based embeddings - no external API needed."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # Build vocabulary from all texts
        all_words: set[str] = set()
        tokenized = []
        for text in texts:
            words = text.lower().split()
            tokenized.append(words)
            all_words.update(words)

        if not all_words:
            return [[0.0] for _ in texts]

        vocab = sorted(all_words)
        word_to_idx = {w: i for i, w in enumerate(vocab)}

        # Compute TF-IDF vectors
        # Document frequency
        df = Counter()
        for words in tokenized:
            for w in set(words):
                df[w] += 1

        n_docs = len(texts)
        embeddings = []
        for words in tokenized:
            tf = Counter(words)
            vec = [0.0] * len(vocab)
            for w, count in tf.items():
                idx = word_to_idx[w]
                idf = math.log(n_docs / (1 + df[w])) + 1
                vec[idx] = count * idf

            # Normalize
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]
            embeddings.append(vec)

        return embeddings


def get_embedding_provider(config) -> EmbeddingProvider:
    """Factory to create the appropriate embedding provider."""
    name = config.embedding_provider
    if name == "jina" and config.jina_api_key:
        return JinaEmbeddingProvider(config.jina_api_key)
    return LocalEmbeddingProvider()
