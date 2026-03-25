"""Abstract base class for embedding providers.

An ``Embeddings`` implementation converts text into dense vector
representations suitable for similarity search and retrieval.
"""

from __future__ import annotations

import abc


class Embeddings(abc.ABC):
    """Abstract base class for text embedding providers.

    Subclasses must implement ``embed()``, ``embed_batch()``, and the
    ``dimension`` property.

    Example:
        >>> vecs = await embedder.embed_batch(["hello", "world"])
        >>> len(vecs[0]) == embedder.dimension
        True
    """

    @abc.abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            A dense vector of length ``dimension``.
        """

    @abc.abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single call.

        Args:
            texts: A list of texts to embed.

        Returns:
            A list of dense vectors, one per input text.
        """

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """The dimensionality of the embedding vectors."""
