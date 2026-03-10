"""Orbiter Retrieval: Embeddings, vector stores, and RAG pipeline."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from orbiter.retrieval.embeddings import Embeddings  # pyright: ignore[reportMissingImports]
from orbiter.retrieval.openai_embeddings import OpenAIEmbeddings  # pyright: ignore[reportMissingImports]
from orbiter.retrieval.types import (  # pyright: ignore[reportMissingImports]
    Chunk,
    Document,
    RetrievalError,
    RetrievalResult,
)

__all__ = [
    "Chunk",
    "Document",
    "Embeddings",
    "OpenAIEmbeddings",
    "RetrievalError",
    "RetrievalResult",
]
