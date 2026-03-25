"""Exo Memory Backends: storage implementations.

Backend modules have optional dependencies — import directly:
  from exo.memory.backends.sqlite import SQLiteMemoryStore
  from exo.memory.backends.postgres import PostgresMemoryStore
  from exo.memory.backends.vector import (
      VectorMemoryStore, Embeddings,
      EmbeddingProvider, OpenAIEmbeddingProvider,
      SentenceTransformerEmbeddingProvider, ChromaVectorMemoryStore,
  )
"""
