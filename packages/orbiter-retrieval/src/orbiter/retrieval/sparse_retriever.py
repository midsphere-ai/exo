"""BM25 sparse retriever for keyword-based search.

``SparseRetriever`` builds an inverted index over ``Chunk`` objects and
scores them against a query using the
`BM25 <https://en.wikipedia.org/wiki/Okapi_BM25>`_ ranking function.
Pure Python — no external dependencies beyond stdlib.
"""

from __future__ import annotations

import math
import re
from typing import Any

from orbiter.retrieval.retriever import Retriever  # pyright: ignore[reportMissingImports]
from orbiter.retrieval.types import Chunk, RetrievalResult  # pyright: ignore[reportMissingImports]

# BM25 default parameters (Okapi BM25)
_DEFAULT_K1 = 1.5
_DEFAULT_B = 0.75

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer."""
    return [m.group().lower() for m in _TOKEN_RE.finditer(text)]


class SparseRetriever(Retriever):
    """BM25 sparse retriever for keyword-based search.

    Builds an inverted index over chunks and ranks them using BM25
    scoring.  Entirely pure-Python with no external dependencies.

    Args:
        k1: Term-frequency saturation parameter (default 1.5).
        b: Length normalisation parameter (default 0.75).
    """

    def __init__(self, *, k1: float = _DEFAULT_K1, b: float = _DEFAULT_B) -> None:
        self.k1 = k1
        self.b = b

        # Indexed state
        self._chunks: list[Chunk] = []
        self._doc_token_counts: list[dict[str, int]] = []
        self._doc_lengths: list[int] = []
        self._avg_dl: float = 0.0
        # term -> set of chunk indices containing the term
        self._inverted_index: dict[str, set[int]] = {}

    def index(self, chunks: list[Chunk]) -> None:
        """Build the inverted index over a list of chunks.

        Replaces any previously indexed data.

        Args:
            chunks: The chunks to index.
        """
        self._chunks = list(chunks)
        self._doc_token_counts = []
        self._doc_lengths = []
        self._inverted_index = {}

        total_length = 0
        for idx, chunk in enumerate(self._chunks):
            tokens = _tokenize(chunk.content)
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self._doc_token_counts.append(tf)
            self._doc_lengths.append(len(tokens))
            total_length += len(tokens)

            for term in tf:
                if term not in self._inverted_index:
                    self._inverted_index[term] = set()
                self._inverted_index[term].add(idx)

        n = len(self._chunks)
        self._avg_dl = total_length / n if n > 0 else 0.0

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """Retrieve chunks ranked by BM25 score.

        Args:
            query: The search query text.
            top_k: Maximum number of results to return.
            **kwargs: Unused; accepted for interface compatibility.

        Returns:
            A list of ``RetrievalResult`` objects ranked by BM25 score
            (highest first).  Only chunks with a positive score are
            returned.
        """
        if not self._chunks:
            return []

        query_terms = _tokenize(query)
        if not query_terms:
            return []

        n = len(self._chunks)
        scores: dict[int, float] = {}

        for term in query_terms:
            if term not in self._inverted_index:
                continue

            df = len(self._inverted_index[term])
            # IDF with floor at 0 to avoid negative scores for very common terms
            idf = max(
                math.log((n - df + 0.5) / (df + 0.5) + 1.0),
                0.0,
            )

            for idx in self._inverted_index[term]:
                tf = self._doc_token_counts[idx].get(term, 0)
                dl = self._doc_lengths[idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self._avg_dl)
                score = idf * (tf * (self.k1 + 1.0)) / denom
                scores[idx] = scores.get(idx, 0.0) + score

        # Sort by score descending, only keep positive scores
        ranked = sorted(
            ((idx, s) for idx, s in scores.items() if s > 0),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            RetrievalResult(chunk=self._chunks[idx], score=score)
            for idx, score in ranked[:top_k]
        ]
