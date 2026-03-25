"""Knowledge graph triple extraction from text chunks.

``TripleExtractor`` uses an LLM to extract subject-predicate-object triples
from ``Chunk`` objects, producing structured knowledge graph data.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Triple:
    """An immutable subject-predicate-object triple for knowledge graphs.

    Attributes:
        subject: The subject entity.
        predicate: The relationship or predicate.
        object: The object entity.
        confidence: Confidence score between 0 and 1.
        source_chunk_id: Identifier linking back to the originating chunk
            (formatted as ``document_id:index``).
    """

    subject: str
    predicate: str
    object: str
    confidence: float
    source_chunk_id: str


_DEFAULT_PROMPT = """You are a knowledge graph extraction engine. Extract subject-predicate-object triples from the following text.

Text:
{text}

Return a JSON array of objects, each with keys: "subject", "predicate", "object", "confidence" (0-1 float).
For example:
[
  {{"subject": "Python", "predicate": "is", "object": "programming language", "confidence": 0.95}},
  {{"subject": "Guido van Rossum", "predicate": "created", "object": "Python", "confidence": 0.9}}
]

Return ONLY the JSON array, no other text."""


class TripleExtractor:
    """Extracts knowledge graph triples from text chunks via an LLM.

    Uses an LLM provider to identify entities and relationships in text,
    returning structured ``Triple`` objects suitable for knowledge graph
    construction.

    Args:
        model: Model string, e.g. ``"openai:gpt-4o"``.
        prompt_template: Template with ``{text}`` placeholder. Defaults to
            a built-in triple extraction prompt.
        provider_kwargs: Extra keyword arguments forwarded to
            ``get_provider()`` (e.g. ``api_key``, ``base_url``).
    """

    def __init__(
        self,
        model: str,
        *,
        prompt_template: str | None = None,
        **provider_kwargs: Any,
    ) -> None:
        self.model = model
        self.prompt_template = prompt_template or _DEFAULT_PROMPT
        self._provider_kwargs = provider_kwargs

    async def extract(
        self,
        chunks: list[Any],
    ) -> list[Triple]:
        """Extract triples from a list of chunks.

        Each chunk is sent to the LLM individually. Triples are tagged with
        the source chunk identifier (``document_id:index``).

        Args:
            chunks: List of ``Chunk`` objects to extract triples from.

        Returns:
            A flat list of ``Triple`` objects extracted from all chunks.
        """
        if not chunks:
            return []

        from exo.models import get_provider  # pyright: ignore[reportMissingImports]
        from exo.types import UserMessage

        provider = get_provider(self.model, **self._provider_kwargs)
        all_triples: list[Triple] = []

        for chunk in chunks:
            prompt = self.prompt_template.format(text=chunk.content)
            response = await provider.complete([UserMessage(content=prompt)])
            chunk_id = f"{chunk.document_id}:{chunk.index}"
            triples = self._parse_triples(response.content, chunk_id)
            all_triples.extend(triples)

        return all_triples

    @staticmethod
    def _parse_triples(content: str, source_chunk_id: str) -> list[Triple]:
        """Parse the LLM response into Triple objects.

        Falls back to an empty list if parsing fails.

        Args:
            content: Raw LLM response text.
            source_chunk_id: Chunk identifier to attach to each triple.

        Returns:
            A list of parsed ``Triple`` objects.
        """
        # Try to extract a JSON array from the response
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if not match:
            return []

        try:
            data = json.loads(match.group())
        except (json.JSONDecodeError, TypeError):
            return []

        if not isinstance(data, list):
            return []

        triples: list[Triple] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            subject = item.get("subject")
            predicate = item.get("predicate")
            obj = item.get("object")
            confidence = item.get("confidence", 0.5)
            if not (
                isinstance(subject, str) and isinstance(predicate, str) and isinstance(obj, str)
            ):
                continue
            if not isinstance(confidence, (int, float)):
                confidence = 0.5
            confidence = max(0.0, min(1.0, float(confidence)))
            triples.append(
                Triple(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    confidence=confidence,
                    source_chunk_id=source_chunk_id,
                )
            )

        return triples
