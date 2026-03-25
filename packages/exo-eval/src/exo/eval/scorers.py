"""Rule-based scorers for format validation, correctness, and quality."""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar

from exo.eval.base import Scorer, ScorerResult  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# FormatValidationScorer
# ---------------------------------------------------------------------------


class FormatValidationScorer(Scorer):
    """Validates that output conforms to a specified format.

    Supported formats: json, xml, yaml, markdown, csv.
    """

    __slots__ = ("_format", "_name")

    _VALIDATORS: ClassVar[dict[str, Any]] = {}

    def __init__(self, fmt: str = "json", *, name: str | None = None) -> None:
        fmt = fmt.lower()
        if fmt not in self._VALIDATORS:
            msg = f"Unsupported format: {fmt!r}. Choose from: {sorted(self._VALIDATORS)}"
            raise ValueError(msg)
        self._format = fmt
        self._name = name or f"format_{fmt}"

    async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult:
        text = str(output) if output is not None else ""
        ok, detail = self._VALIDATORS[self._format](text)
        return ScorerResult(
            scorer_name=self._name,
            score=1.0 if ok else 0.0,
            details={"format": self._format, **detail},
        )

    @staticmethod
    def _check_json(text: str) -> tuple[bool, dict[str, Any]]:
        try:
            json.loads(text)
            return True, {}
        except (json.JSONDecodeError, ValueError) as exc:
            return False, {"error": str(exc)}

    @staticmethod
    def _check_xml(text: str) -> tuple[bool, dict[str, Any]]:
        # Lightweight: check for matching root-level open/close tags
        text = text.strip()
        if not text.startswith("<") or not text.endswith(">"):
            return False, {"error": "Does not start/end with angle brackets"}
        # Try stdlib xml parser
        import xml.etree.ElementTree as ET

        try:
            ET.fromstring(text)
            return True, {}
        except ET.ParseError as exc:
            return False, {"error": str(exc)}

    @staticmethod
    def _check_yaml(text: str) -> tuple[bool, dict[str, Any]]:
        # YAML is a superset of JSON — any string is technically valid YAML.
        # We check for structured content: must parse to dict/list.
        try:
            import yaml  # type: ignore[import-untyped]

            result = yaml.safe_load(text)
            if isinstance(result, (dict, list)):
                return True, {}
            return False, {"error": "Parsed as scalar, not structured data"}
        except ImportError:
            # Fallback: check for key: value patterns
            if re.search(r"^\s*[\w-]+\s*:", text, re.MULTILINE):
                return True, {}
            return False, {"error": "No YAML-like structure detected (pyyaml not installed)"}
        except Exception as exc:
            return False, {"error": str(exc)}

    @staticmethod
    def _check_markdown(text: str) -> tuple[bool, dict[str, Any]]:
        # Check for at least one markdown element: heading, list, link, code block
        patterns = [
            r"^#{1,6}\s",  # headings
            r"^\s*[-*+]\s",  # unordered list
            r"^\s*\d+\.\s",  # ordered list
            r"\[.+?\]\(.+?\)",  # links
            r"```",  # code fences
            r"^\s*>\s",  # blockquotes
            r"\*\*.+?\*\*",  # bold
        ]
        for pat in patterns:
            if re.search(pat, text, re.MULTILINE):
                return True, {}
        return False, {"error": "No markdown elements detected"}

    @staticmethod
    def _check_csv(text: str) -> tuple[bool, dict[str, Any]]:
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        if len(lines) < 2:
            return False, {"error": "CSV requires at least a header and one data row"}
        delimiters = [",", "\t", ";", "|"]
        for delim in delimiters:
            counts = [ln.count(delim) for ln in lines]
            if counts[0] > 0 and all(c == counts[0] for c in counts):
                return True, {"delimiter": delim}
        return False, {"error": "Inconsistent column counts across rows"}


FormatValidationScorer._VALIDATORS = {
    "json": FormatValidationScorer._check_json,
    "xml": FormatValidationScorer._check_xml,
    "yaml": FormatValidationScorer._check_yaml,
    "markdown": FormatValidationScorer._check_markdown,
    "csv": FormatValidationScorer._check_csv,
}


# ---------------------------------------------------------------------------
# SchemaValidationScorer
# ---------------------------------------------------------------------------


class SchemaValidationScorer(Scorer):
    """Validates JSON output against a JSON Schema."""

    __slots__ = ("_name", "_schema")

    def __init__(self, schema: dict[str, Any], *, name: str = "schema") -> None:
        self._schema = schema
        self._name = name

    async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult:
        text = str(output) if output is not None else ""
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            return ScorerResult(
                scorer_name=self._name,
                score=0.0,
                details={"error": f"Invalid JSON: {exc}"},
            )
        errors = self._validate(data, self._schema)
        return ScorerResult(
            scorer_name=self._name,
            score=1.0 if not errors else 0.0,
            details={"errors": errors} if errors else {},
        )

    @staticmethod
    def _validate(data: Any, schema: dict[str, Any]) -> list[str]:
        """Minimal JSON Schema validation (type + required + properties)."""
        errors: list[str] = []
        expected_type = schema.get("type")
        if expected_type:
            type_map: dict[str, type | tuple[type, ...]] = {
                "object": dict,
                "array": list,
                "string": str,
                "number": (int, float),
                "integer": int,
                "boolean": bool,
            }
            if expected_type in type_map and not isinstance(data, type_map[expected_type]):
                errors.append(f"Expected type {expected_type}, got {type(data).__name__}")
                return errors

        if isinstance(data, dict):
            for req in schema.get("required", []):
                if req not in data:
                    errors.append(f"Missing required field: {req!r}")
            props = schema.get("properties", {})
            for key, sub_schema in props.items():
                if key in data:
                    errors.extend(
                        f"{key}.{e}"
                        for e in SchemaValidationScorer._validate(data[key], sub_schema)
                    )
        return errors


# ---------------------------------------------------------------------------
# OutputCorrectnessScorer
# ---------------------------------------------------------------------------


class OutputCorrectnessScorer(Scorer):
    """Checks output against a ground truth or keyword list.

    Provide *ground_truth* for exact/normalized match, or *keywords* for
    keyword presence checking.
    """

    __slots__ = ("_ground_truth", "_keywords", "_name", "_normalize")

    def __init__(
        self,
        *,
        ground_truth: str | None = None,
        keywords: list[str] | None = None,
        normalize: bool = True,
        name: str = "correctness",
    ) -> None:
        self._ground_truth = ground_truth
        self._keywords = keywords or []
        self._normalize = normalize
        self._name = name

    async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult:
        text = str(output) if output is not None else ""
        details: dict[str, Any] = {}

        if self._ground_truth is not None:
            a = self._norm(text) if self._normalize else text
            b = self._norm(self._ground_truth) if self._normalize else self._ground_truth
            match = a == b
            details["match"] = match
            return ScorerResult(
                scorer_name=self._name, score=1.0 if match else 0.0, details=details
            )

        if self._keywords:
            lower = text.lower()
            found = [kw for kw in self._keywords if kw.lower() in lower]
            ratio = len(found) / len(self._keywords)
            details["found"] = found
            details["missing"] = [kw for kw in self._keywords if kw.lower() not in lower]
            return ScorerResult(scorer_name=self._name, score=ratio, details=details)

        return ScorerResult(
            scorer_name=self._name, score=0.0, details={"error": "No ground_truth or keywords"}
        )

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join(text.lower().split())


# ---------------------------------------------------------------------------
# OutputLengthScorer
# ---------------------------------------------------------------------------


class OutputLengthScorer(Scorer):
    """Scores based on output length being within [min_length, max_length]."""

    __slots__ = ("_max_length", "_min_length", "_name")

    def __init__(
        self,
        *,
        min_length: int = 1,
        max_length: int = 10_000,
        name: str = "length",
    ) -> None:
        self._min_length = min_length
        self._max_length = max_length
        self._name = name

    async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult:
        text = str(output) if output is not None else ""
        length = len(text)
        ok = self._min_length <= length <= self._max_length
        return ScorerResult(
            scorer_name=self._name,
            score=1.0 if ok else 0.0,
            details={"length": length, "min": self._min_length, "max": self._max_length},
        )


# ---------------------------------------------------------------------------
# OutputRelevanceScorer
# ---------------------------------------------------------------------------


class OutputRelevanceScorer(Scorer):
    """Keyword-overlap relevance between input and output."""

    __slots__ = ("_name",)

    def __init__(self, *, name: str = "relevance") -> None:
        self._name = name

    async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult:
        in_words = set(str(input).lower().split()) if input else set()
        out_words = set(str(output).lower().split()) if output else set()
        if not in_words:
            return ScorerResult(scorer_name=self._name, score=0.0, details={"overlap": 0})
        overlap = len(in_words & out_words)
        ratio = overlap / len(in_words)
        return ScorerResult(
            scorer_name=self._name,
            score=min(ratio, 1.0),
            details={"overlap": overlap, "input_words": len(in_words)},
        )


# ---------------------------------------------------------------------------
# OutputCompletenessScorer
# ---------------------------------------------------------------------------


class OutputCompletenessScorer(Scorer):
    """Checks that output covers all *required_sections* (substring match)."""

    __slots__ = ("_name", "_sections")

    def __init__(self, required_sections: list[str], *, name: str = "completeness") -> None:
        self._sections = required_sections
        self._name = name

    async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult:
        text = str(output).lower() if output is not None else ""
        found = [s for s in self._sections if s.lower() in text]
        missing = [s for s in self._sections if s.lower() not in text]
        ratio = len(found) / len(self._sections) if self._sections else 0.0
        return ScorerResult(
            scorer_name=self._name,
            score=ratio,
            details={"found": found, "missing": missing},
        )
