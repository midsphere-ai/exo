# exo.eval.scorers

Rule-based scorers for format validation, schema validation, correctness, length, relevance, and completeness.

## Module Path

```python
from exo.eval.scorers import (
    FormatValidationScorer,
    SchemaValidationScorer,
    OutputCorrectnessScorer,
    OutputLengthScorer,
    OutputRelevanceScorer,
    OutputCompletenessScorer,
)
```

---

## FormatValidationScorer

Validates that output conforms to a specified format. Supports five formats: json, xml, yaml, markdown, csv.

### Constructor

```python
FormatValidationScorer(fmt: str = "json", *, name: str | None = None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fmt` | `str` | `"json"` | Target format to validate against |
| `name` | `str \| None` | `None` | Scorer name (defaults to `"format_{fmt}"`) |

**Raises:** `ValueError` if `fmt` is not one of the supported formats.

### Supported Formats

| Format | Validation Logic |
|---|---|
| `"json"` | Parses with `json.loads()` |
| `"xml"` | Parses with `xml.etree.ElementTree.fromstring()` |
| `"yaml"` | Parses with `yaml.safe_load()`, must produce dict or list. Falls back to regex if `pyyaml` not installed |
| `"markdown"` | Detects headings, lists, links, code fences, blockquotes, or bold markers |
| `"csv"` | Requires header + data row with consistent delimiter counts. Checks `,`, `\t`, `;`, `\|` |

### Methods

#### score()

```python
async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult
```

Returns score `1.0` if format is valid, `0.0` otherwise. Details include `{"format": "<fmt>"}` and any error information.

### Example

```python
import asyncio
from exo.eval import FormatValidationScorer

async def main():
    json_scorer = FormatValidationScorer("json")
    xml_scorer = FormatValidationScorer("xml")
    md_scorer = FormatValidationScorer("markdown")

    r1 = await json_scorer.score("c1", None, '{"key": "value"}')
    print(r1.score)  # 1.0

    r2 = await json_scorer.score("c2", None, "not json")
    print(r2.score)  # 0.0

    r3 = await xml_scorer.score("c3", None, "<root><item>text</item></root>")
    print(r3.score)  # 1.0

    r4 = await md_scorer.score("c4", None, "# Hello\n\nSome **bold** text")
    print(r4.score)  # 1.0

asyncio.run(main())
```

---

## SchemaValidationScorer

Validates JSON output against a JSON Schema. Performs minimal validation: type checking, required fields, and recursive property validation.

### Constructor

```python
SchemaValidationScorer(schema: dict[str, Any], *, name: str = "schema")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `schema` | `dict[str, Any]` | *(required)* | JSON Schema to validate against |
| `name` | `str` | `"schema"` | Scorer name |

### Methods

#### score()

```python
async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult
```

Returns `1.0` if the output is valid JSON matching the schema, `0.0` otherwise. Details include an `"errors"` list when validation fails.

### Supported Schema Keywords

| Keyword | Description |
|---|---|
| `type` | Checks `object`, `array`, `string`, `number`, `integer`, `boolean` |
| `required` | Checks for presence of required fields in objects |
| `properties` | Recursively validates nested properties |

### Example

```python
import asyncio
from exo.eval import SchemaValidationScorer

async def main():
    schema = {
        "type": "object",
        "required": ["name", "age"],
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }
    scorer = SchemaValidationScorer(schema)

    r1 = await scorer.score("c1", None, '{"name": "Alice", "age": 30}')
    print(r1.score)  # 1.0

    r2 = await scorer.score("c2", None, '{"name": "Bob"}')
    print(r2.score)  # 0.0
    print(r2.details)  # {'errors': ["Missing required field: 'age'"]}

asyncio.run(main())
```

---

## OutputCorrectnessScorer

Checks output against a ground truth string or keyword list.

### Constructor

```python
OutputCorrectnessScorer(
    *,
    ground_truth: str | None = None,
    keywords: list[str] | None = None,
    normalize: bool = True,
    name: str = "correctness",
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ground_truth` | `str \| None` | `None` | Expected output for exact matching |
| `keywords` | `list[str] \| None` | `None` | Keywords to check for presence |
| `normalize` | `bool` | `True` | Normalize whitespace and case for ground truth comparison |
| `name` | `str` | `"correctness"` | Scorer name |

### Methods

#### score()

```python
async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult
```

**With `ground_truth`:** Returns `1.0` for exact match (after optional normalization), `0.0` otherwise. Details include `{"match": bool}`.

**With `keywords`:** Returns fraction of keywords found (case-insensitive substring match). Details include `{"found": [...], "missing": [...]}`.

**With neither:** Returns `0.0` with an error.

### Example

```python
import asyncio
from exo.eval import OutputCorrectnessScorer

async def main():
    # Exact match
    exact = OutputCorrectnessScorer(ground_truth="Hello World")
    r1 = await exact.score("c1", None, "  hello   world  ")
    print(r1.score)  # 1.0 (normalized match)

    # Keyword check
    kw = OutputCorrectnessScorer(keywords=["Python", "machine learning", "AI"])
    r2 = await kw.score("c2", None, "Python is great for AI applications")
    print(r2.score)  # 0.667 (2 of 3 keywords found)
    print(r2.details["missing"])  # ['machine learning']

asyncio.run(main())
```

---

## OutputLengthScorer

Scores based on output length being within a specified range.

### Constructor

```python
OutputLengthScorer(
    *,
    min_length: int = 1,
    max_length: int = 10_000,
    name: str = "length",
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `min_length` | `int` | `1` | Minimum allowed character count |
| `max_length` | `int` | `10_000` | Maximum allowed character count |
| `name` | `str` | `"length"` | Scorer name |

### Methods

#### score()

```python
async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult
```

Returns `1.0` if `min_length <= len(output) <= max_length`, else `0.0`. Details include `{"length": int, "min": int, "max": int}`.

### Example

```python
import asyncio
from exo.eval import OutputLengthScorer

async def main():
    scorer = OutputLengthScorer(min_length=10, max_length=100)

    r1 = await scorer.score("c1", None, "This is a valid length response.")
    print(r1.score)  # 1.0

    r2 = await scorer.score("c2", None, "Short")
    print(r2.score)  # 0.0
    print(r2.details)  # {'length': 5, 'min': 10, 'max': 100}

asyncio.run(main())
```

---

## OutputRelevanceScorer

Keyword-overlap relevance between input and output. Computes the fraction of input words that appear in the output.

### Constructor

```python
OutputRelevanceScorer(*, name: str = "relevance")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `"relevance"` | Scorer name |

### Methods

#### score()

```python
async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult
```

Score = `min(overlap_count / input_word_count, 1.0)`. Uses case-insensitive word splitting. Returns `0.0` if input is empty.

Details include `{"overlap": int, "input_words": int}`.

### Example

```python
import asyncio
from exo.eval import OutputRelevanceScorer

async def main():
    scorer = OutputRelevanceScorer()

    r = await scorer.score(
        "c1",
        "What is Python programming?",
        "Python is a popular programming language used for many tasks.",
    )
    print(f"Score: {r.score:.2f}")
    print(r.details)  # {'overlap': 3, 'input_words': 4}

asyncio.run(main())
```

---

## OutputCompletenessScorer

Checks that output covers all required sections (case-insensitive substring match).

### Constructor

```python
OutputCompletenessScorer(required_sections: list[str], *, name: str = "completeness")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `required_sections` | `list[str]` | *(required)* | Sections that must appear in the output |
| `name` | `str` | `"completeness"` | Scorer name |

### Methods

#### score()

```python
async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult
```

Score = fraction of required sections found. Details include `{"found": [...], "missing": [...]}`.

### Example

```python
import asyncio
from exo.eval import OutputCompletenessScorer

async def main():
    scorer = OutputCompletenessScorer(
        required_sections=["introduction", "methodology", "results", "conclusion"]
    )

    output = """
    # Introduction
    This study examines...
    # Methodology
    We used a survey approach...
    # Results
    The findings show...
    """

    r = await scorer.score("c1", None, output)
    print(r.score)  # 0.75 (3 of 4 sections found)
    print(r.details["missing"])  # ['conclusion']

asyncio.run(main())
```
