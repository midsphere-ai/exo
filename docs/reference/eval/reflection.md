# exo.eval.reflection

Reflection framework with LLM-powered analysis, insight extraction, and suggestion generation. Provides a three-step pipeline: analyze, insight, suggest.

## Module Path

```python
from exo.eval.reflection import (
    ReflectionType,
    ReflectionLevel,
    ReflectionResult,
    ReflectionHistory,
    Reflector,
    GeneralReflector,
)
```

---

## ReflectionType

Category of a reflection result.

```python
class ReflectionType(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    OPTIMIZATION = "optimization"
    PATTERN = "pattern"
    INSIGHT = "insight"
```

| Value | Description |
|---|---|
| `SUCCESS` | Reflection on a successful outcome |
| `FAILURE` | Reflection on a failed outcome |
| `OPTIMIZATION` | Opportunities for improvement |
| `PATTERN` | Recurring patterns detected |
| `INSIGHT` | General insights and learnings |

---

## ReflectionLevel

Depth of reflection analysis.

```python
class ReflectionLevel(StrEnum):
    SHALLOW = "shallow"
    MEDIUM = "medium"
    DEEP = "deep"
    META = "meta"
```

| Value | Description |
|---|---|
| `SHALLOW` | Surface-level observation |
| `MEDIUM` | Standard analysis depth |
| `DEEP` | In-depth root cause analysis |
| `META` | Meta-level reflection on the reflection process itself |

---

## ReflectionResult

Output of a single reflection pass.

**Decorator:** `@dataclass(frozen=True, slots=True)`

### Constructor

```python
ReflectionResult(
    reflection_type: ReflectionType,
    level: ReflectionLevel,
    summary: str,
    key_findings: list[str] = [],
    root_causes: list[str] = [],
    insights: list[str] = [],
    suggestions: list[str] = [],
    metadata: dict[str, Any] = {},
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `reflection_type` | `ReflectionType` | *(required)* | Category of this reflection |
| `level` | `ReflectionLevel` | *(required)* | Depth of analysis |
| `summary` | `str` | *(required)* | One-sentence overview |
| `key_findings` | `list[str]` | `[]` | Notable observations |
| `root_causes` | `list[str]` | `[]` | Underlying reasons for outcomes |
| `insights` | `list[str]` | `[]` | Patterns and learnings |
| `suggestions` | `list[str]` | `[]` | Actionable improvements |
| `metadata` | `dict[str, Any]` | `{}` | Additional context |

---

## ReflectionHistory

Tracks reflection results over time with aggregate statistics.

**Decorator:** `@dataclass(slots=True)`

### Constructor

```python
ReflectionHistory()
```

### Attributes

| Attribute | Type | Default | Description |
|---|---|---|---|
| `total_count` | `int` | `0` | Total reflections recorded |
| `success_count` | `int` | `0` | Count of `SUCCESS` type reflections |
| `failure_count` | `int` | `0` | Count of `FAILURE` type reflections |

### Methods

#### add()

```python
def add(self, result: ReflectionResult) -> None
```

Append a result and update counters. Increments `success_count` for `ReflectionType.SUCCESS` and `failure_count` for `ReflectionType.FAILURE`.

#### get_recent()

```python
def get_recent(self, n: int = 5) -> list[ReflectionResult]
```

Return the `n` most recent reflections.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n` | `int` | `5` | Number of recent entries to return |

#### get_by_type()

```python
def get_by_type(self, rtype: ReflectionType) -> list[ReflectionResult]
```

Return all reflections matching the given type.

| Parameter | Type | Description |
|---|---|---|
| `rtype` | `ReflectionType` | Type to filter by |

#### summarize()

```python
def summarize(self) -> dict[str, Any]
```

Return aggregate statistics.

**Returns:** Dict with keys:

| Key | Type | Description |
|---|---|---|
| `total` | `int` | Total reflections |
| `success` | `int` | Success count |
| `failure` | `int` | Failure count |
| `types` | `dict[str, int]` | Count per `ReflectionType` value |

---

## Reflector (ABC)

Abstract reflector with a three-step template: analyze, insight, suggest.

### Constructor

```python
Reflector(
    name: str = "reflector",
    reflection_type: ReflectionType = ReflectionType.INSIGHT,
    level: ReflectionLevel = ReflectionLevel.MEDIUM,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `"reflector"` | Reflector name |
| `reflection_type` | `ReflectionType` | `INSIGHT` | Default category for results |
| `level` | `ReflectionLevel` | `MEDIUM` | Default analysis depth |

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `name` | `str` | Reflector name |
| `reflection_type` | `ReflectionType` | Default reflection category |
| `level` | `ReflectionLevel` | Default analysis depth |

### Methods

#### reflect()

```python
async def reflect(self, context: dict[str, Any]) -> ReflectionResult
```

Run the full three-step reflection pipeline: `analyze()` -> `insight()` -> `suggest()`.

| Parameter | Type | Description |
|---|---|---|
| `context` | `dict[str, Any]` | Execution context to reflect on |

**Returns:** `ReflectionResult` assembled from the three steps.

**Behavior:**
1. Calls `analyze(context)` to extract facts and key findings
2. Calls `insight(analysis)` to derive insights
3. Calls `suggest(analysis)` to generate actionable suggestions
4. Assembles `ReflectionResult` from the combined outputs

#### analyze() (abstract)

```python
async def analyze(self, context: dict[str, Any]) -> dict[str, Any]
```

Step 1: Extract facts and key findings from the execution context.

**Returns:** Dict with keys like `"summary"`, `"key_findings"`, `"root_causes"`, `"insights"`, `"suggestions"`.

#### insight()

```python
async def insight(self, analysis: dict[str, Any]) -> dict[str, Any]
```

Step 2: Derive insights from the analysis. Default implementation passes through `analysis["insights"]`.

#### suggest()

```python
async def suggest(self, insights: dict[str, Any]) -> dict[str, Any]
```

Step 3: Generate actionable suggestions from insights. Default implementation passes through `insights["suggestions"]`.

---

## GeneralReflector

LLM-powered reflector using a judge callable `(prompt: str) -> str`.

### Constructor

```python
GeneralReflector(
    judge: Any = None,
    *,
    system_prompt: str | None = None,
    name: str = "general_reflector",
    reflection_type: ReflectionType = ReflectionType.INSIGHT,
    level: ReflectionLevel = ReflectionLevel.DEEP,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `judge` | `Any` | `None` | Async callable `(prompt: str) -> str` |
| `system_prompt` | `str \| None` | `None` | Custom system prompt (uses default if `None`) |
| `name` | `str` | `"general_reflector"` | Reflector name |
| `reflection_type` | `ReflectionType` | `INSIGHT` | Default reflection category |
| `level` | `ReflectionLevel` | `DEEP` | Default analysis depth |

### Default System Prompt

The built-in system prompt instructs the LLM to provide a structured reflection with five sections (Summary, Key Findings, Root Causes, Insights, Suggestions) and return JSON.

### Context Keys

The `analyze()` method reads these keys from the context dict:

| Key | Used For |
|---|---|
| `input` | Formatted as `[Input]` section |
| `output` | Formatted as `[Output]` section |
| `error` | Formatted as `[Error]` section |
| `iteration` | Formatted as `[Iteration]` marker |

### Overridden Methods

#### analyze()

Calls the LLM judge with the execution context and parses the response. Returns parsed JSON or a fallback dict with `"parse_error": True`.

#### insight()

Passes through insights from the LLM analysis result.

#### suggest()

Passes through suggestions from the LLM analysis result (already extracted in `analyze`).

### Example

```python
import asyncio
from exo.eval import (
    GeneralReflector,
    ReflectionHistory,
    ReflectionType,
)

async def my_judge(prompt: str) -> str:
    return '''{
        "summary": "The agent failed to handle edge cases in the input.",
        "key_findings": ["Missing null check", "No retry logic"],
        "root_causes": ["Insufficient error handling"],
        "insights": ["Edge cases need explicit handling"],
        "suggestions": ["Add input validation", "Implement retry with backoff"]
    }'''

async def main():
    reflector = GeneralReflector(
        judge=my_judge,
        reflection_type=ReflectionType.FAILURE,
    )

    context = {
        "input": "Process this data: null",
        "output": "Error: NoneType has no attribute 'strip'",
        "error": "AttributeError",
        "iteration": 3,
    }

    result = await reflector.reflect(context)
    print(f"Type: {result.reflection_type}")  # failure
    print(f"Summary: {result.summary}")
    print(f"Suggestions: {result.suggestions}")

    # Track in history
    history = ReflectionHistory()
    history.add(result)
    print(history.summarize())
    # {'total': 1, 'success': 0, 'failure': 1, 'types': {...}}

asyncio.run(main())
```
