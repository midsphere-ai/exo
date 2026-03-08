"""Context window sizes for well-known LLM models."""

MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "o1": 200000,
    "claude-sonnet-4-6": 200000,
    "claude-opus-4-6": 200000,
    "claude-haiku-4-5-20251001": 200000,
    "gemini-2.0-flash": 1048576,
    "gemini-1.5-pro": 2097152,
    "zai-org/glm-5-maas": 128000,
}
