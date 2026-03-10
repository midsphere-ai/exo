"""Orbiter Memory: Pluggable memory backends."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from orbiter.memory.base import (  # pyright: ignore[reportMissingImports]
    AIMemory,
    HumanMemory,
    MemoryCategory,
    MemoryError,
    MemoryItem,
    MemoryMetadata,
    MemoryStatus,
    MemoryStore,
    SystemMemory,
    ToolMemory,
)
from orbiter.memory.events import (  # pyright: ignore[reportMissingImports]
    MEMORY_ADDED,
    MEMORY_CLEARED,
    MEMORY_SEARCHED,
    MemoryEventEmitter,
)
from orbiter.memory.long_term import (  # pyright: ignore[reportMissingImports]
    ExtractionTask,
    ExtractionType,
    Extractor,
    LongTermMemory,
    MemoryOrchestrator,
    OrchestratorConfig,
    TaskStatus,
)
from orbiter.memory.short_term import (  # pyright: ignore[reportMissingImports]
    ShortTermMemory,
)
from orbiter.memory.encrypted import (  # pyright: ignore[reportMissingImports]
    EncryptedMemoryStore,
    derive_key,
)
from orbiter.memory.summary import (  # pyright: ignore[reportMissingImports]
    Summarizer,
    SummaryConfig,
    SummaryResult,
    SummaryTemplate,
    check_trigger,
    generate_summary,
)

__all__ = [
    # encryption
    "EncryptedMemoryStore",
    "derive_key",
    # events
    "MEMORY_ADDED",
    "MEMORY_CLEARED",
    "MEMORY_SEARCHED",
    # base types
    "AIMemory",
    "MemoryCategory",
    # long-term
    "ExtractionTask",
    "ExtractionType",
    "Extractor",
    "HumanMemory",
    "LongTermMemory",
    "MemoryError",
    "MemoryEventEmitter",
    "MemoryItem",
    "MemoryMetadata",
    "MemoryOrchestrator",
    "MemoryStatus",
    "MemoryStore",
    "OrchestratorConfig",
    # short-term
    "ShortTermMemory",
    "Summarizer",
    # summary
    "SummaryConfig",
    "SummaryResult",
    "SummaryTemplate",
    "SystemMemory",
    "TaskStatus",
    "ToolMemory",
    "check_trigger",
    "generate_summary",
]
