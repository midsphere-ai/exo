"""Orbiter Memory: Pluggable memory backends."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from orbiter.memory.backends.vector import (  # pyright: ignore[reportMissingImports]
    Embeddings,
    OpenAIEmbeddings,
    VectorMemoryStore,
    VertexEmbeddings,
)
from orbiter.memory.base import (  # pyright: ignore[reportMissingImports]
    AgentMemory,
    AIMemory,
    HumanMemory,
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
from orbiter.memory.persistence import (  # pyright: ignore[reportMissingImports]
    MemoryPersistence,
)
from orbiter.memory.short_term import (  # pyright: ignore[reportMissingImports]
    ShortTermMemory,
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
    "AIMemory",
    "AgentMemory",
    "Embeddings",
    "ExtractionTask",
    "ExtractionType",
    "Extractor",
    "HumanMemory",
    "LongTermMemory",
    "MEMORY_ADDED",
    "MEMORY_CLEARED",
    "MEMORY_SEARCHED",
    "MemoryError",
    "MemoryEventEmitter",
    "MemoryItem",
    "MemoryMetadata",
    "MemoryOrchestrator",
    "MemoryPersistence",
    "MemoryStatus",
    "MemoryStore",
    "OpenAIEmbeddings",
    "OrchestratorConfig",
    "ShortTermMemory",
    "Summarizer",
    "SummaryConfig",
    "SummaryResult",
    "SummaryTemplate",
    "SystemMemory",
    "TaskStatus",
    "ToolMemory",
    "VertexEmbeddings",
    "VectorMemoryStore",
    "check_trigger",
    "generate_summary",
]
