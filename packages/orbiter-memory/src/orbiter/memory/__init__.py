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
from orbiter.memory.dedup import (  # pyright: ignore[reportMissingImports]
    MemUpdateChecker,
    MergeResult,
    UpdateDecision,
)
from orbiter.memory.encrypted import (  # pyright: ignore[reportMissingImports]
    EncryptedMemoryStore,
)
from orbiter.memory.evolution import (  # pyright: ignore[reportMissingImports]
    MemoryEvolutionStrategy,
)
from orbiter.memory.evolution.ace import (  # pyright: ignore[reportMissingImports]
    ACEStrategy,
)
from orbiter.memory.evolution.reasoning_bank import (  # pyright: ignore[reportMissingImports]
    ReasoningBankStrategy,
)
from orbiter.memory.evolution.reme import (  # pyright: ignore[reportMissingImports]
    ReMeStrategy,
)
from orbiter.memory.migrations import (  # pyright: ignore[reportMissingImports]
    Migration,
    MigrationRegistry,
)
from orbiter.memory.search import (  # pyright: ignore[reportMissingImports]
    SearchManager,
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
    "ACEStrategy",
    "AIMemory",
    "AgentMemory",
    "Embeddings",
    "EncryptedMemoryStore",
    "ExtractionTask",
    "ExtractionType",
    "Extractor",
    "HumanMemory",
    "LongTermMemory",
    "MEMORY_ADDED",
    "MEMORY_CLEARED",
    "MEMORY_SEARCHED",
    "MemUpdateChecker",
    "MemoryError",
    "MemoryEventEmitter",
    "MemoryEvolutionStrategy",
    "MemoryItem",
    "MemoryMetadata",
    "MemoryOrchestrator",
    "MemoryPersistence",
    "MemoryStatus",
    "MergeResult",
    "Migration",
    "MigrationRegistry",
    "MemoryStore",
    "OpenAIEmbeddings",
    "OrchestratorConfig",
    "ReMeStrategy",
    "ReasoningBankStrategy",
    "SearchManager",
    "ShortTermMemory",
    "Summarizer",
    "SummaryConfig",
    "SummaryResult",
    "SummaryTemplate",
    "SystemMemory",
    "TaskStatus",
    "ToolMemory",
    "UpdateDecision",
    "VertexEmbeddings",
    "VectorMemoryStore",
    "check_trigger",
    "generate_summary",
]
