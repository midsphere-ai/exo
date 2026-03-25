"""Backward-compatible shim — re-exports from exo.observability.logging.

All new code should import directly from ``exo.observability.logging``.
This module exists solely so that ``from exo.log import get_logger``
continues to work.
"""

from exo.observability.logging import (  # pyright: ignore[reportMissingImports]
    TextFormatter as _Formatter,
)
from exo.observability.logging import (  # pyright: ignore[reportMissingImports]
    configure_logging as configure,
)
from exo.observability.logging import (  # pyright: ignore[reportMissingImports]
    get_logger,
)

_PREFIX = "exo"

__all__ = ["_PREFIX", "_Formatter", "configure", "get_logger"]
