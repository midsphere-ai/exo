"""Sandboxed Python code execution service.

Executes user-provided Python code in an isolated subprocess with:
- Restricted imports (allowlist only)
- CPU timeout
- Workspace-scoped filesystem access
- Stdout/stderr capture
- Generated file collection (images, CSVs, etc.)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


@dataclass
class SandboxConfig:
    """Configuration for a sandbox execution."""

    allowed_libraries: list[str] = field(
        default_factory=lambda: [
            "pandas",
            "numpy",
            "matplotlib",
            "json",
            "csv",
            "math",
            "statistics",
            "collections",
            "itertools",
            "functools",
            "re",
            "datetime",
            "io",
            "os.path",
            "pathlib",
        ]
    )
    timeout_seconds: int = 30
    memory_limit_mb: int = 256


@dataclass
class SandboxResult:
    """Result of a sandbox execution."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    generated_files: list[dict[str, Any]] = field(default_factory=list)
    execution_time_ms: float = 0.0


# Builtins blocked in the sandbox. exec/eval/compile are NOT blocked because
# the import system's exec_module() needs exec internally. The real security
# boundary is the subprocess isolation + import restriction + filesystem restriction.
_BLOCKED_BUILTINS = {
    "breakpoint",
    "exit",
    "quit",
}

# Max file size to include as base64 in results (1 MB)
_MAX_FILE_BYTES = 1_048_576


def _build_runner_script(
    code: str,
    workspace_dir: str,
    allowed_libraries: list[str],
) -> str:
    """Build a self-contained Python script that runs user code in a restricted env."""
    allowed_json = json.dumps(allowed_libraries)
    # Use repr() to safely serialize the user code — handles \r, \t, \0, null bytes,
    # embedded quotes, triple-quotes, and all other special characters without injection.
    code_repr = repr(code)

    return textwrap.dedent(f"""\
        import builtins as _builtins
        import importlib
        import json
        import os
        import sys

        # -- Restrict filesystem to workspace --
        os.chdir({workspace_dir!r})

        # -- Import restriction --
        # Only modules in the explicit allowlist + a minimal set of safe stdlib
        # modules needed for basic operations are permitted.
        _SAFE_STDLIB = {{
            "builtins", "_thread", "abc", "codecs", "encodings",
            "errno", "importlib", "io", "marshal", "posixpath",
            "sys", "types", "warnings", "_warnings", "_io",
            "_signal", "_abc", "_codecs", "_collections_abc",
            "_frozen_importlib", "_frozen_importlib_external",
            "_weakref", "_weakrefset", "genericpath", "posix",
            "stat", "_stat", "os", "os.path", "_collections",
        }}
        _USER_LIBS = set({allowed_json})
        _ALLOWED = _USER_LIBS | _SAFE_STDLIB
        _USER_LIBS_STR = ", ".join(sorted(_USER_LIBS))
        _original_import = _builtins.__import__

        def _restricted_import(name, *args, **kwargs):
            top = name.split(".")[0]
            if top not in _ALLOWED:
                raise ImportError(
                    "Import of '" + name + "' is not allowed. "
                    "Allowed libraries: " + _USER_LIBS_STR
                )
            return _original_import(name, *args, **kwargs)

        _builtins.__import__ = _restricted_import

        # -- Block dangerous builtins --
        for _name in {_BLOCKED_BUILTINS!r}:
            if hasattr(_builtins, _name):
                delattr(_builtins, _name)

        # -- Restrict open() to workspace only --
        _workspace = os.path.realpath({workspace_dir!r})
        _original_open = _builtins.open

        def _safe_open(file, mode="r", *args, **kwargs):
            path = os.path.realpath(os.path.join(_workspace, str(file)))
            if not path.startswith(_workspace):
                raise PermissionError(f"Access denied: {{file}} is outside workspace")
            return _original_open(path, mode, *args, **kwargs)

        _builtins.open = _safe_open

        # -- Configure matplotlib for non-interactive backend --
        try:
            import matplotlib
            matplotlib.use("Agg")
        except ImportError:
            pass

        # -- Execute user code --
        _user_code = {code_repr}
        exec(compile(_user_code, "<sandbox>", "exec"))
    """)


def _collect_files(workspace: str) -> list[dict[str, Any]]:
    """Collect generated files from workspace with base64-encoded content."""
    files: list[dict[str, Any]] = []
    for entry in Path(workspace).rglob("*"):
        if not entry.is_file() or entry.name == "_runner.py":
            continue
        rel = str(entry.relative_to(workspace))
        size = entry.stat().st_size
        file_info: dict[str, Any] = {
            "name": rel,
            "size_bytes": size,
        }
        # Include base64 content for reasonably-sized files
        if size <= _MAX_FILE_BYTES:
            file_info["content_base64"] = base64.b64encode(entry.read_bytes()).decode("ascii")
        files.append(file_info)
    return files


def execute_code(
    code: str,
    config: SandboxConfig | None = None,
) -> SandboxResult:
    """Execute Python code in an isolated subprocess.

    Creates a temporary workspace directory, runs the code with restrictions,
    captures output, and collects any generated files.
    """
    if config is None:
        config = SandboxConfig()

    _log.debug("execute_code: code_len=%d timeout=%ds", len(code), config.timeout_seconds)
    start = time.monotonic()
    workspace = tempfile.mkdtemp(prefix="exo_sandbox_")

    try:
        runner_script = _build_runner_script(code, workspace, config.allowed_libraries)

        # Write runner to a temp file in the workspace
        script_path = os.path.join(workspace, "_runner.py")
        with open(script_path, "w") as f:
            f.write(runner_script)

        # Build subprocess environment — inherit PATH for library access
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": workspace,
            "TMPDIR": workspace,
            "PYTHONPATH": "",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

        # Execute in subprocess with timeout
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=config.timeout_seconds,
                cwd=workspace,
                env=env,
            )
        except subprocess.TimeoutExpired:
            elapsed = (time.monotonic() - start) * 1000
            _log.warning("execute_code: timed out after %ds", config.timeout_seconds)
            return SandboxResult(
                success=False,
                error=f"Execution timed out after {config.timeout_seconds}s",
                execution_time_ms=elapsed,
            )

        elapsed = (time.monotonic() - start) * 1000

        # Collect generated files before workspace cleanup
        generated_files = _collect_files(workspace)

        success = result.returncode == 0
        error_msg = None
        if not success and result.stderr:
            lines = result.stderr.strip().splitlines()
            error_msg = lines[-1] if lines else "Unknown error"

        return SandboxResult(
            success=success,
            stdout=result.stdout,
            stderr=result.stderr,
            error=error_msg,
            generated_files=generated_files,
            execution_time_ms=round(elapsed, 2),
        )

    finally:
        shutil.rmtree(workspace, ignore_errors=True)
