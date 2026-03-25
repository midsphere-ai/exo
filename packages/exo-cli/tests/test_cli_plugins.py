"""Tests for exo_cli.plugins — plugin system."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exo_cli.plugins import (
    PluginError,
    PluginHook,
    PluginManager,
    PluginSpec,
    _load_plugin_file,
)

# ---------------------------------------------------------------------------
# PluginHook enum
# ---------------------------------------------------------------------------


class TestPluginHook:
    def test_values(self) -> None:
        assert PluginHook.STARTUP == "startup"
        assert PluginHook.SHUTDOWN == "shutdown"
        assert PluginHook.PRE_RUN == "pre_run"
        assert PluginHook.POST_RUN == "post_run"

    def test_count(self) -> None:
        assert len(PluginHook) == 4


# ---------------------------------------------------------------------------
# PluginSpec
# ---------------------------------------------------------------------------


class TestPluginSpec:
    def test_minimal(self) -> None:
        spec = PluginSpec(name="test")
        assert spec.name == "test"
        assert spec.version == "0.0.0"
        assert spec.description == ""
        assert spec.hooks == {}

    def test_full(self) -> None:
        hook_fn = AsyncMock()
        spec = PluginSpec(
            name="my-plugin",
            hooks={PluginHook.STARTUP: hook_fn},
            version="1.2.3",
            description="A test plugin",
        )
        assert spec.name == "my-plugin"
        assert spec.version == "1.2.3"
        assert spec.description == "A test plugin"
        assert PluginHook.STARTUP in spec.hooks

    def test_hooks_is_copy(self) -> None:
        hook_fn = AsyncMock()
        spec = PluginSpec(name="p", hooks={PluginHook.STARTUP: hook_fn})
        h = spec.hooks
        h[PluginHook.SHUTDOWN] = AsyncMock()
        assert PluginHook.SHUTDOWN not in spec.hooks

    def test_repr(self) -> None:
        spec = PluginSpec(name="foo", version="0.1.0")
        r = repr(spec)
        assert "PluginSpec" in r
        assert "foo" in r
        assert "0.1.0" in r


# ---------------------------------------------------------------------------
# PluginError
# ---------------------------------------------------------------------------


class TestPluginError:
    def test_is_exception(self) -> None:
        exc = PluginError("bad plugin")
        assert isinstance(exc, Exception)
        assert str(exc) == "bad plugin"


# ---------------------------------------------------------------------------
# PluginManager — registration
# ---------------------------------------------------------------------------


class TestPluginManagerRegister:
    def test_register(self) -> None:
        mgr = PluginManager()
        spec = PluginSpec(name="alpha")
        mgr.register(spec)
        assert mgr.get("alpha") is spec
        assert "alpha" in mgr.plugins

    def test_duplicate_raises(self) -> None:
        mgr = PluginManager()
        mgr.register(PluginSpec(name="dup"))
        with pytest.raises(PluginError, match="Duplicate"):
            mgr.register(PluginSpec(name="dup"))

    def test_plugins_is_copy(self) -> None:
        mgr = PluginManager()
        mgr.register(PluginSpec(name="a"))
        p = mgr.plugins
        p["b"] = PluginSpec(name="b")
        assert mgr.get("b") is None

    def test_get_missing(self) -> None:
        mgr = PluginManager()
        assert mgr.get("missing") is None

    def test_multiple_plugins(self) -> None:
        mgr = PluginManager()
        mgr.register(PluginSpec(name="x"))
        mgr.register(PluginSpec(name="y"))
        assert len(mgr.plugins) == 2


# ---------------------------------------------------------------------------
# PluginManager — directory loading
# ---------------------------------------------------------------------------


class TestPluginManagerLoadDirectory:
    def test_load_valid_plugin(self, tmp_path: Path) -> None:
        plugin_file = tmp_path / "hello.py"
        plugin_file.write_text(
            textwrap.dedent("""\
                from exo_cli.plugins import PluginSpec
                plugin = PluginSpec(name="hello", version="1.0.0")
            """)
        )
        mgr = PluginManager()
        count = mgr.load_directory(tmp_path)
        assert count == 1
        assert mgr.get("hello") is not None
        assert mgr.get("hello").version == "1.0.0"  # type: ignore[union-attr]

    def test_skip_underscore_files(self, tmp_path: Path) -> None:
        (tmp_path / "_internal.py").write_text("plugin = 'not a PluginSpec'")
        mgr = PluginManager()
        count = mgr.load_directory(tmp_path)
        assert count == 0

    def test_skip_non_py_files(self, tmp_path: Path) -> None:
        (tmp_path / "readme.txt").write_text("not a plugin")
        mgr = PluginManager()
        count = mgr.load_directory(tmp_path)
        assert count == 0

    def test_skip_no_plugin_attribute(self, tmp_path: Path) -> None:
        (tmp_path / "empty.py").write_text("x = 42")
        mgr = PluginManager()
        count = mgr.load_directory(tmp_path)
        assert count == 0

    def test_skip_non_pluginspec_attribute(self, tmp_path: Path) -> None:
        (tmp_path / "wrong.py").write_text("plugin = 'not a spec'")
        mgr = PluginManager()
        count = mgr.load_directory(tmp_path)
        assert count == 0

    def test_missing_directory(self) -> None:
        mgr = PluginManager()
        with pytest.raises(PluginError, match="not found"):
            mgr.load_directory("/nonexistent/path")

    def test_duplicate_across_files(self, tmp_path: Path) -> None:
        for fname in ("a.py", "b.py"):
            (tmp_path / fname).write_text(
                textwrap.dedent("""\
                    from exo_cli.plugins import PluginSpec
                    plugin = PluginSpec(name="same")
                """)
            )
        mgr = PluginManager()
        with pytest.raises(PluginError, match="Duplicate"):
            mgr.load_directory(tmp_path)

    def test_syntax_error_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "broken.py").write_text("def unclosed(:")
        mgr = PluginManager()
        count = mgr.load_directory(tmp_path)
        assert count == 0

    def test_multiple_plugins(self, tmp_path: Path) -> None:
        for i in range(3):
            (tmp_path / f"p{i}.py").write_text(
                textwrap.dedent(f"""\
                    from exo_cli.plugins import PluginSpec
                    plugin = PluginSpec(name="p{i}")
                """)
            )
        mgr = PluginManager()
        count = mgr.load_directory(tmp_path)
        assert count == 3


# ---------------------------------------------------------------------------
# PluginManager — entry points
# ---------------------------------------------------------------------------


class TestPluginManagerEntrypoints:
    def test_load_entrypoint(self) -> None:
        spec = PluginSpec(name="ep-plugin", version="2.0.0")
        mock_ep = MagicMock()
        mock_ep.name = "ep-plugin"
        mock_ep.load.return_value = spec

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("exo_cli.plugins.importlib.metadata.entry_points", return_value=mock_eps):
            mgr = PluginManager()
            count = mgr.load_entrypoints()

        assert count == 1
        assert mgr.get("ep-plugin") is spec

    def test_load_callable_entrypoint(self) -> None:
        spec = PluginSpec(name="factory-plugin")

        def factory() -> PluginSpec:
            return spec

        mock_ep = MagicMock()
        mock_ep.name = "factory-plugin"
        mock_ep.load.return_value = factory

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("exo_cli.plugins.importlib.metadata.entry_points", return_value=mock_eps):
            mgr = PluginManager()
            count = mgr.load_entrypoints()

        assert count == 1
        assert mgr.get("factory-plugin") is spec

    def test_skip_non_pluginspec(self) -> None:
        mock_ep = MagicMock()
        mock_ep.name = "bad"
        mock_ep.load.return_value = "not a spec"

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("exo_cli.plugins.importlib.metadata.entry_points", return_value=mock_eps):
            mgr = PluginManager()
            count = mgr.load_entrypoints()

        assert count == 0

    def test_broken_entrypoint_skipped(self) -> None:
        mock_ep = MagicMock()
        mock_ep.name = "broken"
        mock_ep.load.side_effect = ImportError("missing dep")

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("exo_cli.plugins.importlib.metadata.entry_points", return_value=mock_eps):
            mgr = PluginManager()
            count = mgr.load_entrypoints()

        assert count == 0

    def test_no_entrypoints(self) -> None:
        mock_eps = MagicMock()
        mock_eps.select.return_value = []

        with patch("exo_cli.plugins.importlib.metadata.entry_points", return_value=mock_eps):
            mgr = PluginManager()
            count = mgr.load_entrypoints()

        assert count == 0


# ---------------------------------------------------------------------------
# PluginManager — lifecycle hooks
# ---------------------------------------------------------------------------


class TestPluginManagerHooks:
    async def test_startup_calls_hooks(self) -> None:
        hook_fn = AsyncMock()
        spec = PluginSpec(name="p1", hooks={PluginHook.STARTUP: hook_fn})
        mgr = PluginManager()
        mgr.register(spec)

        await mgr.startup(config={"key": "val"})
        hook_fn.assert_awaited_once_with(config={"key": "val"})

    async def test_shutdown_calls_hooks(self) -> None:
        hook_fn = AsyncMock()
        spec = PluginSpec(name="p1", hooks={PluginHook.SHUTDOWN: hook_fn})
        mgr = PluginManager()
        mgr.register(spec)

        await mgr.shutdown()
        hook_fn.assert_awaited_once_with()

    async def test_run_hook_pre_run(self) -> None:
        hook_fn = AsyncMock()
        spec = PluginSpec(name="p1", hooks={PluginHook.PRE_RUN: hook_fn})
        mgr = PluginManager()
        mgr.register(spec)

        await mgr.run_hook(PluginHook.PRE_RUN, input="hello")
        hook_fn.assert_awaited_once_with(input="hello")

    async def test_run_hook_post_run(self) -> None:
        hook_fn = AsyncMock()
        spec = PluginSpec(name="p1", hooks={PluginHook.POST_RUN: hook_fn})
        mgr = PluginManager()
        mgr.register(spec)

        await mgr.run_hook(PluginHook.POST_RUN, result="done")
        hook_fn.assert_awaited_once_with(result="done")

    async def test_multiple_plugins_hook_order(self) -> None:
        order: list[str] = []

        async def hook_a(**kwargs: Any) -> None:
            order.append("a")

        async def hook_b(**kwargs: Any) -> None:
            order.append("b")

        mgr = PluginManager()
        mgr.register(PluginSpec(name="a", hooks={PluginHook.STARTUP: hook_a}))
        mgr.register(PluginSpec(name="b", hooks={PluginHook.STARTUP: hook_b}))

        await mgr.startup()
        assert order == ["a", "b"]

    async def test_hook_error_propagates(self) -> None:
        async def bad_hook(**kwargs: Any) -> None:
            raise RuntimeError("hook failure")

        spec = PluginSpec(name="bad", hooks={PluginHook.STARTUP: bad_hook})
        mgr = PluginManager()
        mgr.register(spec)

        with pytest.raises(RuntimeError, match="hook failure"):
            await mgr.startup()

    async def test_no_hooks_is_noop(self) -> None:
        spec = PluginSpec(name="empty")
        mgr = PluginManager()
        mgr.register(spec)

        # Should not raise
        await mgr.startup()
        await mgr.shutdown()
        await mgr.run_hook(PluginHook.PRE_RUN)


# ---------------------------------------------------------------------------
# _load_plugin_file helper
# ---------------------------------------------------------------------------


class TestLoadPluginFile:
    def test_valid_file(self, tmp_path: Path) -> None:
        f = tmp_path / "good.py"
        f.write_text(
            textwrap.dedent("""\
                from exo_cli.plugins import PluginSpec
                plugin = PluginSpec(name="good")
            """)
        )
        spec = _load_plugin_file(f)
        assert spec is not None
        assert spec.name == "good"

    def test_no_plugin_attr(self, tmp_path: Path) -> None:
        f = tmp_path / "nope.py"
        f.write_text("x = 1\n")
        spec = _load_plugin_file(f)
        assert spec is None

    def test_wrong_type(self, tmp_path: Path) -> None:
        f = tmp_path / "wrong.py"
        f.write_text("plugin = 42\n")
        spec = _load_plugin_file(f)
        assert spec is None

    def test_execution_error(self, tmp_path: Path) -> None:
        f = tmp_path / "crash.py"
        f.write_text("raise ValueError('boom')\n")
        with pytest.raises(ValueError, match="boom"):
            _load_plugin_file(f)
