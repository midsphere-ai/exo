"""Tests for startup validation logic in app.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from exo_web.app import _validate_startup


class TestStartupRejectsDefaultKey:
    """Test that _validate_startup() enforces secret key policy."""

    def test_startup_rejects_default_key_in_production(self) -> None:
        """RuntimeError raised when debug=False and secret key is default."""
        with (
            patch("exo_web.app.settings") as mock_settings,
        ):
            mock_settings.debug = False
            mock_settings.secret_key = "change-me-in-production"

            with pytest.raises(RuntimeError, match="EXO_SECRET_KEY must be changed in production"):
                _validate_startup()

    def test_startup_allows_default_key_in_debug_mode(self) -> None:
        """No RuntimeError raised when debug=True even with default secret key."""
        with (
            patch("exo_web.app.settings") as mock_settings,
            patch("exo_web.app._DB_PATH", "/tmp/exo-test.db"),
            patch("exo_web.app.Path") as mock_path_cls,
        ):
            mock_settings.debug = True
            mock_settings.secret_key = "change-me-in-production"
            mock_settings.session_expiry_hours = 72
            mock_settings.cors_origins = []

            # Make path checks pass
            mock_path = mock_path_cls.return_value
            mock_path.parent.exists.return_value = True

            with patch("exo_web.app.os.access", return_value=True):
                # Should not raise
                _validate_startup()

    def test_startup_allows_custom_secret_key_in_production(self) -> None:
        """No RuntimeError raised when a custom secret key is set, even with debug=False."""
        with (
            patch("exo_web.app.settings") as mock_settings,
            patch("exo_web.app._DB_PATH", "/tmp/exo-test.db"),
            patch("exo_web.app.Path") as mock_path_cls,
        ):
            mock_settings.debug = False
            mock_settings.secret_key = "super-secret-production-key-123"
            mock_settings.session_expiry_hours = 72
            mock_settings.cors_origins = []

            mock_path = mock_path_cls.return_value
            mock_path.parent.exists.return_value = True

            with patch("exo_web.app.os.access", return_value=True):
                # Should not raise
                _validate_startup()
