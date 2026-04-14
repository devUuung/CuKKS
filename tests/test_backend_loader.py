from __future__ import annotations

from unittest.mock import patch

import pytest

import cukks.backend_loader as backend_loader


def test_load_backend_does_not_attempt_auto_install_when_backend_missing() -> None:
    with patch.object(backend_loader, "_cache", None), patch.object(
        backend_loader, "_try_import", return_value=(None, None)
    ), patch.object(
        backend_loader, "_build_error_message", return_value="backend missing"
    ):
        with pytest.raises(RuntimeError, match="backend missing"):
            backend_loader.load_backend()


def test_load_backend_returns_imported_classes_without_side_effects() -> None:
    sentinel_config = object()
    sentinel_context = object()

    with patch.object(backend_loader, "_cache", None), patch.object(
        backend_loader, "_try_import", return_value=(sentinel_config, sentinel_context)
    ):
        cfg, ctx = backend_loader.load_backend()

    assert cfg is sentinel_config
    assert ctx is sentinel_context
