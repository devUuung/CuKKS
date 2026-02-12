"""
Tests for cukks._cuda_compat — CUDA version detection, package resolution,
and runtime validation.

All tests use mocks — no actual CUDA or GPU required.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cukks._cuda_compat import (
    CUDA_TO_PACKAGE,
    PACKAGE_TO_CUDA,
    SUPPORTED_CUDA_VERSIONS,
    _parse_version,
    detect_cuda_version,
    get_installed_backend,
    get_installed_backend_cuda_version,
    resolve_backend_package,
    validate_backend_cuda,
)


# ═══════════════════════════════════════════════════════════════════════════════
# _parse_version
# ═══════════════════════════════════════════════════════════════════════════════


class TestParseVersion:
    """Tests for _parse_version()."""

    def test_basic(self) -> None:
        assert _parse_version("12.1") == (12, 1)

    def test_three_part(self) -> None:
        """Only major.minor should be extracted."""
        assert _parse_version("12.1.1") == (12, 1)

    def test_leading_trailing_whitespace(self) -> None:
        assert _parse_version("  11.8 ") == (11, 8)

    def test_single_number_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_version("12")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_version("")

    def test_garbage_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_version("abc.def")


# ═══════════════════════════════════════════════════════════════════════════════
# detect_cuda_version
# ═══════════════════════════════════════════════════════════════════════════════


class TestDetectCudaVersion:
    """Tests for detect_cuda_version()."""

    def test_torch_not_installed(self) -> None:
        """When torch cannot be imported, return None."""
        with patch.dict(sys.modules, {"torch": None}):
            assert detect_cuda_version() is None

    def test_torch_cpu_only(self) -> None:
        """When torch.version.cuda is None (CPU build), return None."""
        mock_torch = MagicMock()
        mock_torch.version.cuda = None
        with patch.dict(sys.modules, {"torch": mock_torch}):
            assert detect_cuda_version() is None

    def test_torch_cuda_128(self) -> None:
        mock_torch = MagicMock()
        mock_torch.version.cuda = "12.8"
        with patch.dict(sys.modules, {"torch": mock_torch}):
            assert detect_cuda_version() == "12.8"

    def test_torch_cuda_three_part(self) -> None:
        """torch.version.cuda sometimes returns "12.1.1" — normalise to "12.1"."""
        mock_torch = MagicMock()
        mock_torch.version.cuda = "12.1.1"
        with patch.dict(sys.modules, {"torch": mock_torch}):
            assert detect_cuda_version() == "12.1"

    def test_torch_cuda_malformed(self) -> None:
        mock_torch = MagicMock()
        mock_torch.version.cuda = "not_a_version"
        with patch.dict(sys.modules, {"torch": mock_torch}):
            assert detect_cuda_version() is None


# ═══════════════════════════════════════════════════════════════════════════════
# resolve_backend_package
# ═══════════════════════════════════════════════════════════════════════════════


class TestResolveBackendPackage:
    """Tests for resolve_backend_package() — round-down only."""

    def test_none_input(self) -> None:
        assert resolve_backend_package(None) is None

    # ── Exact matches ────────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "cuda_ver,expected",
        [
            ("11.8", "cukks-cu118"),
            ("12.1", "cukks-cu121"),
            ("12.4", "cukks-cu124"),
            ("12.8", "cukks-cu128"),
        ],
    )
    def test_exact_matches(self, cuda_ver: str, expected: str) -> None:
        assert resolve_backend_package(cuda_ver) == expected

    # ── Round-down within same major ─────────────────────────────────────

    def test_round_down_12_3(self) -> None:
        """12.3 → 12.1 (highest ≤ 12.3)."""
        assert resolve_backend_package("12.3") == "cukks-cu121"

    def test_round_down_12_5(self) -> None:
        """12.5 → 12.4 (highest ≤ 12.5)."""
        assert resolve_backend_package("12.5") == "cukks-cu124"

    def test_round_down_12_7(self) -> None:
        """12.7 → 12.4 (highest ≤ 12.7)."""
        assert resolve_backend_package("12.7") == "cukks-cu124"

    def test_round_down_12_9(self) -> None:
        """12.9 → 12.8 (highest ≤ 12.9)."""
        assert resolve_backend_package("12.9") == "cukks-cu128"

    def test_round_down_11_9(self) -> None:
        """11.9 → 11.8 (highest ≤ 11.9)."""
        assert resolve_backend_package("11.9") == "cukks-cu118"

    # ── No match (too low or wrong major) ────────────────────────────────

    def test_no_match_12_0(self) -> None:
        """12.0 → None (no 12.x with minor ≤ 0)."""
        assert resolve_backend_package("12.0") is None

    def test_no_match_11_7(self) -> None:
        """11.7 → None (no 11.x with minor ≤ 7)."""
        assert resolve_backend_package("11.7") is None

    def test_no_match_13_0(self) -> None:
        """13.0 → None (no major 13 in registry)."""
        assert resolve_backend_package("13.0") is None

    def test_no_match_10_2(self) -> None:
        """10.2 → None (no major 10 in registry)."""
        assert resolve_backend_package("10.2") is None

    # ── Never rounds up ──────────────────────────────────────────────────

    def test_never_rounds_up_11_0(self) -> None:
        """11.0 must NOT resolve to cukks-cu118."""
        assert resolve_backend_package("11.0") is None

    # ── Malformed input ──────────────────────────────────────────────────

    def test_malformed_input(self) -> None:
        assert resolve_backend_package("abc") is None

    def test_single_number(self) -> None:
        assert resolve_backend_package("12") is None


# ═══════════════════════════════════════════════════════════════════════════════
# get_installed_backend
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetInstalledBackend:
    """Tests for get_installed_backend()."""

    def _make_dist(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(metadata={"Name": name})

    def test_backend_found(self) -> None:
        dists = [self._make_dist("numpy"), self._make_dist("cukks-cu121")]
        with patch("cukks._cuda_compat.importlib.metadata.distributions", return_value=iter(dists)):
            assert get_installed_backend() == "cukks-cu121"

    def test_no_backend(self) -> None:
        dists = [self._make_dist("numpy"), self._make_dist("cukks")]
        with patch("cukks._cuda_compat.importlib.metadata.distributions", return_value=iter(dists)):
            assert get_installed_backend() is None

    def test_empty_distributions(self) -> None:
        with patch("cukks._cuda_compat.importlib.metadata.distributions", return_value=iter([])):
            assert get_installed_backend() is None

    def test_exception_returns_none(self) -> None:
        with patch(
            "cukks._cuda_compat.importlib.metadata.distributions",
            side_effect=RuntimeError("fail"),
        ):
            assert get_installed_backend() is None


# ═══════════════════════════════════════════════════════════════════════════════
# get_installed_backend_cuda_version
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetInstalledBackendCudaVersion:
    """Tests for get_installed_backend_cuda_version()."""

    def test_known_backend(self) -> None:
        with patch("cukks._cuda_compat.get_installed_backend", return_value="cukks-cu128"):
            assert get_installed_backend_cuda_version() == "12.8"

    def test_no_backend(self) -> None:
        with patch("cukks._cuda_compat.get_installed_backend", return_value=None):
            assert get_installed_backend_cuda_version() is None

    def test_unknown_backend(self) -> None:
        with patch("cukks._cuda_compat.get_installed_backend", return_value="cukks-cu999"):
            assert get_installed_backend_cuda_version() is None


# ═══════════════════════════════════════════════════════════════════════════════
# validate_backend_cuda
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateBackendCuda:
    """Tests for validate_backend_cuda()."""

    def test_ok_exact_match(self) -> None:
        with (
            patch("cukks._cuda_compat.detect_cuda_version", return_value="12.8"),
            patch("cukks._cuda_compat.get_installed_backend_cuda_version", return_value="12.8"),
            patch("cukks._cuda_compat.get_installed_backend", return_value="cukks-cu128"),
        ):
            status, _ = validate_backend_cuda()
            assert status == "ok"

    def test_approximate_lower_minor(self) -> None:
        """Backend cu121 with torch CUDA 12.4 → approximate (safe)."""
        with (
            patch("cukks._cuda_compat.detect_cuda_version", return_value="12.4"),
            patch("cukks._cuda_compat.get_installed_backend_cuda_version", return_value="12.1"),
            patch("cukks._cuda_compat.get_installed_backend", return_value="cukks-cu121"),
        ):
            status, _ = validate_backend_cuda()
            assert status == "approximate"

    def test_mismatch_higher_minor(self) -> None:
        """Backend cu128 with torch CUDA 12.4 → mismatch (dangerous)."""
        with (
            patch("cukks._cuda_compat.detect_cuda_version", return_value="12.4"),
            patch("cukks._cuda_compat.get_installed_backend_cuda_version", return_value="12.8"),
            patch("cukks._cuda_compat.get_installed_backend", return_value="cukks-cu128"),
        ):
            status, msg = validate_backend_cuda()
            assert status == "mismatch"
            assert "segfault" in msg.lower()

    def test_mismatch_different_major(self) -> None:
        """Backend cu118 with torch CUDA 12.1 → mismatch."""
        with (
            patch("cukks._cuda_compat.detect_cuda_version", return_value="12.1"),
            patch("cukks._cuda_compat.get_installed_backend_cuda_version", return_value="11.8"),
            patch("cukks._cuda_compat.get_installed_backend", return_value="cukks-cu118"),
        ):
            status, _ = validate_backend_cuda()
            assert status == "mismatch"

    def test_no_cuda(self) -> None:
        with patch("cukks._cuda_compat.detect_cuda_version", return_value=None):
            status, _ = validate_backend_cuda()
            assert status == "no_cuda"

    def test_no_backend(self) -> None:
        with (
            patch("cukks._cuda_compat.detect_cuda_version", return_value="12.8"),
            patch("cukks._cuda_compat.get_installed_backend_cuda_version", return_value=None),
            patch("cukks._cuda_compat.get_installed_backend", return_value=None),
        ):
            status, _ = validate_backend_cuda()
            assert status == "no_backend"


# ═══════════════════════════════════════════════════════════════════════════════
# Registry consistency
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegistryConsistency:
    """Sanity checks on the package registry."""

    def test_cuda_to_package_and_back(self) -> None:
        for cuda_ver, pkg_name in CUDA_TO_PACKAGE.items():
            assert PACKAGE_TO_CUDA[pkg_name] == cuda_ver

    def test_supported_versions_sorted(self) -> None:
        parsed = [tuple(int(x) for x in v.split(".")) for v in SUPPORTED_CUDA_VERSIONS]
        assert parsed == sorted(parsed)

    def test_all_packages_follow_naming(self) -> None:
        import re

        for pkg in CUDA_TO_PACKAGE.values():
            assert re.match(r"^cukks-cu\d+$", pkg), f"Bad package name: {pkg}"
