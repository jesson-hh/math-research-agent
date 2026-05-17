"""Shared pytest fixtures."""
import pytest
from pathlib import Path


@pytest.fixture
def tmp_vault(tmp_path: Path) -> Path:
    """A clean temporary vault directory for tests that need one."""
    vault = tmp_path / "vault"
    vault.mkdir()
    return vault
