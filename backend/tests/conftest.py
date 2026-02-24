"""Shared test fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def sample_csv_path() -> Path:
    """Path to the sample session CSV."""
    return Path(__file__).parent.parent.parent / "Saturday, February 21, 2026.csv"


@pytest.fixture
def sample_csv_content(sample_csv_path: Path) -> str:
    """Raw content of the sample session CSV."""
    return sample_csv_path.read_text(encoding="utf-8-sig")
