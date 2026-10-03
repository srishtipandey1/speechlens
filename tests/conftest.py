"""Shared optional local-data fixtures for integration tests."""

from pathlib import Path

import pytest

from speechlens.alignment.align import has_cached_mms_fa_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_CLIP_PATH = PROJECT_ROOT / "data" / "raw" / "test_clip.wav"
REAL_TRANSCRIPT_PATH = PROJECT_ROOT / "data" / "raw" / "test_clip.txt"


@pytest.fixture
def real_alignment_fixture() -> tuple[Path, Path]:
    """Require local speech, transcript, and cached weights without downloading."""
    missing = []
    if not REAL_CLIP_PATH.is_file():
        missing.append("data/raw/test_clip.wav")
    if not REAL_TRANSCRIPT_PATH.is_file():
        missing.append("data/raw/test_clip.txt")
    if not has_cached_mms_fa_model():
        missing.append("locally cached MMS_FA model")
    if missing:
        pytest.skip(f"optional real-audio test prerequisites missing: {', '.join(missing)}")
    return REAL_CLIP_PATH, REAL_TRANSCRIPT_PATH
