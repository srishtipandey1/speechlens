"""Real-clip injection properties; no data or model downloads are attempted."""

from pathlib import Path

import numpy as np
import pytest

from speechlens.alignment.align import SAMPLE_RATE_HZ, align, has_cached_mms_fa_model, load_audio
from speechlens.injection import (
    filler,
    long_pause,
    monotone,
    pace_fast,
    pace_slow,
    stumble_repeat,
    volume_dropoff,
)
from speechlens.schema import WordTiming


CLIP_PATH = Path("data/raw/test_clip.wav")
TRANSCRIPT_PATH = Path("data/raw/test_clip.txt")
INJECTORS = (
    pace_fast,
    pace_slow,
    long_pause,
    monotone,
    volume_dropoff,
    filler,
    stumble_repeat,
)


def _short_gap_region(timings: list[WordTiming]) -> tuple[int, int]:
    """Find a two-word region containing a pause-eligible word boundary."""
    for previous_index in range(1, len(timings) - 1):
        gap_s = timings[previous_index + 1].start_s - timings[previous_index].end_s
        if 0.0 <= gap_s < 0.6:
            return previous_index, previous_index + 2
    raise AssertionError("real test clip has no short natural word boundary")


@pytest.mark.skipif(
    not (CLIP_PATH.is_file() and has_cached_mms_fa_model()),
    reason="test clip or locally cached MMS_FA model is missing",
)
def test_all_injectors_preserve_real_clip_properties() -> None:
    """Apply each effect to real speech and check labels, timings, and audio."""
    if not TRANSCRIPT_PATH.is_file():
        pytest.fail("test_clip.txt is required when the real audio fixture is present")

    audio = load_audio(CLIP_PATH)
    transcript = TRANSCRIPT_PATH.read_text(encoding="utf-8")
    timings = align(audio, transcript)
    region = _short_gap_region(timings)

    for injector in INJECTORS:
        changed, labels, updated = injector(
            audio,
            timings,
            region,
            0.6,
            np.random.default_rng(2026),
        )
        assert labels
        assert np.isfinite(changed).all()
        assert float(np.max(np.abs(changed))) < 1.0
        assert all(
            previous.end_s <= current.start_s
            for previous, current in zip(updated, updated[1:])
        )
        assert all(item.end_s <= changed.size / SAMPLE_RATE_HZ for item in updated)
        assert all(label.rendered_end_s <= changed.size / SAMPLE_RATE_HZ for label in labels)