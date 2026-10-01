"""Real-clip alignment sanity checks; never download data or model weights."""

from pathlib import Path

import numpy as np
import pytest

from speechlens.alignment.align import (
    SAMPLE_RATE_HZ,
    align,
    has_cached_mms_fa_model,
    load_audio,
)


CLIP_PATH = Path("data/raw/test_clip.wav")
TRANSCRIPT_PATH = Path("data/raw/test_clip.txt")


@pytest.mark.skipif(
    not (CLIP_PATH.is_file() and has_cached_mms_fa_model()),
    reason="test clip or locally cached MMS_FA model is missing",
)
def test_alignment_is_monotonic_and_invariant_to_two_second_silence_shift() -> None:
    """Prepending exact silence shifts every timing without changing word order."""
    if not TRANSCRIPT_PATH.is_file():
        pytest.fail("test_clip.txt is required when the real audio fixture is present")

    waveform = load_audio(CLIP_PATH)
    transcript = TRANSCRIPT_PATH.read_text(encoding="utf-8")
    original_timings = align(waveform, transcript)
    shift_samples = 2 * SAMPLE_RATE_HZ
    shifted_waveform = np.concatenate(
        (np.zeros(shift_samples, dtype=np.float32), waveform)
    )
    shifted_timings = align(shifted_waveform, transcript)
    tolerance_s = 0.03

    assert original_timings
    assert len(original_timings) == len(shifted_timings)
    for original, shifted in zip(original_timings, shifted_timings):
        assert original.word == shifted.word
        assert abs((shifted.start_s - original.start_s) - 2.0) <= tolerance_s
        assert abs((shifted.end_s - original.end_s) - 2.0) <= tolerance_s

    audio_duration_s = shifted_waveform.size / SAMPLE_RATE_HZ
    for timings in (original_timings, shifted_timings):
        assert all(
            earlier.start_s <= later.start_s and earlier.end_s <= later.start_s
            for earlier, later in zip(timings, timings[1:])
        )
        assert timings[-1].end_s <= audio_duration_s