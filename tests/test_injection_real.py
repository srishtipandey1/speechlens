"""Real-clip injection properties; no data or model downloads are attempted."""

from pathlib import Path
import tempfile

import numpy as np
import parselmouth
import pytest
import soundfile as sf
import yaml

from speechlens.alignment.align import SAMPLE_RATE_HZ, align, load_audio
from speechlens.injection import (
    filler,
    long_pause,
    monotone,
    pace_fast,
    pace_slow,
    stumble_repeat,
    volume_dropoff,
)
from speechlens.injection.engine import _quietest_room_tone
from speechlens.schema import WordTiming
from scripts.make_dataset import _write_flac
from scripts.demo_injection import _choose_region


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
    for previous_index in range(2, len(timings) - 1):
        gap_s = timings[previous_index + 1].start_s - timings[previous_index].end_s
        if 0.0 <= gap_s < 0.6:
            return previous_index, previous_index + 2
    raise AssertionError("real test clip has no short natural word boundary")


@pytest.mark.slow
def test_all_injectors_preserve_real_clip_properties(
    real_alignment_fixture: tuple[Path, Path],
) -> None:
    """Apply each effect to real speech and check labels, timings, and audio."""
    clip_path, transcript_path = real_alignment_fixture
    audio = load_audio(clip_path)
    transcript = transcript_path.read_text(encoding="utf-8")
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


def _f0_std_in_semitones(waveform: np.ndarray, start_s: float, end_s: float) -> float:
    """Measure voiced-only F0 standard deviation relative to local median."""
    with Path("config/injection.yaml").open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)["praat"]
    sound = parselmouth.Sound(waveform, sampling_frequency=SAMPLE_RATE_HZ)
    pitch = sound.to_pitch_ac(
        time_step=float(config["time_step_s"]),
        pitch_floor=float(config["pitch_floor_hz"]),
        pitch_ceiling=float(config["pitch_ceiling_hz"]),
    )
    frequencies = pitch.selected_array["frequency"]
    times = pitch.x1 + np.arange(frequencies.size) * pitch.dx
    selected = frequencies[(frequencies > 0.0) & (times >= start_s) & (times <= end_s)]
    assert selected.size > 0
    semitones = 12.0 * np.log2(selected / np.median(selected))
    return float(np.std(semitones))


@pytest.mark.slow
@pytest.mark.parametrize(
    ("severity", "minimum_ratio", "maximum_ratio"),
    [(0.3, 0.60, 0.95), (0.9, 0.0, 0.30)],
)
def test_monotone_reduces_real_clip_voiced_f0_variation(
    severity: float,
    minimum_ratio: float,
    maximum_ratio: float,
    real_alignment_fixture: tuple[Path, Path],
) -> None:
    """Real-speech monotone F0 variation meets its severity-specific range."""
    clip_path, transcript_path = real_alignment_fixture
    audio = load_audio(clip_path)
    timings = align(audio, transcript_path.read_text(encoding="utf-8"))
    region = _choose_region(len(timings))
    changed, labels, _ = monotone(
        audio,
        timings,
        region,
        severity,
        np.random.default_rng(2026),
    )
    label = labels[0]
    original_std = _f0_std_in_semitones(
        audio, label.original_start_s, label.original_end_s
    )
    flawed_std = _f0_std_in_semitones(
        changed, label.rendered_start_s, label.rendered_end_s
    )
    ratio = flawed_std / original_std
    print(f"severity={severity:.1f} F0 std ratio={ratio:.4f}")

    assert minimum_ratio <= ratio <= maximum_ratio


@pytest.mark.slow
def test_real_pause_room_tone_survives_flac_and_matches_source_level(
    real_alignment_fixture: tuple[Path, Path],
) -> None:
    """Real pause tone remains nonzero after FLAC encoding and RMS-matches source."""
    clip_path, transcript_path = real_alignment_fixture
    audio = load_audio(clip_path)
    timings = align(audio, transcript_path.read_text(encoding="utf-8"))
    region = _short_gap_region(timings)
    changed, labels, _ = long_pause(
        audio, timings, region, 0.9, np.random.default_rng(742)
    )
    label = labels[0]
    start = round(label.rendered_start_s * SAMPLE_RATE_HZ)
    end = round(label.rendered_end_s * SAMPLE_RATE_HZ)
    with tempfile.TemporaryDirectory(prefix="speechlens-real-pause-", dir=Path.cwd()) as directory:
        path = Path(directory) / "pause.flac"
        _write_flac(path, changed)
        encoded, _ = sf.read(path, dtype="float32")
    pause = encoded[start:end].astype(np.float64)
    reference = _quietest_room_tone(audio, round(0.1 * SAMPLE_RATE_HZ)).astype(np.float64)
    pause_rms = float(np.sqrt(np.mean(pause**2)))
    reference_rms = float(np.sqrt(np.mean(reference**2)))
    difference_db = abs(20.0 * np.log10(pause_rms / reference_rms))

    assert np.any(pause != 0.0)
    assert difference_db <= 3.0