"""Property-style tests for deterministic audio flaw injection."""

import numpy as np
import pytest
import soundfile as sf
import tempfile
from pathlib import Path

from speechlens.injection import (
    FlawSpec,
    apply_flaws,
    filler,
    long_pause,
    monotone,
    pace_fast,
    pace_slow,
    stumble_repeat,
    volume_dropoff,
)
from speechlens.injection.engine import _psola_duration
from scripts.make_dataset import _write_flac
from speechlens.schema import WordTiming


SAMPLE_RATE_HZ = 16000
REGION = (3, 7)
INJECTORS = {
    "pace_fast": pace_fast,
    "pace_slow": pace_slow,
    "long_pause": long_pause,
    "monotone": monotone,
    "volume_dropoff": volume_dropoff,
    "filler": filler,
    "stumble_repeat": stumble_repeat,
}


def make_signal(duration_s: float = 7.0) -> np.ndarray:
    """Create a deterministic voiced chirp with headroom for transformations."""
    times = np.arange(round(duration_s * SAMPLE_RATE_HZ), dtype=np.float64) / SAMPLE_RATE_HZ
    phase = 2.0 * np.pi * (150.0 * times + 3.0 * times**2)
    return (0.28 * np.sin(phase)).astype(np.float32)


def make_timings(count: int = 10) -> list[WordTiming]:
    """Create ordered synthetic word intervals with short natural gaps."""
    return [
        WordTiming(
            word=f"word{index}",
            start_s=0.15 + index * 0.55,
            end_s=0.48 + index * 0.55,
            confidence=0.9,
        )
        for index in range(count)
    ]


def call_injector(flaw_type: str, severity: float, seed: int = 27):
    """Call one registered effect with fresh reproducible synthetic inputs."""
    return INJECTORS[flaw_type](
        make_signal(),
        make_timings(),
        REGION,
        severity,
        np.random.default_rng(seed),
    )


def assert_timings_valid(timings: list[WordTiming], audio_samples: int) -> None:
    """Assert word timings remain monotonic, non-overlapping, and in-bounds."""
    assert all(
        previous.start_s <= current.start_s
        and previous.end_s <= current.start_s
        for previous, current in zip(timings, timings[1:])
    )
    assert all(timing.end_s <= audio_samples / SAMPLE_RATE_HZ for timing in timings)


@pytest.mark.parametrize("flaw_type", list(INJECTORS))
def test_zero_severity_is_byte_identical_and_unlabeled(flaw_type: str) -> None:
    """Every effect is an exact no-op at zero severity."""
    audio = make_signal()
    timings = make_timings()
    output, labels, updated = INJECTORS[flaw_type](
        audio, timings, REGION, 0.0, np.random.default_rng(13)
    )

    assert output.tobytes() == audio.tobytes()
    assert labels == []
    assert updated == timings


@pytest.mark.parametrize("flaw_type", list(INJECTORS))
def test_effect_outputs_have_valid_labels_timings_and_audio(flaw_type: str) -> None:
    """Each transformation returns finite unclipped audio and a contained label."""
    audio = make_signal()
    timings = make_timings()
    output, labels, updated = INJECTORS[flaw_type](
        audio, timings, REGION, 0.7, np.random.default_rng(19)
    )

    assert len(labels) == 1
    label = labels[0]
    assert label.flaw_type == flaw_type
    assert label.rendered_start_s < label.rendered_end_s
    assert label.rendered_end_s <= output.size / SAMPLE_RATE_HZ
    assert np.isfinite(output).all()
    assert float(np.max(np.abs(output))) < 1.0
    assert_timings_valid(updated, output.size)


@pytest.mark.parametrize(
    ("flaw_type", "rate_factor"),
    [("pace_fast", 1.56), ("pace_slow", 0.685)],
)
def test_pace_duration_matches_configured_rate(
    flaw_type: str,
    rate_factor: float,
) -> None:
    """PSOLA region length equals source length divided by its rate factor."""
    audio = make_signal()
    timings = make_timings()
    output, labels, _ = INJECTORS[flaw_type](
        audio, timings, REGION, 0.7, np.random.default_rng(23)
    )
    start_sample = round(timings[REGION[0]].start_s * SAMPLE_RATE_HZ)
    end_sample = round(timings[REGION[1] - 1].end_s * SAMPLE_RATE_HZ)
    expected_region_samples = round((end_sample - start_sample) / rate_factor)
    actual_region_samples = round(
        (labels[0].rendered_end_s - labels[0].rendered_start_s) * SAMPLE_RATE_HZ
    )

    assert abs(actual_region_samples - expected_region_samples) <= 1
    assert output.size - audio.size == expected_region_samples - (end_sample - start_sample)


@pytest.mark.parametrize("flaw_type", ["long_pause", "filler", "stumble_repeat"])
def test_insertions_change_duration_by_labeled_rendered_audio(flaw_type: str) -> None:
    """Inserted sample count agrees with the labeled rendered interval."""
    audio = make_signal()
    output, labels, _ = INJECTORS[flaw_type](
        audio, make_timings(), REGION, 0.7, np.random.default_rng(29)
    )
    inserted_samples = round(
        (labels[0].rendered_end_s - labels[0].rendered_start_s) * SAMPLE_RATE_HZ
    )

    assert output.size - audio.size == inserted_samples


def test_long_pause_uses_nonzero_room_tone_at_source_noise_floor() -> None:
    """Inserted room tone is nonzero and RMS-matched to the clip's quiet segment."""
    times = np.arange(5 * SAMPLE_RATE_HZ, dtype=np.float32) / SAMPLE_RATE_HZ
    audio = np.zeros(times.size, dtype=np.float32)
    active_voice = (times >= 0.2) & (times <= 1.0)
    quiet_room = (times >= 3.0) & (times <= 4.0)
    audio[active_voice] = 0.2 * np.sin(2 * np.pi * 180 * times[active_voice])
    audio[quiet_room] = 0.002 * np.sin(2 * np.pi * 180 * times[quiet_room])
    timings = [
        WordTiming(word="one", start_s=0.25, end_s=0.55),
        WordTiming(word="two", start_s=0.7, end_s=0.95),
    ]

    changed, labels, _ = long_pause(
        audio, timings, (0, 1), 0.5, np.random.default_rng(5)
    )
    start = round(labels[0].rendered_start_s * SAMPLE_RATE_HZ)
    end = round(labels[0].rendered_end_s * SAMPLE_RATE_HZ)
    pause = changed[start:end]
    reference = audio[round(3.2 * SAMPLE_RATE_HZ) : round(3.3 * SAMPLE_RATE_HZ)]
    pause_rms = float(np.sqrt(np.mean(np.square(pause, dtype=np.float64))))
    reference_rms = float(np.sqrt(np.mean(np.square(reference, dtype=np.float64))))
    difference_db = abs(20.0 * np.log10(pause_rms / reference_rms))

    assert np.any(pause != 0.0)
    assert pause[0] == 0.0
    assert pause[-1] == 0.0
    assert difference_db <= 3.0


@pytest.mark.parametrize(
    ("severity", "expected_duration_s"),
    [(0.2, 0.6), (0.5, 1.3), (1.0, 3.0)],
)
def test_long_pause_duration_uses_configured_severity_knots(
    severity: float,
    expected_duration_s: float,
) -> None:
    """Configured pause severity knots map to the requested duration values."""
    output, labels, _ = long_pause(
        make_signal(),
        make_timings(),
        REGION,
        severity,
        np.random.default_rng(4),
    )

    actual_duration_s = labels[0].rendered_end_s - labels[0].rendered_start_s
    assert output.size > 0
    assert abs(actual_duration_s - expected_duration_s) <= 1 / SAMPLE_RATE_HZ


def test_dataset_flac_preserves_sub_pcm16_room_tone() -> None:
    """The generated 24-bit FLAC retains quiet samples below the PCM16 LSB."""
    import tempfile

    quiet = np.full(SAMPLE_RATE_HZ, 1.2e-5, dtype=np.float32)
    with tempfile.TemporaryDirectory(prefix="speechlens-pcm24-", dir=Path.cwd()) as directory:
        path = Path(directory) / "quiet.flac"
        _write_flac(path, quiet)
        decoded, sample_rate = sf.read(path, dtype="float32")

    assert sample_rate == SAMPLE_RATE_HZ
    assert np.any(decoded != 0.0)
    assert float(np.sqrt(np.mean(np.square(decoded, dtype=np.float64)))) > 0.0


def test_stumble_repeat_label_duration_grows_with_severity() -> None:
    """Stumble bands repeat progressively more source material and pauses."""
    durations = []
    for severity in (0.2, 0.5, 0.9):
        _, labels, _ = stumble_repeat(
            make_signal(),
            make_timings(),
            REGION,
            severity,
            np.random.default_rng(1),
        )
        durations.append(labels[0].rendered_end_s - labels[0].rendered_start_s)

    assert durations[0] < durations[1] < durations[2]


@pytest.mark.parametrize(
    ("severity", "expected_added_words"),
    [(0.399, 1), (0.4, 2), (0.7, 2), (0.701, 4)],
)
def test_stumble_repeat_uses_configured_severity_bands(
    severity: float,
    expected_added_words: int,
) -> None:
    """Stumble word count changes at the configured severity boundaries."""
    _, _, updated = stumble_repeat(
        make_signal(),
        make_timings(),
        REGION,
        severity,
        np.random.default_rng(1),
    )

    assert len(updated) == len(make_timings()) + expected_added_words


@pytest.mark.parametrize("flaw_type", ["filler", "stumble_repeat"])
def test_randomized_effects_are_seed_deterministic(flaw_type: str) -> None:
    """Identical seeds produce byte-identical waveforms, labels, and timings."""
    first = call_injector(flaw_type, 0.8, seed=33)
    second = call_injector(flaw_type, 0.8, seed=33)

    assert first[0].tobytes() == second[0].tobytes()
    assert first[1:] == second[1:]


def test_duration_psola_reuses_byte_identical_cached_samples() -> None:
    """Identical duration-transform requests are stable across Praat calls."""
    audio = make_signal()

    first = _psola_duration(audio, 1.4)
    second = _psola_duration(audio, 1.4)

    assert first.tobytes() == second.tobytes()


def test_long_pause_rejects_a_preexisting_long_natural_gap() -> None:
    """The pause injector refuses boundaries already occupied by a long gap."""
    timings = make_timings()
    previous_index = len(timings) - 2
    next_index = previous_index + 1
    timings[next_index] = WordTiming(
        word=timings[next_index].word,
        start_s=timings[previous_index].end_s + 0.9,
        end_s=timings[previous_index].end_s + 1.23,
        confidence=timings[next_index].confidence,
    )

    with pytest.raises(ValueError, match="no boundary below"):
        long_pause(
            make_signal(), timings, (previous_index, next_index), 0.8,
            np.random.default_rng(1),
        )


def test_compose_shifts_later_labels_and_preserves_original_timeline() -> None:
    """Earlier insertions shift rendered labels but never source intervals."""
    audio = make_signal()
    timings = make_timings()
    specs = [
        FlawSpec("long_pause", (1, 2), 0.8),
        FlawSpec("volume_dropoff", (7, 9), 0.6),
    ]

    output, updated, labels = apply_flaws(audio, timings, specs, seed=41)

    assert [label.flaw_type for label in labels] == ["long_pause", "volume_dropoff"]
    pause, volume = labels
    assert pause.original_start_s < pause.original_end_s
    assert volume.original_start_s == timings[7].start_s
    assert volume.rendered_start_s > volume.original_start_s
    assert len(updated) == len(timings)
    assert_timings_valid(updated, output.size)
    assert output.size > audio.size