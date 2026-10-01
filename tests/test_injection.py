"""Property-style tests for deterministic audio flaw injection."""

import numpy as np
import pytest

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


@pytest.mark.parametrize("flaw_type", ["filler", "stumble_repeat"])
def test_randomized_effects_are_seed_deterministic(flaw_type: str) -> None:
    """Identical seeds produce byte-identical waveforms, labels, and timings."""
    first = call_injector(flaw_type, 0.8, seed=33)
    second = call_injector(flaw_type, 0.8, seed=33)

    assert first[0].tobytes() == second[0].tobytes()
    assert first[1:] == second[1:]


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