"""Synthetic tests for speaker-normalized acoustic feature extraction."""

import numpy as np
from speechlens.features import (
    extract_frame_features,
    extract_recording_features,
    hz_to_semitones,
    modulation_spectrum,
)
from speechlens.features.extract import _correct_isolated_octave_jumps
from speechlens.schema import WordTiming


SAMPLE_RATE_HZ = 16000


def tone(frequency_hz: float, duration_s: float = 1.0, amplitude: float = 0.2) -> np.ndarray:
    """Create a reproducible float32 sine wave."""
    times = np.arange(round(SAMPLE_RATE_HZ * duration_s), dtype=np.float64) / SAMPLE_RATE_HZ
    return (amplitude * np.sin(2.0 * np.pi * frequency_hz * times)).astype(np.float32)


def test_180_hz_sine_has_expected_f0_and_zero_speaker_relative_semitones() -> None:
    """A stable voiced tone centers at its own median F0."""
    frames = extract_frame_features(
        tone(180.0),
        [WordTiming(word="tone", start_s=0.0, end_s=1.0)],
    )
    voiced = frames.loc[frames["voiced"]]

    assert 178.0 <= float(voiced["f0_hz"].median()) <= 182.0
    assert abs(float(voiced["f0_semitones"].median())) < 0.05
    assert frames.shape[0] == 101
    assert abs(float(np.median(np.diff(frames["time_s"]))) - 0.01) < 1e-9


def test_doubled_pitch_is_twelve_semitones_above_reference() -> None:
    """The logarithmic pitch conversion reports an octave as twelve semitones."""
    assert float(hz_to_semitones(np.array([360.0]), reference_hz=180.0)[0]) == 12.0


def test_isolated_octave_jump_is_corrected() -> None:
    """A one-frame octave slip bracketed by stable F0 is replaced locally."""
    source = np.array([180.0, 181.0, 362.0, 182.0, 180.0])
    corrected = _correct_isolated_octave_jumps(
        source,
        target_semitones=12.0,
        tolerance_semitones=2.0,
        neighbor_tolerance_semitones=2.0,
    )

    assert 180.0 < corrected[2] < 183.0
    np.testing.assert_array_equal(corrected[[0, 1, 3, 4]], source[[0, 1, 3, 4]])


def test_six_db_gain_changes_raw_intensity_but_not_normalized_intensity() -> None:
    """Per-recording median centering cancels a constant recording gain."""
    original = tone(180.0, duration_s=2.0)
    gain_factor = 10.0 ** (6.0 / 20.0)
    gained = (original * gain_factor).astype(np.float32)
    timings = [WordTiming(word="tone", start_s=0.0, end_s=2.0)]
    original_frames = extract_frame_features(original, timings)
    gained_frames = extract_frame_features(gained, timings)

    raw_shift = float(
        (gained_frames["intensity_raw_dbfs"] - original_frames["intensity_raw_dbfs"]).median()
    )
    normalized_difference = np.nanmax(
        np.abs(gained_frames["intensity_db"] - original_frames["intensity_db"])
    )
    assert abs(raw_shift - 6.0) < 0.02
    assert normalized_difference < 1e-5


def test_amplitude_modulated_noise_has_four_hz_modulation_peak() -> None:
    """A four-Hz amplitude envelope produces a nearby modulation-spectrum peak."""
    rng = np.random.default_rng(43)
    sample_rate_hz = 1000
    duration_s = 10.0
    times = np.arange(round(sample_rate_hz * duration_s)) / sample_rate_hz
    carrier = rng.normal(size=times.size)
    audio = (1.0 + 0.7 * np.sin(2.0 * np.pi * 4.0 * times)) * carrier
    envelope_frames = np.sqrt(
        np.mean(audio[: (audio.size // 10) * 10].reshape(-1, 10) ** 2, axis=1)
    )
    _, peak_hz, _, _ = modulation_spectrum(envelope_frames, sample_rate_hz=100.0, frequency_band_hz=(3.0, 6.0))

    assert abs(peak_hz - 4.0) <= 0.2


def test_silence_has_no_voiced_frames() -> None:
    """Silence is not assigned pitch by the Praat voicing track."""
    frames = extract_frame_features(
        np.zeros(SAMPLE_RATE_HZ, dtype=np.float32),
        [WordTiming(word="silence", start_s=0.0, end_s=1.0)],
    )

    assert not frames["voiced"].any()
    assert (frames["f0_hz"] == 0.0).all()


def test_non_word_frames_receive_minus_one_word_index() -> None:
    """Frame centers outside an aligned word use the documented sentinel."""
    frames = extract_frame_features(
        tone(180.0),
        [WordTiming(word="tone", start_s=0.2, end_s=0.8)],
    )

    assert (frames.loc[frames["time_s"] < 0.2, "word_index"] == -1).all()
    assert (frames.loc[frames["time_s"] >= 0.2, "word_index"] == -1).any()


def test_full_feature_bundle_is_deterministic_and_has_word_phrase_outputs() -> None:
    """Identical input yields identical frames and aggregates on repeated calls."""
    audio = np.concatenate((tone(180.0, 0.5), np.zeros(8000), tone(190.0, 0.5)))
    timings = [
        WordTiming(word="hello,", start_s=0.0, end_s=0.45),
        WordTiming(word="world.", start_s=1.0, end_s=1.45),
    ]
    first = extract_recording_features(audio, timings, "hello, world.")
    second = extract_recording_features(audio, timings, "hello, world.")

    assert first.frames.equals(second.frames)
    assert first.words.equals(second.words)
    assert first.phrases.equals(second.phrases)
    assert first.pauses.equals(second.pauses)
    assert first.pauses.iloc[0]["boundary_type"] == "mid_phrase_boundary"
    assert len(first.phrases) == 1
    assert first.modulation == second.modulation


def test_punctuation_pause_is_counted_in_preceding_phrase_ratio() -> None:
    """A pause after sentence punctuation belongs to the phrase that precedes it."""
    audio = np.concatenate((tone(180.0, 0.4), np.zeros(6400), tone(180.0, 0.4)))
    timings = [
        WordTiming(word="one.", start_s=0.0, end_s=0.35),
        WordTiming(word="two", start_s=0.75, end_s=1.1),
    ]
    bundle = extract_recording_features(audio, timings, "one. two")

    assert len(bundle.phrases) == 2
    assert bundle.pauses.iloc[0]["boundary_type"] == "punctuation_boundary"
    assert bundle.phrases.iloc[0]["pause_count"] == 1
    assert bundle.phrases.iloc[1]["pause_count"] == 0
    assert bundle.phrases.iloc[0]["pause_ratio"] > 0.0