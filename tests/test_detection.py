"""Synthetic tests for temporal detection primitives and rule classification."""

import numpy as np
import json
import tempfile
from pathlib import Path
import librosa
import pandas as pd

from speechlens.detection.core import (
    _paired_timing_indices,
    classify_feature_row,
    hysteresis_regions,
    pause_is_flaggable,
    snap_interval_to_words,
)
from speechlens.features import extract_frame_features
from speechlens.schema import WordTiming
from speechlens.explain import (
    explanation_record,
    render_explanation,
    render_explanations_json,
)
from scripts.eval_detection import (
    _cached_forced_alignment,
    _cached_recording_features,
    _transcript_with_wildcards,
)
from speechlens.detection.measurements import (
    detect_typed_regions,
    measure_duration_ratio,
    measure_f0_std_ratio,
    measure_intensity_drop_db,
)


def test_hysteresis_requires_enter_then_uses_exit_threshold() -> None:
    """Scores between thresholds extend, but cannot start, a region."""
    times = np.arange(8, dtype=np.float64) * 0.1
    scores = [0.8, 1.2, 2.1, 1.1, 0.9, 1.4, 0.7, 0.0]

    regions = hysteresis_regions(scores, times, 2.0, 1.0, 0.2, 0.0)

    assert regions == [(0.2, 0.4)]


def test_hysteresis_merges_nearby_regions_and_drops_short_regions() -> None:
    """Near regions merge before minimum-duration filtering."""
    times = np.arange(10, dtype=np.float64) * 0.1
    scores = [0, 2.1, 2.1, 0, 0, 2.2, 2.2, 0, 0, 0]

    regions = hysteresis_regions(scores, times, 2.0, 1.0, 0.4, 0.2)

    np.testing.assert_allclose(regions, [(0.1, 0.7)])


def test_boundary_snapping_only_moves_edges_within_tolerance() -> None:
    """Each edge snaps to its closest eligible word boundary."""
    assert snap_interval_to_words(1.08, 2.17, [1.0, 2.0, 2.2], 0.1) == (1.0, 2.2)
    assert snap_interval_to_words(1.2, 2.17, [1.0, 2.0, 2.2], 0.1) == (1.2, 2.2)


def test_word_alignment_keeps_later_anchors_after_inserted_repeat() -> None:
    """An extra timed word does not shift subsequent paired word matches."""
    ideal = [
        WordTiming(word="we", start_s=0.0, end_s=0.1),
        WordTiming(word="go", start_s=0.2, end_s=0.3),
        WordTiming(word="home.", start_s=0.4, end_s=0.6),
    ]
    participant = [
        WordTiming(word="we", start_s=0.0, end_s=0.1),
        WordTiming(word="go", start_s=0.2, end_s=0.3),
        WordTiming(word="go", start_s=0.4, end_s=0.5),
        WordTiming(word="home.", start_s=0.6, end_s=0.8),
    ]

    assert _paired_timing_indices(ideal, participant, 2) == {0: 0, 1: 1, 3: 2}


def test_duplicate_alignment_prefers_original_timing_occurrence() -> None:
    """Keep the original word aligned and leave the following repeat unmatched."""
    ideal = [
        WordTiming(word="know", start_s=0.0, end_s=0.2),
        WordTiming(word="alexander", start_s=0.3, end_s=0.7),
        WordTiming(word="mainhall", start_s=0.8, end_s=1.1),
    ]
    participant = [
        WordTiming(word="know", start_s=0.0, end_s=0.2),
        WordTiming(word="alexander", start_s=0.3, end_s=0.7),
        WordTiming(word="alexander", start_s=1.3, end_s=1.7),
        WordTiming(word="mainhall", start_s=1.8, end_s=2.1),
    ]

    assert _paired_timing_indices(ideal, participant, 2) == {0: 0, 1: 1, 3: 2}


def test_sequence_alignment_recognizes_repeated_two_word_phrase() -> None:
    """A copied two-word sequence stays unmatched as a single repeat event."""
    ideal = [
        WordTiming(word="the", start_s=0.0, end_s=0.1),
        WordTiming(word="zeal", start_s=0.1, end_s=0.3),
        WordTiming(word="returns", start_s=0.4, end_s=0.7),
    ]
    participant = [
        WordTiming(word="the", start_s=0.0, end_s=0.1),
        WordTiming(word="zeal", start_s=0.1, end_s=0.3),
        WordTiming(word="the", start_s=0.5, end_s=0.6),
        WordTiming(word="zeal", start_s=0.6, end_s=0.8),
        WordTiming(word="returns", start_s=0.9, end_s=1.2),
    ]

    assert _paired_timing_indices(ideal, participant, 2) == {0: 0, 1: 1, 4: 2}


def test_pause_at_comma_is_never_flagged_as_a_flaw() -> None:
    """Natural rhetorical pauses at punctuation are excluded."""
    assert not pause_is_flaggable("punctuation_boundary", 1.2, 0.1, 0.4, 0.45)
    assert not pause_is_flaggable("mid_phrase_boundary", 1.2, 0.6, 0.4, 0.45)
    assert pause_is_flaggable("mid_phrase_boundary", 0.8, 0.1, 0.4, 0.45)


def test_rule_classifier_uses_feature_patterns_in_documented_order() -> None:
    """Hand-built feature rows select each supported transparent rule."""
    thresholds = {
        "repeat_score": 0.9,
        "pause_excess_s": 0.5,
        "rate_z": 2.0,
        "monotone_f0_std_ratio": 0.45,
        "intensity_drop_z": 2.0,
    }
    assert classify_feature_row({"rate_z": 2.8}, thresholds) == "pace_fast"
    assert classify_feature_row({"rate_z": -2.2}, thresholds) == "pace_slow"
    assert classify_feature_row({"pause_excess_s": 0.8}, thresholds) == "long_pause"
    assert classify_feature_row({"f0_std_ratio": 0.3}, thresholds) == "monotone"
    assert classify_feature_row({"intensity_z": -3.0}, thresholds) == "volume_dropoff"
    assert classify_feature_row({"nonlexical_voiced": True}, thresholds) == "filler"
    assert classify_feature_row({"repeat_score": 0.95}, thresholds) == "stumble_repeat"
    assert classify_feature_row({}, thresholds) == "other_deviation"


def test_explanation_rendering_is_byte_deterministic_and_contains_numbers() -> None:
    """Identical input yields identical structured JSON and rendered text."""
    region = {
        "start_s": 42.1,
        "end_s": 45.8,
        "words": ["we", "shall", "never"],
        "type": "pace_fast",
        "features": {"rate_z": 2.8},
        "observed_rate_sps": 4.9,
        "expected_rate_sps": 3.1,
        "severity": 0.7,
        "confidence": 0.9,
    }
    first_record = explanation_record(region)
    second_record = explanation_record(region)

    assert render_explanations_json([first_record]).encode() == render_explanations_json(
        [second_record]
    ).encode()
    assert render_explanation(first_record).encode() == render_explanation(
        second_record
    ).encode()
    assert "4.9 syllables/s vs 3.1 expected (z=+2.8)" in first_record["sentence"]
    assert first_record["suggestion"]


def test_filler_explanation_includes_numeric_expected_value() -> None:
    """Every explanation type includes the numeric fields required by schema."""
    record = explanation_record(
        {
            "start_s": 1.0,
            "end_s": 1.4,
            "words": [],
            "type": "filler",
            "severity": 0.3,
            "confidence": 0.8,
        }
    )

    assert np.isclose(record["observed_numeric"], 0.4)
    assert record["expected_numeric"] == 0.0
    assert record["formula"]


def test_each_flaw_type_has_a_complete_explanation_record() -> None:
    """All public classes satisfy the structured explanation contract."""
    flaw_types = (
        "pace_fast", "pace_slow", "long_pause", "monotone", "volume_dropoff",
        "filler", "stumble_repeat", "other_deviation",
    )
    for flaw_type in flaw_types:
        record = explanation_record(
            {
                "start_s": 2.0,
                "end_s": 2.5,
                "words": ["sample"],
                "type": flaw_type,
                "features": {"rate_z": 2.0, "repeat_score": 1.0},
                "observed_rate_sps": 4.0,
                "expected_rate_sps": 3.0,
                "severity": 0.5,
                "confidence": 0.8,
            }
        )
        assert record["formula"]
        assert isinstance(record["expected_numeric"], float)
        assert record["sentence"] in record["rendered_text"]


def test_wildcard_alignment_cache_uses_audio_transcript_and_reuses_result(monkeypatch) -> None:
    """A cache miss aligns the wildcard transcript and a hit avoids real alignment."""
    import scripts.eval_detection as evaluator

    audio = np.zeros(1600, dtype=np.float32)
    expected_timing = WordTiming(word="hello", start_s=0.0, end_s=0.1, confidence=0.9)
    calls = []

    def fake_align(waveform, text):
        calls.append((waveform.copy(), text))
        return [expected_timing]

    monkeypatch.setattr(evaluator, "align", fake_align)
    with tempfile.TemporaryDirectory(prefix="speechlens-align-cache-", dir=Path.cwd()) as temp_dir:
        cache_dir = Path(temp_dir)
        first = _cached_forced_alignment(audio, "one two three", cache_dir, 2)
        monkeypatch.setattr(
            evaluator,
            "align",
            lambda *_: (_ for _ in ()).throw(AssertionError("cache was not used")),
        )
        second = _cached_forced_alignment(audio, "one two three", cache_dir, 2)

        assert _transcript_with_wildcards("one two three", 2) == "one two * three"
        assert calls[0][1] == "one two * three"
        assert first == second == [expected_timing]
        assert len(list(cache_dir.glob("*.json"))) == 1


def test_recording_feature_cache_reuses_precomputed_bundle(monkeypatch) -> None:
    """Feature extraction is memoized per transcript and timing payload."""
    import scripts.eval_detection as evaluator

    audio = np.zeros(1600, dtype=np.float32)
    timings = [WordTiming(word="hello", start_s=0.0, end_s=0.1), WordTiming(word="world", start_s=0.1, end_s=0.2)]
    calls = 0

    def fake_extract(waveform, word_timings, transcript_text):
        nonlocal calls
        calls += 1
        return {"payload": 42}

    monkeypatch.setattr(evaluator, "extract_recording_features", fake_extract)
    with tempfile.TemporaryDirectory(prefix="speechlens-feature-cache-", dir=Path.cwd()) as temp_dir:
        cache_dir = Path(temp_dir)
        first = _cached_recording_features(audio, timings, "hello world", cache_dir)
        monkeypatch.setattr(
            evaluator,
            "extract_recording_features",
            lambda *_: (_ for _ in ()).throw(AssertionError("cache was not used")),
        )
        second = _cached_recording_features(audio, timings, "hello world", cache_dir)

        assert first == second == {"payload": 42}
        assert calls == 1
        assert len(list(cache_dir.glob("*.pkl"))) == 1


def test_time_stretched_synthetic_region_has_expected_duration_ratio() -> None:
    """A time-stretched synthetic tone reports its measured duration ratio."""
    sample_rate_hz = 16000
    times = np.arange(sample_rate_hz // 2, dtype=np.float64) / sample_rate_hz
    signal = (0.2 * np.sin(2.0 * np.pi * 220.0 * times)).astype(np.float32)
    stretched = librosa.effects.time_stretch(signal, rate=0.5)
    original_duration_s = len(signal) / sample_rate_hz
    stretched_duration_s = len(stretched) / sample_rate_hz

    ratio = measure_duration_ratio([stretched_duration_s], [original_duration_s])

    assert np.isclose(ratio, 2.0, atol=0.02)


def test_flattened_pitch_has_expected_f0_std_ratio() -> None:
    """Praat tracks of a pitch sweep and flattened sine report reduced F0 spread."""
    from speechlens.schema import WordTiming

    sample_rate_hz = 16000
    duration_s = 1.2
    times = np.arange(round(sample_rate_hz * duration_s)) / sample_rate_hz
    ideal_hz = np.linspace(130.0, 270.0, times.size)
    ideal_phase = 2.0 * np.pi * np.cumsum(ideal_hz) / sample_rate_hz
    ideal_audio = (0.2 * np.sin(ideal_phase)).astype(np.float32)
    flat_audio = (0.2 * np.sin(2.0 * np.pi * 200.0 * times)).astype(np.float32)
    timings = [WordTiming(word="tone", start_s=0.0, end_s=duration_s)]
    ideal_track = extract_frame_features(ideal_audio, timings)
    flat_track = extract_frame_features(flat_audio, timings)
    ideal_f0_semitones = ideal_track.loc[ideal_track["voiced"], "f0_semitones"]
    flattened_f0_semitones = flat_track.loc[flat_track["voiced"], "f0_semitones"]

    ratio = measure_f0_std_ratio(flattened_f0_semitones, ideal_f0_semitones)

    assert ratio < 0.2


def test_synthetic_gain_ramp_reports_expected_db_drop() -> None:
    """Frame RMS levels of a gain-ramped carrier yield the configured dB fall."""
    sample_rate_hz = 16000
    sample_count = sample_rate_hz
    times = np.arange(sample_count, dtype=np.float64) / sample_rate_hz
    envelope_db = np.linspace(0.0, -6.0, sample_count)
    carrier = np.sin(2.0 * np.pi * 220.0 * times)
    ramped = 0.2 * carrier * np.power(10.0, envelope_db / 20.0)
    frame_levels = [
        20.0 * np.log10(np.sqrt(np.mean(ramped[start : start + 1600] ** 2)))
        for start in range(0, sample_count, 1600)
    ]

    drop_db = measure_intensity_drop_db(frame_levels)

    assert 5.0 <= drop_db <= 6.2


def test_detection_sources_do_not_read_labels_or_sidecar_timings() -> None:
    """The detector package must remain independent of label and sidecar I/O."""
    detection_root = Path(__file__).resolve().parents[1] / "src" / "speechlens" / "detection"
    for source_path in detection_root.glob("*.py"):
        source = source_path.read_text(encoding="utf-8").casefold()
        assert "data/labels" not in source
        assert "sidecar" not in source
        assert "json.loads" not in source
    evaluator_source = (
        Path(__file__).resolve().parents[1] / "scripts" / "eval_detection.py"
    ).read_text(encoding="utf-8")
    assert "realistic_timings = _cached_forced_alignment(" in evaluator_source
    assert "oracle_timings = [WordTiming.model_validate(item) for item in sidecar[\"word_timings\"]]" in evaluator_source


def test_typed_detector_uses_only_the_crossed_type_threshold() -> None:
    """A pace-fast score cannot create regions for unrelated detector types."""
    from speechlens.detection.core import load_detection_config

    config = load_detection_config()
    config["detectors"]["pace_fast"].update(
        enter_threshold=0.1, exit_threshold=0.05, minimum_duration_s=0.1
    )
    times = np.arange(100, dtype=np.float64) * 0.01
    table = pd.DataFrame({
        "time_s": times,
        "word_index": np.zeros(times.size, dtype=np.int64),
        "alignment_confidence": np.ones(times.size),
        "explicit_filler": np.zeros(times.size, dtype=bool),
        "nonlexical_voiced": np.zeros(times.size, dtype=bool),
        "rate_ratio": np.ones(times.size),
        "f0_std_ratio": np.ones(times.size),
        "intensity_drop_db": np.zeros(times.size),
        "pause_excess_s": np.zeros(times.size),
        **{f"score_{name}": np.zeros(times.size) for name in (
            "pace_fast", "pace_slow", "long_pause", "monotone",
            "volume_dropoff", "filler", "stumble_repeat",
        )},
    })
    table.loc[20:69, "score_pace_fast"] = 0.2
    timings = [WordTiming(word="rushed", start_s=0.0, end_s=1.0)]

    regions = detect_typed_regions(table, timings, config)

    assert regions
    assert {region["type"] for region in regions} == {"pace_fast"}
