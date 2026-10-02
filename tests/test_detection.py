"""Synthetic tests for temporal detection primitives and rule classification."""

import numpy as np

from speechlens.detection.core import (
    _paired_timing_indices,
    classify_feature_row,
    hysteresis_regions,
    pause_is_flaggable,
    snap_interval_to_words,
)
from speechlens.schema import WordTiming
from speechlens.explain import (
    explanation_record,
    render_explanation,
    render_explanations_json,
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
