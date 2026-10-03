"""Deterministic six-dimension speech delivery scoring."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from math import exp, log
from typing import Any

import numpy as np
import pandas as pd

from speechlens.detection.measurements import normalize_token
from speechlens.features import FeatureBundle
from speechlens.schema import WordTiming

DIMENSION_NAMES = (
    "pacing",
    "pausing_fluency",
    "intonation",
    "energy",
    "articulation",
    "disfluency",
)
SCORING_WEIGHTS = {
    "pacing": 0.20,
    "pausing_fluency": 0.20,
    "intonation": 0.15,
    "energy": 0.15,
    "articulation": 0.15,
    "disfluency": 0.15,
}

RATE_LOG_SCALE = log(1.20)
F0_LOG_SCALE = log(1.50)
PAUSE_EXCESS_SCALE_S = 0.50
ENERGY_DROP_SCALE_DB = 6.0
ROBUST_Z_SCALE = 2.0
DETECTOR_EVIDENCE_SCALE = 2.0
SCORE_ROUND_DIGITS = 6
PENALTY_CAP = 20.0
FRAME_STEP_FALLBACK_S = 0.01
_FLOAT_FLOOR = 1e-6
_PUNCTUATION = re.compile(r"[.!?;:,][\"')\]]*$")


def aggregate_reference_folds(reference_artifact: Mapping[str, Any]) -> dict:
    """Combine stored DEV leave-one-passage-out folds without reading TEST data."""
    source = reference_artifact.get("realistic", reference_artifact)
    folds = list(source.values())
    if not folds:
        raise ValueError("DEV-IDEAL reference artifact contains no folds")
    positions = sorted({
        position
        for fold in folds
        for position in fold.get("position_statistics", {})
    })
    position_statistics: dict[str, dict[str, dict[str, float]]] = {}
    for position in positions:
        features = sorted({
            feature
            for fold in folds
            for feature in fold.get("position_statistics", {}).get(position, {})
        })
        position_statistics[position] = {}
        for feature in features:
            stats = [
                fold["position_statistics"][position][feature]
                for fold in folds
                if feature in fold.get("position_statistics", {}).get(position, {})
            ]
            position_statistics[position][feature] = {
                "median": float(np.median([item["median"] for item in stats])),
                "mad": float(np.median([item["mad"] for item in stats])),
            }
    return {
        "schema_version": 3,
        "fit_source": "aggregate of stored DEV IDEAL leave-one-passage-out folds",
        "position_statistics": position_statistics,
    }


def _word_values(bundle: FeatureBundle, index: int) -> dict[str, float]:
    if bundle.words.empty or "word_index" not in bundle.words:
        return {}
    rows = bundle.words.loc[bundle.words["word_index"] == index]
    if rows.empty:
        return {}
    row = rows.iloc[0]
    return {
        key: float(row[key])
        for key in (
            "articulation_rate_sps",
            "syllable_nuclei",
            "syllable_count_abs_difference",
            "f0_std_semitones",
            "intensity_db",
        )
        if key in row and pd.notna(row[key])
    }


def _word_positions(transcript: str, timings: Sequence[WordTiming]) -> dict[int, str]:
    transcript_words = transcript.split()
    positions = ["middle"] * len(transcript_words)
    phrase_start = 0
    for index, word in enumerate(transcript_words):
        if _PUNCTUATION.search(word) or index == len(transcript_words) - 1:
            if phrase_start <= index:
                positions[phrase_start] = "first"
                positions[index] = "last"
            phrase_start = index + 1
    mapping: dict[int, str] = {}
    matcher = SequenceMatcher(
        a=[normalize_token(word) for word in transcript_words],
        b=[normalize_token(timing.word) for timing in timings],
        autojunk=False,
    )
    for text_start, timing_start, count in matcher.get_matching_blocks():
        for offset in range(count):
            text_index = text_start + offset
            timing_index = timing_start + offset
            if text_index < len(positions):
                mapping[timing_index] = positions[text_index]
    return mapping


def _matched_pairs(
    participant_timings: Sequence[WordTiming],
    baseline_timings: Sequence[WordTiming],
) -> list[tuple[int, int]]:
    matcher = SequenceMatcher(
        a=[normalize_token(item.word) for item in baseline_timings],
        b=[normalize_token(item.word) for item in participant_timings],
        autojunk=False,
    )
    return [
        (participant_start + offset, baseline_start + offset)
        for baseline_start, participant_start, count in matcher.get_matching_blocks()
        for offset in range(count)
    ]


def _robust_z(value: float, stats: Mapping[str, float], floor: float) -> float:
    return (value - float(stats["median"])) / max(
        1.4826 * float(stats["mad"]), floor
    )


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _voiced_unassigned_fraction(bundle: FeatureBundle) -> float:
    frames = bundle.frames
    if frames.empty:
        return 0.0
    active = frames["speech_activity"].to_numpy(dtype=bool)
    voiced = frames["voiced"].to_numpy(dtype=bool)
    unassigned = frames["word_index"].to_numpy(dtype=np.int64) < 0
    return float(np.count_nonzero(active & voiced & unassigned) / max(1, np.count_nonzero(active)))


def _detector_evidence(
    regions: Sequence[Mapping], detector_types: Sequence[str], duration_s: float
) -> float:
    duration = max(duration_s, _FLOAT_FLOOR)
    return DETECTOR_EVIDENCE_SCALE * sum(
        max(0.0, float(region.get("severity", 0.0)))
        * max(0.0, float(region["end_s"]) - float(region["start_s"]))
        / duration
        for region in regions
        if region.get("type") in detector_types
    )


def _paired_penalties(
    participant_bundle: FeatureBundle,
    participant_timings: Sequence[WordTiming],
    baseline_bundle: FeatureBundle,
    baseline_timings: Sequence[WordTiming],
    transcript: str,
) -> tuple[dict[str, float], dict[str, float]]:
    pairs = _matched_pairs(participant_timings, baseline_timings)
    duration_ratios: list[float] = []
    articulation_ratios: list[float] = []
    f0_ratios: list[float] = []
    syllable_errors: list[float] = []
    level_by_participant: dict[int, float] = {}
    level_by_baseline: dict[int, float] = {}
    for participant_index, baseline_index in pairs:
        participant_timing = participant_timings[participant_index]
        baseline_timing = baseline_timings[baseline_index]
        participant_duration = participant_timing.end_s - participant_timing.start_s
        baseline_duration = baseline_timing.end_s - baseline_timing.start_s
        if participant_duration > 0.0 and baseline_duration > 0.0:
            duration_ratios.append(participant_duration / baseline_duration)
        participant_values = _word_values(participant_bundle, participant_index)
        baseline_values = _word_values(baseline_bundle, baseline_index)
        participant_rate = participant_values.get("articulation_rate_sps", 0.0)
        baseline_rate = baseline_values.get("articulation_rate_sps", 0.0)
        if participant_rate > 0.0 and baseline_rate > 0.0:
            articulation_ratios.append(participant_rate / baseline_rate)
        participant_f0 = max(participant_values.get("f0_std_semitones", 0.0), _FLOAT_FLOOR)
        baseline_f0 = max(baseline_values.get("f0_std_semitones", 0.0), _FLOAT_FLOOR)
        f0_ratios.append(participant_f0 / baseline_f0)
        participant_nuclei = participant_values.get("syllable_nuclei", 0.0)
        baseline_nuclei = baseline_values.get("syllable_nuclei", 0.0)
        if baseline_nuclei > 0.0:
            syllable_errors.append(abs(participant_nuclei - baseline_nuclei) / baseline_nuclei)
        level_by_participant[participant_index] = participant_values.get("intensity_db", 0.0)
        level_by_baseline[baseline_index] = baseline_values.get("intensity_db", 0.0)

    pause_excesses: list[float] = []
    energy_drops: list[float] = []
    transcript_words = transcript.split()
    for (participant_index, baseline_index), (next_participant, next_baseline) in zip(pairs, pairs[1:]):
        if next_participant != participant_index + 1 or next_baseline != baseline_index + 1:
            continue
        if participant_index < len(transcript_words) and _PUNCTUATION.search(transcript_words[participant_index]):
            continue
        participant_gap = max(
            0.0,
            participant_timings[next_participant].start_s - participant_timings[participant_index].end_s,
        )
        baseline_gap = max(
            0.0,
            baseline_timings[next_baseline].start_s - baseline_timings[baseline_index].end_s,
        )
        pause_excesses.append(max(0.0, participant_gap - baseline_gap))
        participant_drop = max(
            0.0,
            level_by_participant.get(participant_index, 0.0)
            - level_by_participant.get(next_participant, 0.0),
        )
        baseline_drop = max(
            0.0,
            level_by_baseline.get(baseline_index, 0.0)
            - level_by_baseline.get(next_baseline, 0.0),
        )
        energy_drops.append(max(0.0, participant_drop - baseline_drop))

    duration_s = float(participant_bundle.summary.get("duration_s", 0.0))
    participant_unassigned = _voiced_unassigned_fraction(participant_bundle)
    baseline_unassigned = _voiced_unassigned_fraction(baseline_bundle)
    penalties = {
        "pacing": _mean([abs(log(max(ratio, _FLOAT_FLOOR))) / RATE_LOG_SCALE for ratio in duration_ratios]),
        "pausing_fluency": _mean(pause_excesses) / PAUSE_EXCESS_SCALE_S,
        "intonation": _mean([abs(log(max(ratio, _FLOAT_FLOOR))) / F0_LOG_SCALE for ratio in f0_ratios]),
        "energy": _mean(energy_drops) / ENERGY_DROP_SCALE_DB,
        "articulation": (
            _mean([abs(log(max(ratio, _FLOAT_FLOOR))) / RATE_LOG_SCALE for ratio in articulation_ratios])
            + _mean(syllable_errors)
        ),
        "disfluency": max(0.0, participant_unassigned - baseline_unassigned),
    }
    features = {
        "matched_word_count": float(len(pairs)),
        "mean_duration_ratio": _mean(duration_ratios) if duration_ratios else 1.0,
        "mean_articulation_rate_ratio": _mean(articulation_ratios) if articulation_ratios else 1.0,
        "mean_f0_spread_ratio": _mean(f0_ratios) if f0_ratios else 1.0,
        "mean_pause_excess_s": _mean(pause_excesses),
        "mean_extra_downward_db": _mean(energy_drops),
        "syllable_count_relative_error": _mean(syllable_errors),
        "unassigned_voiced_fraction_excess": max(0.0, participant_unassigned - baseline_unassigned),
        "duration_s": duration_s,
    }
    return penalties, features


def _reference_free_penalties(
    bundle: FeatureBundle,
    timings: Sequence[WordTiming],
    transcript: str,
    reference: Mapping,
    robust_floor: float,
) -> tuple[dict[str, float], dict[str, float]]:
    positions = _word_positions(transcript, timings)
    position_stats = reference.get("position_statistics", {})
    duration_s = float(bundle.summary.get("duration_s", 0.0))
    word_duration = float(np.median([
        timing.end_s - timing.start_s for timing in timings
    ])) if timings else 0.0
    rate_z: list[float] = []
    pitch_z: list[float] = []
    articulation_z: list[float] = []
    pause_z: list[float] = []
    energy_z: list[float] = []
    disfluency_z: list[float] = []
    word_levels: dict[int, float] = {}
    frames = bundle.frames
    frame_times = frames["time_s"].to_numpy(dtype=np.float64)
    unassigned_frames = frames["word_index"].to_numpy(dtype=np.int64) < 0
    active_voiced_frames = (
        frames["speech_activity"].to_numpy(dtype=bool)
        & frames["voiced"].to_numpy(dtype=bool)
    )
    frame_step_s = (
        float(np.median(np.diff(frame_times)))
        if len(frame_times) > 1
        else FRAME_STEP_FALLBACK_S
    )
    word_levels = {
        index: _word_values(bundle, index).get("intensity_db", 0.0)
        for index in range(len(timings))
    }
    for index, timing in enumerate(timings):
        stats = position_stats.get(positions.get(index, "middle"), position_stats.get("middle", {}))
        local_ratio = (timing.end_s - timing.start_s) / max(word_duration, _FLOAT_FLOOR)
        duration_stats = stats.get("duration_ratio")
        if duration_stats:
            z = _robust_z(local_ratio, duration_stats, robust_floor)
            rate_z.append(abs(z))
            articulation_z.append(abs(z))
        values = _word_values(bundle, index)
        f0_stats = stats.get("f0_std")
        if f0_stats:
            pitch_z.append(max(0.0, -_robust_z(values.get("f0_std_semitones", 0.0), f0_stats, robust_floor)))
        if index + 1 < len(timings):
            gap = max(0.0, timings[index + 1].start_s - timing.end_s)
            pause_stats = stats.get("pause_s")
            if pause_stats:
                pause_z.append(max(0.0, _robust_z(gap, pause_stats, robust_floor)))
                gap_mask = (
                    (frame_times >= timing.end_s)
                    & (frame_times < timings[index + 1].start_s)
                    & unassigned_frames
                    & active_voiced_frames
                )
                voiced_gap_s = float(np.count_nonzero(gap_mask)) * frame_step_s
                disfluency_z.append(
                    max(0.0, _robust_z(voiced_gap_s, pause_stats, robust_floor))
                )
            volume_stats = stats.get("volume_drop_db")
            if volume_stats:
                downward_db = max(0.0, word_levels[index] - word_levels[index + 1])
                energy_z.append(
                    max(0.0, _robust_z(downward_db, volume_stats, robust_floor))
                )
    unassigned = _voiced_unassigned_fraction(bundle)
    penalties = {
        "pacing": _mean(rate_z) / ROBUST_Z_SCALE,
        "pausing_fluency": _mean(pause_z) / ROBUST_Z_SCALE,
        "intonation": _mean(pitch_z) / ROBUST_Z_SCALE,
        "energy": _mean(energy_z) / ROBUST_Z_SCALE,
        "articulation": _mean(articulation_z) / ROBUST_Z_SCALE,
        "disfluency": max(unassigned, _mean(disfluency_z) / ROBUST_Z_SCALE),
    }
    return penalties, {
        "duration_s": duration_s,
        "mean_duration_ratio": _mean([
            (timing.end_s - timing.start_s) / max(word_duration, _FLOAT_FLOOR)
            for timing in timings
        ]) if timings else 1.0,
        "mean_unassigned_voiced_fraction": unassigned,
        "reference_fit_source": str(reference.get("fit_source", "DEV IDEAL")),
    }


def score_recording(
    participant_bundle: FeatureBundle,
    participant_timings: Sequence[WordTiming],
    transcript: str,
    mode: str,
    detector_regions: Sequence[Mapping],
    config: Mapping,
    *,
    baseline_bundle: FeatureBundle | None = None,
    baseline_timings: Sequence[WordTiming] | None = None,
    reference: Mapping | None = None,
) -> dict:
    """Calculate fixed weighted dimensions from paired deltas or DEV robust z-scores."""
    if mode not in {"paired", "reference_free"}:
        raise ValueError("mode must be 'paired' or 'reference_free'")
    if mode == "paired":
        if baseline_bundle is None or baseline_timings is None:
            raise ValueError("paired scoring requires ideal features and timings")
        penalties, features = _paired_penalties(
            participant_bundle,
            participant_timings,
            baseline_bundle,
            baseline_timings,
            transcript,
        )
    else:
        if reference is None:
            raise ValueError("reference-free scoring requires DEV-IDEAL statistics")
        penalties, features = _reference_free_penalties(
            participant_bundle,
            participant_timings,
            transcript,
            reference,
            float(config["measurements"]["robust_scale_floor"]),
        )

    detector_map = {
        "pacing": ("pace_fast", "pace_slow"),
        "pausing_fluency": ("long_pause",),
        "intonation": ("monotone",),
        "energy": ("volume_dropoff",),
        "articulation": ("pace_fast", "pace_slow"),
        "disfluency": ("filler", "stumble_repeat"),
    }
    detector_penalties = {
        dimension: _detector_evidence(
            detector_regions,
            tuple(
                flaw_type
                for flaw_type in flaw_types
                if config["detectors"][flaw_type].get("enabled_modes", {}).get(
                    mode, config["detectors"][flaw_type].get("enabled", True)
                )
            ),
            float(features.get("duration_s", 0.0)),
        )
        for dimension, flaw_types in detector_map.items()
    }
    dimension_scores = {
        name: round(100.0 * exp(-min(PENALTY_CAP, max(0.0, penalties[name] + detector_penalties[name]))), SCORE_ROUND_DIGITS)
        for name in DIMENSION_NAMES
    }
    total = round(sum(
        SCORING_WEIGHTS[name] * dimension_scores[name]
        for name in DIMENSION_NAMES
    ), SCORE_ROUND_DIGITS)
    if mode == "paired":
        dimension_reliability = {name: "direct paired ideal comparison" for name in DIMENSION_NAMES}
    else:
        dimension_reliability = {
            "pacing": "limited: reference-free pace detectors are disabled",
            "pausing_fluency": "moderate: robust DEV pause reference and enabled long-pause detector",
            "intonation": "limited: reference-free monotone detector is disabled",
            "energy": "limited: reference-free volume detector is disabled",
            "articulation": "limited: reference uses duration-ratio proxy, not lexical correctness",
            "disfluency": "moderate: voiced-gap cue and enabled filler detector; lexical content is unknown",
        }
    return {
        "mode": mode,
        "scores": dimension_scores,
        "total": total,
        "weights": dict(SCORING_WEIGHTS),
        "penalties": {
            name: round(float(penalties[name] + detector_penalties[name]), SCORE_ROUND_DIGITS)
            for name in DIMENSION_NAMES
        },
        "detector_penalties": {
            name: round(float(value), SCORE_ROUND_DIGITS)
            for name, value in detector_penalties.items()
        },
        "features": {
            key: round(float(value), SCORE_ROUND_DIGITS)
            if isinstance(value, (int, float, np.number))
            else value
            for key, value in features.items()
        },
        "dimension_reliability": dimension_reliability,
    }


__all__ = [
    "DIMENSION_NAMES",
    "SCORING_WEIGHTS",
    "aggregate_reference_folds",
    "score_recording",
]
