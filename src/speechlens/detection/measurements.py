"""Independent, interpretable per-type acoustic measurements."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
import re

import numpy as np
import pandas as pd

from speechlens.features import FeatureBundle
from speechlens.schema import WordTiming


DETECTOR_TYPES = (
    "pace_fast", "pace_slow", "long_pause", "monotone",
    "volume_dropoff", "filler", "stumble_repeat",
)
_PUNCTUATION = re.compile(r"[.!?;:,][\"')\]]*$")
_NORMALIZER = re.compile(r"[^a-z0-9']+")


def normalize_token(token: str) -> str:
    """Normalize punctuation and case for sequence alignment."""
    return _NORMALIZER.sub("", token.casefold())


def measure_duration_ratio(
    participant_durations_s: Sequence[float],
    ideal_durations_s: Sequence[float],
) -> float:
    """Return the local participant-to-IDEAL duration ratio."""
    ideal_total = float(np.sum(ideal_durations_s))
    if ideal_total <= 0:
        raise ValueError("ideal duration must be positive")
    return float(np.sum(participant_durations_s)) / ideal_total


def measure_f0_std_ratio(
    participant_f0: Sequence[float], ideal_f0: Sequence[float], floor: float = 0.1
) -> float:
    """Return participant/ideal voiced semitone standard-deviation ratio."""
    participant = np.asarray(participant_f0, dtype=np.float64)
    ideal = np.asarray(ideal_f0, dtype=np.float64)
    participant = participant[np.isfinite(participant)]
    ideal = ideal[np.isfinite(ideal)]
    if floor <= 0:
        raise ValueError("floor must be positive")
    if not participant.size or not ideal.size:
        return 1.0
    participant_std = max(float(np.std(participant)), floor)
    ideal_std = max(float(np.std(ideal)), floor)
    return participant_std / ideal_std


def measure_intensity_drop_db(values_db: Sequence[float]) -> float:
    """Return first-minus-last intensity level across a speech window."""
    values = np.asarray(values_db, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values[0] - values[-1]) if values.size >= 2 else 0.0


def _robust_stats(values: Sequence[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    median = float(np.median(data)) if data.size else 0.0
    mad = float(np.median(np.abs(data - median))) if data.size else 0.0
    return {"median": median, "mad": mad}


def _robust_z(value: float, stats: Mapping[str, float], floor: float) -> float:
    return (value - float(stats["median"])) / max(1.4826 * float(stats["mad"]), floor)


def _lexical_timings(timings: Sequence[WordTiming]) -> list[tuple[int, WordTiming]]:
    return [(index, item) for index, item in enumerate(timings) if item.word.strip() != "*"]


def _position_classes(transcript: str) -> list[str]:
    words = transcript.split()
    result = ["middle"] * len(words)
    start = 0
    for index, word in enumerate(words):
        if _PUNCTUATION.search(word) or index == len(words) - 1:
            result[start] = "first"
            result[index] = "last"
            start = index + 1
    return result


def _word_rows(
    bundle: FeatureBundle,
    timings: Sequence[WordTiming],
    transcript: str,
    *,
    intensity_reference_median_db: float | None = None,
) -> list[dict]:
    frames = bundle.frames
    lexical = _lexical_timings(timings)
    transcript_words = transcript.split()
    transcript_tokens = [normalize_token(word) for word in transcript_words]
    timing_tokens = [normalize_token(timing.word) for _, timing in lexical]
    alignment = SequenceMatcher(
        a=transcript_tokens, b=timing_tokens, autojunk=False
    )
    timing_to_text: dict[int, int] = {}
    for text_start, timing_start, count in alignment.get_matching_blocks():
        for offset in range(count):
            timing_to_text[lexical[timing_start + offset][0]] = text_start + offset
    positions = _position_classes(transcript)
    frame_indices = frames["word_index"].to_numpy(dtype=np.int64)
    active = frames["speech_activity"].to_numpy(dtype=bool)
    voiced = frames["voiced"].to_numpy(dtype=bool)
    f0 = frames["f0_semitones"].to_numpy(dtype=np.float64)
    intensity_column = (
        "intensity_raw_dbfs" if "intensity_raw_dbfs" in frames else "intensity_db"
    )
    intensity = frames[intensity_column].to_numpy(dtype=np.float64)
    active_levels = intensity[active & np.isfinite(intensity)]
    own_median_db = float(np.median(active_levels)) if active_levels.size else 0.0
    target_median_db = (
        own_median_db
        if intensity_reference_median_db is None
        else intensity_reference_median_db
    )
    intensity = intensity - own_median_db + target_median_db
    result = []
    for frame_index, timing in lexical:
        mask = frame_indices == frame_index
        pitch = f0[mask & voiced & np.isfinite(f0)]
        level = intensity[mask & active & np.isfinite(intensity)]
        text_index = timing_to_text.get(frame_index)
        result.append({
            "frame_index": frame_index,
            "text_index": text_index,
            "token": normalize_token(timing.word),
            "position": positions[text_index] if text_index is not None else "middle",
            "start_s": float(timing.start_s),
            "end_s": float(timing.end_s),
            "duration_s": float(timing.end_s - timing.start_s),
            "confidence": float(timing.confidence) if timing.confidence is not None else 1.0,
            "f0": pitch,
            "f0_std": float(np.std(pitch)) if pitch.size else 0.0,
            "intensity_db": float(np.median(level)) if level.size else 0.0,
        })
    return result


def fit_reference_free(
    ideal_records: Sequence[Mapping],
    window_sizes: Sequence[int],
) -> dict:
    """Fit robust feature distributions from DEV-IDEAL 3-to-5-word windows."""
    samples: dict[str, dict[str, list[float]]] = {}
    widths = tuple(sorted({int(width) for width in window_sizes}))
    if not widths or any(width < 2 for width in widths):
        raise ValueError("reference window_sizes must contain widths of at least two words")
    for record in ideal_records:
        words = _word_rows(record["bundle"], record["timings"], record["transcript"])
        if not words:
            continue
        passage_duration = float(np.median([word["duration_s"] for word in words]))
        for width in widths:
            for start in range(len(words) - width + 1):
                window = words[start : start + width]
                group = samples.setdefault(window[0]["position"], {
                    "duration_ratio": [], "f0_std": [], "pause_s": [],
                    "volume_drop_db": [],
                })
                group["duration_ratio"].append(
                    float(np.mean([item["duration_s"] for item in window]))
                    / max(passage_duration, 1e-6)
                )
                window_f0 = np.concatenate([item["f0"] for item in window])
                group["f0_std"].append(
                    float(np.std(window_f0)) if window_f0.size else 0.0
                )
                group["volume_drop_db"].append(
                    measure_intensity_drop_db([item["intensity_db"] for item in window])
                )
        transcript_words = record["transcript"].split()
        for index, (first, second) in enumerate(zip(words, words[1:])):
            if first["text_index"] is not None and _PUNCTUATION.search(transcript_words[first["text_index"]]):
                continue
            group = samples.setdefault(first["position"], {
                "duration_ratio": [], "f0_std": [], "pause_s": [],
                "volume_drop_db": [],
            })
            group["pause_s"].append(
                max(0.0, second["start_s"] - first["end_s"])
            )
    return {
        "schema_version": 3,
        "fit_source": "DEV IDEAL recordings only",
        "position_statistics": {
            position: {
                feature: _robust_stats(values)
                for feature, values in sorted(features.items())
            }
            for position, features in sorted(samples.items())
        },
    }


def _matched_word_indices(
    participant: Sequence[dict], ideal: Sequence[dict]
) -> dict[int, int]:
    matcher = SequenceMatcher(
        a=[word["token"] for word in ideal],
        b=[word["token"] for word in participant],
        autojunk=False,
    )
    mapping: dict[int, int] = {}
    for ideal_start, participant_start, count in matcher.get_matching_blocks():
        for offset in range(count):
            mapping[participant_start + offset] = ideal_start + offset
    return mapping


def _word_frame_mask(frames: pd.DataFrame, word_indices: Sequence[int]) -> np.ndarray:
    return frames["word_index"].isin(word_indices).to_numpy(dtype=bool)


def _time_mask(frames: pd.DataFrame, start_s: float, end_s: float) -> np.ndarray:
    times = frames["time_s"].to_numpy(dtype=np.float64)
    return (times >= start_s) & (times < end_s)


def _raise_score(table: pd.DataFrame, mask: np.ndarray, flaw_type: str, value: float) -> None:
    if value > 0 and np.any(mask):
        column = f"score_{flaw_type}"
        table.loc[mask, column] = np.maximum(table.loc[mask, column], value)


def build_typed_deviation_table(
    bundle: FeatureBundle,
    timings: Sequence[WordTiming],
    transcript: str,
    config: Mapping,
    *,
    ideal_bundle: FeatureBundle | None = None,
    ideal_timings: Sequence[WordTiming] | None = None,
    reference: Mapping | None = None,
    stumble_intervals: Sequence[tuple[float, float, float]] = (),
) -> pd.DataFrame:
    """Produce separate type-specific frame scores from direct measurements."""
    frames = bundle.frames
    table = frames[["time_s", "word_index"]].copy()
    for flaw_type in DETECTOR_TYPES:
        table[f"score_{flaw_type}"] = 0.0
    table["alignment_confidence"] = 1.0
    table["explicit_filler"] = False
    table["nonlexical_voiced"] = False
    table["rate_ratio"] = 1.0
    table["f0_std_ratio"] = 1.0
    table["intensity_drop_db"] = 0.0
    table["pause_excess_s"] = 0.0

    reference_median_db = None
    if ideal_bundle is not None:
        ideal_frames = ideal_bundle.frames
        ideal_activity = ideal_frames["speech_activity"].to_numpy(dtype=bool)
        intensity_column = (
            "intensity_raw_dbfs"
            if "intensity_raw_dbfs" in ideal_frames
            else "intensity_db"
        )
        ideal_levels = ideal_frames[intensity_column].to_numpy(dtype=np.float64)
        selected_ideal_levels = ideal_levels[ideal_activity & np.isfinite(ideal_levels)]
        if selected_ideal_levels.size:
            reference_median_db = float(np.median(selected_ideal_levels))
    words = _word_rows(
        bundle,
        timings,
        transcript,
        intensity_reference_median_db=reference_median_db,
    )
    ideal_words = (
        _word_rows(
            ideal_bundle,
            ideal_timings,
            transcript,
            intensity_reference_median_db=reference_median_db,
        )
        if ideal_bundle is not None and ideal_timings is not None
        else []
    )
    paired = bool(ideal_words)
    mapping = _matched_word_indices(words, ideal_words) if paired else {}
    measurement = config["measurements"]
    refs = (reference or {}).get("position_statistics", {})
    robust_floor = float(measurement["robust_scale_floor"])
    pace = measurement["pace"]
    widths = tuple(int(value) for value in pace["window_words"])
    for participant_index, word in enumerate(words):
        frame_mask = frames["word_index"].to_numpy(dtype=np.int64) == word["frame_index"]
        table.loc[frame_mask, "alignment_confidence"] = word["confidence"]

    for width in widths:
        for start in range(len(words) - width + 1):
            window = words[start : start + width]
            window_mask = _word_frame_mask(frames, [item["frame_index"] for item in window])
            if paired:
                ideal_indices = [mapping.get(start + offset) for offset in range(width)]
                if any(index is None for index in ideal_indices):
                    continue
                expected = [ideal_words[int(index)] for index in ideal_indices]
                ratio = measure_duration_ratio(
                    [item["duration_s"] for item in window],
                    [item["duration_s"] for item in expected],
                )
                participant_f0 = np.concatenate([item["f0"] for item in window])
                expected_f0 = np.concatenate([item["f0"] for item in expected])
                f0_ratio = measure_f0_std_ratio(
                    participant_f0,
                    expected_f0,
                    float(measurement["monotone"]["f0_std_floor"]),
                )
                observed_drop = measure_intensity_drop_db([item["intensity_db"] for item in window])
                expected_drop = measure_intensity_drop_db([item["intensity_db"] for item in expected])
                volume_drop = max(0.0, observed_drop - expected_drop)
            else:
                position = window[0]["position"]
                stats = refs.get(position, refs.get("middle", {}))
                passage_word_duration = float(np.median([item["duration_s"] for item in words]))
                ratio = float(np.mean([item["duration_s"] for item in window])) / max(passage_word_duration, 1e-6)
                rate_z = _robust_z(
                    ratio,
                    stats.get("duration_ratio", {"median": 1.0, "mad": 0.0}),
                    robust_floor,
                )
                participant_f0 = np.concatenate([item["f0"] for item in window])
                f0_z = _robust_z(
                    float(np.std(participant_f0)) if participant_f0.size else 0.0,
                    stats.get("f0_std", {"median": 0.0, "mad": 0.0}),
                    robust_floor,
                )
                f0_ratio = float(np.exp(-f0_z))
                observed_drop = measure_intensity_drop_db([item["intensity_db"] for item in window])
                volume_drop = max(
                    0.0,
                    _robust_z(
                        observed_drop,
                        stats.get("volume_drop_db", {"median": 0.0, "mad": 0.0}),
                        robust_floor,
                    ),
                )
            fast_score = max(0.0, 1.0 - ratio) if paired else max(0.0, -rate_z)
            slow_score = max(0.0, ratio - 1.0) if paired else max(0.0, rate_z)
            monotone_score = max(0.0, 1.0 - f0_ratio) if paired else max(0.0, -f0_z)
            _raise_score(table, window_mask, "pace_fast", fast_score)
            _raise_score(table, window_mask, "pace_slow", slow_score)
            _raise_score(table, window_mask, "monotone", monotone_score)
            _raise_score(table, window_mask, "volume_dropoff", volume_drop)
            table.loc[window_mask, "rate_ratio"] = ratio
            table.loc[window_mask, "f0_std_ratio"] = f0_ratio
            table.loc[window_mask, "intensity_drop_db"] = volume_drop


    pause = measurement["long_pause"]
    transcript_words = transcript.split()
    punctuation_after = {
        index for index, token in enumerate(transcript_words) if _PUNCTUATION.search(token)
    }
    for index, (first, second) in enumerate(zip(words, words[1:])):
        if first["text_index"] in punctuation_after:
            continue
        participant_gap = max(0.0, second["start_s"] - first["end_s"])
        ideal_gap = 0.0
        if paired and index in mapping and index + 1 in mapping and mapping[index + 1] == mapping[index] + 1:
            ideal_gap = max(
                0.0,
                ideal_words[mapping[index] + 1]["start_s"] - ideal_words[mapping[index]]["end_s"],
            )
        elif not paired:
            stats = refs.get(first["position"], refs.get("middle", {}))
            ideal_gap = float(stats.get("pause_s", {}).get("median", 0.0))
        if ideal_gap >= float(pause["natural_pause_s"]):
            continue
        excess = max(0.0, participant_gap - ideal_gap - float(pause["minimum_excess_s"]))
        if excess:
            gap_mask = _time_mask(frames, first["end_s"], second["start_s"])
            _raise_score(table, gap_mask, "long_pause", excess)
            table.loc[gap_mask, "pause_excess_s"] = excess

    _mark_filler_scores(
        table,
        bundle,
        words,
        measurement["filler"],
        ideal_bundle=ideal_bundle if paired else None,
        ideal_words=ideal_words,
        mapping=mapping,
    )
    _mark_stumble_scores(table, stumble_intervals)
    table["nonlexical_voiced"] = table.get("nonlexical_voiced", False)
    return table


def detect_typed_regions(
    table: pd.DataFrame,
    timings: Sequence[WordTiming],
    config: Mapping,
    flaw_types: Sequence[str] | None = None,
    *,
    mode: str | None = None,
) -> list[dict]:
    """Run all independent type thresholds and return type-tagged regions."""
    from speechlens.detection.core import hysteresis_regions, snap_interval_to_words

    if table.empty:
        return []
    times = table["time_s"].to_numpy(dtype=np.float64)
    frame_step_s = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
    smoothing_count = max(1, round(float(config["frame_smoothing_s"]) / frame_step_s))
    kernel = np.ones(smoothing_count, dtype=np.float64) / smoothing_count
    boundaries = [
        edge for timing in timings for edge in (timing.start_s, timing.end_s)
    ]
    regions: list[dict] = []
    selected_types = DETECTOR_TYPES if flaw_types is None else tuple(flaw_types)
    if any(flaw_type not in DETECTOR_TYPES for flaw_type in selected_types):
        raise ValueError("flaw_types contains an unsupported detector")
    selected_types = tuple(
        flaw_type
        for flaw_type in selected_types
        if (
            config["detectors"][flaw_type].get("enabled_modes", {}).get(
                mode, config["detectors"][flaw_type].get("enabled", True)
            )
            if mode is not None
            else config["detectors"][flaw_type].get("enabled", True)
        )
    )
    for flaw_type in selected_types:
        settings = config["detectors"][flaw_type]
        thresholds = settings.get("thresholds_by_mode", {}).get(mode, {}) if mode else {}
        scores = table[f"score_{flaw_type}"].fillna(0.0).to_numpy(dtype=np.float64)
        smoothed = np.convolve(scores, kernel, mode="same")
        intervals = hysteresis_regions(
            smoothed,
            times,
            float(thresholds.get("enter_threshold", settings["enter_threshold"])),
            float(thresholds.get("exit_threshold", settings["exit_threshold"])),
            float(settings["minimum_duration_s"]),
            float(settings["merge_gap_s"]),
        )
        for raw_start, raw_end in intervals:
            start_s, end_s = snap_interval_to_words(
                raw_start,
                raw_end,
                boundaries,
                float(config["boundary_tolerance_s"]),
            )
            mask = (times >= raw_start) & (times < raw_end)
            row_slice = table.loc[mask]
            if row_slice.empty:
                continue
            score = float(row_slice[f"score_{flaw_type}"].max())
            confidence_values = row_slice["alignment_confidence"].to_numpy(dtype=np.float64)
            low_threshold = float(config["low_alignment_confidence"])
            low_fraction = float(np.mean(confidence_values < low_threshold))
            confidence = float(np.mean(confidence_values)) * max(
                0.0, 1.0 - float(config["low_confidence_penalty"]) * low_fraction
            )
            word_indices = sorted(
                {
                    int(index)
                    for index in row_slice["word_index"].to_numpy()
                    if int(index) >= 0
                }
            )
            words = [
                timings[index].word
                for index in word_indices
                if index < len(timings) and timings[index].word != "*"
            ]
            regions.append({
                "start_s": start_s,
                "end_s": end_s,
                "raw_start_s": raw_start,
                "raw_end_s": raw_end,
                "words": words,
                "word_indices": word_indices,
                "type": flaw_type,
                "features": {
                    "score": score,
                    "rate_ratio": float(row_slice["rate_ratio"].median()),
                    "f0_std_ratio": float(row_slice["f0_std_ratio"].median()),
                    "intensity_drop_db": float(row_slice["intensity_drop_db"].max()),
                    "pause_excess_s": float(row_slice["pause_excess_s"].max()),
                    "explicit_filler": bool(row_slice["explicit_filler"].any()),
                },
                "max_deviation": score,
                "severity": float(np.clip(
                    score / float(settings["severity_scale"]), 0.0, 1.0
                )),
                "confidence": float(np.clip(confidence, 0.0, 1.0)),
            })
    return sorted(regions, key=lambda item: (item["start_s"], item["type"]))


def _mark_filler_scores(
    table: pd.DataFrame,
    bundle: FeatureBundle,
    words: Sequence[dict],
    settings: Mapping,
    *,
    ideal_bundle: FeatureBundle | None = None,
    ideal_words: Sequence[dict] = (),
    mapping: Mapping[int, int] | None = None,
) -> None:
    """Mark stable voiced segments in plain-alignment gaps as filler candidates.

    Paired mode requires more unassigned voicing in the participant gap than the
    matched IDEAL gap, preventing identical self-comparisons from becoming flaws.
    """
    frames = bundle.frames
    times = frames["time_s"].to_numpy(dtype=np.float64)
    step_s = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
    f0 = frames["f0_semitones"].to_numpy(dtype=np.float64)
    voiced = frames["voiced"].to_numpy(dtype=bool)
    activity = frames["speech_activity"].to_numpy(dtype=bool)
    unassigned = frames["word_index"].to_numpy(dtype=np.int64) < 0
    ideal_voiced_gap_duration: dict[tuple[int, int], float] = {}
    if ideal_bundle is not None and mapping is not None:
        ideal_frames = ideal_bundle.frames
        ideal_times = ideal_frames["time_s"].to_numpy(dtype=np.float64)
        ideal_unassigned = ideal_frames["word_index"].to_numpy(dtype=np.int64) < 0
        ideal_voiced = ideal_frames["voiced"].to_numpy(dtype=bool)
        ideal_activity = ideal_frames["speech_activity"].to_numpy(dtype=bool)
        ideal_step_s = (
            float(np.median(np.diff(ideal_times))) if len(ideal_times) > 1 else step_s
        )
        for participant_index, ideal_index in mapping.items():
            next_ideal_index = mapping.get(participant_index + 1)
            if next_ideal_index != ideal_index + 1:
                continue
            if ideal_index + 1 >= len(ideal_words):
                continue
            ideal_first = ideal_words[ideal_index]
            ideal_second = ideal_words[ideal_index + 1]
            ideal_gap = (
                (ideal_times >= ideal_first["end_s"])
                & (ideal_times < ideal_second["start_s"])
                & ideal_unassigned
                & ideal_voiced
                & ideal_activity
            )
            ideal_voiced_gap_duration[(participant_index, participant_index + 1)] = (
                float(np.count_nonzero(ideal_gap)) * ideal_step_s
            )
    minimum_duration_s = float(settings["minimum_duration_s"])
    maximum_f0_std = float(settings["maximum_f0_std_semitones"])
    for first, second in zip(words, words[1:]):
        gap_mask = (times >= first["end_s"]) & (times < second["start_s"])
        voiced_gap = gap_mask & unassigned & voiced & activity & np.isfinite(f0)
        runs = _runs(voiced_gap)
        paired_gap = ideal_bundle is not None and mapping is not None
        participant_gap_duration = sum((end - start) * step_s for start, end in runs)
        expected_gap_duration = ideal_voiced_gap_duration.get(
            (first["frame_index"], second["frame_index"]), 0.0
        ) if paired_gap else 0.0
        excess_duration = participant_gap_duration - expected_gap_duration
        if paired_gap and excess_duration < minimum_duration_s:
            continue
        for start, end in runs:
            duration_s = min((end - start) * step_s, excess_duration) if paired_gap else (end - start) * step_s
            if duration_s < minimum_duration_s:
                continue
            if float(np.std(f0[start:end])) > maximum_f0_std:
                continue
            segment_mask = np.zeros(len(times), dtype=bool)
            segment_mask[start:end] = True
            _raise_score(table, segment_mask, "filler", duration_s)
            table.loc[segment_mask, "explicit_filler"] = True
            table.loc[segment_mask, "nonlexical_voiced"] = True


def _mark_stumble_scores(
    table: pd.DataFrame,
    intervals: Sequence[tuple[float, float, float]],
) -> None:
    """Mark only ASR-grounded inserted-token intervals as stumble candidates."""
    frames = table["time_s"].to_numpy(dtype=np.float64)
    for start_s, end_s, confidence in intervals:
        if not np.isfinite([start_s, end_s, confidence]).all() or end_s <= start_s:
            continue
        mask = (frames >= start_s) & (frames < end_s)
        _raise_score(table, mask, "stumble_repeat", float(np.clip(confidence, 0.0, 1.0)))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    selected = np.flatnonzero(mask)
    if not selected.size:
        return []
    breaks = np.flatnonzero(np.diff(selected) > 1)
    starts = np.r_[selected[0], selected[breaks + 1]]
    ends = np.r_[selected[breaks], selected[-1]] + 1
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


