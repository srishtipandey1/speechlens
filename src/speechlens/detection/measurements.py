"""Independent, interpretable per-type acoustic measurements."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
import re

import librosa
import numpy as np
import pandas as pd

from speechlens.features import FeatureBundle
from speechlens.features.extract import MFCC_COLUMNS
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
    passage_rate_ratio: float = 1.0,
) -> float:
    """Return local duration ratio after normalizing the passage-wide rate."""
    ideal_total = float(np.sum(ideal_durations_s))
    if ideal_total <= 0 or passage_rate_ratio <= 0:
        raise ValueError("ideal duration and passage_rate_ratio must be positive")
    return float(np.sum(participant_durations_s)) / ideal_total / passage_rate_ratio


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
    return float(np.std(participant) / max(float(np.std(ideal)), floor))


def measure_intensity_drop_db(values_db: Sequence[float]) -> float:
    """Return first-minus-last intensity level across a speech window."""
    values = np.asarray(values_db, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values[0] - values[-1]) if values.size >= 2 else 0.0


def mfcc_dtw_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Return cosine-DTW similarity, clipped to the unit interval."""
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or not left.size or not right.size:
        return 0.0
    cost, path = librosa.sequence.dtw(X=left.T, Y=right.T, metric="cosine")
    mean_cost = float(cost[-1, -1]) / max(1, len(path))
    return float(np.clip(1.0 - mean_cost, 0.0, 1.0))


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
    intensity = frames["intensity_db"].to_numpy(dtype=np.float64)
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


def fit_reference_free(ideal_records: Sequence[Mapping]) -> dict:
    """Fit DEV-IDEAL robust word and phrase distributions only."""
    samples: dict[str, dict[str, list[float]]] = {}
    window_sizes = (1, 3, 5)
    for record in ideal_records:
        words = _word_rows(record["bundle"], record["timings"], record["transcript"])
        if not words:
            continue
        passage_duration = float(np.median([word["duration_s"] for word in words]))
        for word in words:
            group = samples.setdefault(word["position"], {
                "duration_ratio": [], "f0_std": [], "intensity_db": [],
                "pause_s": [], "volume_drop_db": [],
            })
            group["duration_ratio"].append(word["duration_s"] / max(passage_duration, 1e-6))
            group["f0_std"].append(word["f0_std"])
            group["intensity_db"].append(word["intensity_db"])
        for width in window_sizes:
            for start in range(max(0, len(words) - width + 1)):
                window = words[start : start + width]
                group = samples.setdefault(window[0]["position"], {
                    "duration_ratio": [], "f0_std": [], "intensity_db": [],
                    "pause_s": [], "volume_drop_db": [],
                })
                group["volume_drop_db"].append(
                    measure_intensity_drop_db([item["intensity_db"] for item in window])
                )
        transcript_words = record["transcript"].split()
        for index, (first, second) in enumerate(zip(words, words[1:])):
            if first["text_index"] is not None and _PUNCTUATION.search(transcript_words[first["text_index"]]):
                continue
            samples[first["position"]]["pause_s"].append(
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
    wildcard_spans: Sequence[tuple[float, float]] = (),
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

    words = _word_rows(bundle, timings, transcript)
    ideal_words = (
        _word_rows(ideal_bundle, ideal_timings, transcript)
        if ideal_bundle is not None and ideal_timings is not None
        else []
    )
    paired = bool(ideal_words)
    mapping = _matched_word_indices(words, ideal_words) if paired else {}
    passage_rate_ratio = 1.0
    if paired and mapping:
        participant_total = sum(words[p]["duration_s"] for p in mapping)
        ideal_total = sum(ideal_words[i]["duration_s"] for i in mapping.values())
        passage_rate_ratio = participant_total / max(ideal_total, 1e-8)

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
                    passage_rate_ratio,
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
            fast_score = max(0.0, float(pace["fast_ratio_bound"]) - ratio) if paired else max(0.0, -rate_z)
            slow_score = max(0.0, ratio - float(pace["slow_ratio_bound"])) if paired else max(0.0, rate_z)
            monotone_bound = float(measurement["monotone"]["paired_ratio_bound"])
            monotone_score = max(0.0, monotone_bound - f0_ratio) if paired else max(0.0, -f0_z)
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

    _mark_filler_scores(table, bundle, words, wildcard_spans, measurement["filler"])
    _mark_repeat_scores(
        table, bundle, words, wildcard_spans, transcript, measurement["stumble_repeat"]
    )
    table["nonlexical_voiced"] = table.get("nonlexical_voiced", False)
    return table


def detect_typed_regions(
    table: pd.DataFrame,
    timings: Sequence[WordTiming],
    config: Mapping,
    flaw_types: Sequence[str] | None = None,
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
    for flaw_type in selected_types:
        settings = config["detectors"][flaw_type]
        scores = table[f"score_{flaw_type}"].fillna(0.0).to_numpy(dtype=np.float64)
        smoothed = np.convolve(scores, kernel, mode="same")
        intervals = hysteresis_regions(
            smoothed,
            times,
            float(settings["enter_threshold"]),
            float(settings["exit_threshold"]),
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
    wildcard_spans: Sequence[tuple[float, float]],
    settings: Mapping,
) -> None:
    frames = bundle.frames
    times = frames["time_s"].to_numpy(dtype=np.float64)
    step_s = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
    word_index = frames["word_index"].to_numpy(dtype=np.int64)
    f0 = frames["f0_semitones"].to_numpy(dtype=np.float64)
    voiced = frames["voiced"].to_numpy(dtype=bool)
    activity = frames["speech_activity"].to_numpy(dtype=bool)
    known_fillers = {normalize_token(word) for word in settings["known_filler_tokens"]}
    candidates = list(wildcard_spans)
    candidates.extend(
        (word["start_s"], word["end_s"])
        for word in words
        if word["token"] in known_fillers
    )
    for start_s, end_s in candidates:
        mask = (times >= start_s) & (times < end_s)
        duration_s = end_s - start_s
        if duration_s < float(settings["minimum_duration_s"]):
            continue
        voiced_mask = mask & voiced & activity & np.isfinite(f0)
        if not np.any(voiced_mask):
            continue
        f0_std = float(np.std(f0[voiced_mask]))
        voiced_fraction = float(np.mean(voiced[mask])) if np.any(mask) else 0.0
        if (
            f0_std > float(settings["maximum_f0_std_semitones"])
            or voiced_fraction < float(settings["minimum_voiced_fraction"])
        ):
            continue
        _raise_score(table, mask, "filler", duration_s)
        table.loc[mask, "explicit_filler"] = True
        table.loc[mask, "nonlexical_voiced"] = True

    unaligned = (word_index < 0) & voiced & activity & np.isfinite(f0)
    for start, end in _runs(unaligned):
        start_s = float(times[start])
        end_s = float(times[end - 1] + step_s)
        mask = np.zeros(len(times), dtype=bool)
        mask[start:end] = True
        if end_s - start_s < float(settings["minimum_duration_s"]):
            continue
        if float(np.std(f0[mask])) > float(settings["maximum_f0_std_semitones"]):
            continue
        _raise_score(table, mask, "filler", end_s - start_s)
        table.loc[mask, "nonlexical_voiced"] = True


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    selected = np.flatnonzero(mask)
    if not selected.size:
        return []
    breaks = np.flatnonzero(np.diff(selected) > 1)
    starts = np.r_[selected[0], selected[breaks + 1]]
    ends = np.r_[selected[breaks], selected[-1]] + 1
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _mark_repeat_scores(
    table: pd.DataFrame,
    bundle: FeatureBundle,
    words: Sequence[dict],
    wildcard_spans: Sequence[tuple[float, float]],
    transcript: str,
    settings: Mapping,
) -> None:
    frames = bundle.frames
    mfcc = frames.loc[:, MFCC_COLUMNS].to_numpy(dtype=np.float64)
    times = frames["time_s"].to_numpy(dtype=np.float64)
    expected_tokens = [normalize_token(token) for token in transcript.split()]
    max_words = int(settings["maximum_sequence_words"])
    for width in range(1, max_words + 1):
        for start in range(len(words) - 2 * width + 1):
            first = words[start : start + width]
            second = words[start + width : start + 2 * width]
            first_text = [word["text_index"] for word in first]
            second_text = [word["text_index"] for word in second]
            if (
                all(index is not None for index in first_text + second_text)
                and second_text[0] == first_text[-1] + 1
                and [expected_tokens[index] for index in first_text]
                == [expected_tokens[index] for index in second_text]
            ):
                continue
            first_mfcc = mfcc[
                (times >= first[0]["start_s"]) & (times < first[-1]["end_s"])
            ]
            second_mfcc = mfcc[
                (times >= second[0]["start_s"]) & (times < second[-1]["end_s"])
            ]
            if min(len(first_mfcc), len(second_mfcc)) < int(settings["minimum_mfcc_frames"]):
                continue
            mean_similarity = _mean_mfcc_similarity(first_mfcc, second_mfcc)
            if mean_similarity < float(settings["mean_similarity_prefilter"]):
                continue
            similarity = mfcc_dtw_similarity(first_mfcc, second_mfcc)
            if similarity >= float(settings["similarity_threshold"]):
                mask = _word_frame_mask(frames, [word["frame_index"] for word in second])
                _raise_score(table, mask, "stumble_repeat", similarity)
    for start_s, end_s in wildcard_spans:
        unknown = mfcc[(times >= start_s) & (times < end_s)]
        if len(unknown) < int(settings["minimum_mfcc_frames"]):
            continue
        preceding = [word for word in words if word["end_s"] <= start_s]
        best = 0.0
        for width in range(1, min(max_words, len(preceding)) + 1):
            sequence = preceding[-width:]
            context = mfcc[
                (times >= sequence[0]["start_s"])
                & (times < sequence[-1]["end_s"])
            ]
            if len(context) < int(settings["minimum_mfcc_frames"]):
                continue
            if _mean_mfcc_similarity(context, unknown) < float(settings["mean_similarity_prefilter"]):
                continue
            best = max(best, mfcc_dtw_similarity(context, unknown))
        if best >= float(settings["similarity_threshold"]):
            _raise_score(table, (times >= start_s) & (times < end_s), "stumble_repeat", best)


def _mean_mfcc_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first_mean = np.mean(first, axis=0)
    second_mean = np.mean(second, axis=0)
    denominator = float(np.linalg.norm(first_mean) * np.linalg.norm(second_mean))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(first_mean, second_mean) / denominator)