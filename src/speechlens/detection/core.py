"""Pure temporal detection and transparent flaw classification rules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
import json
from pathlib import Path
import re

import librosa
import numpy as np
import pandas as pd
import yaml

from speechlens.features import FeatureBundle
from speechlens.features.extract import MFCC_COLUMNS
from speechlens.schema import WordTiming


FlawType = str
_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "detection.yaml"
_PUNCTUATION = re.compile(r"[.!?;:,][\"')\]]*$")
_WORD_NORMALIZER = re.compile(r"[^a-z0-9']+")


def _normalized_word(word: str) -> str:
    """Normalize timed words for transcript-position alignment."""
    return _WORD_NORMALIZER.sub("", word.casefold())


def _timing_to_transcript_indices(
    word_timings: Sequence[WordTiming], transcript_text: str
) -> dict[int, int]:
    """Map timing indices to transcript indices, leaving insertions unmatched."""
    transcript_words = transcript_text.split()
    expected = [_normalized_word(word) for word in transcript_words]
    timed = [_normalized_word(timing.word) for timing in word_timings]
    mapping: dict[int, int] = {}
    matcher = SequenceMatcher(a=expected, b=timed, autojunk=False)
    for expected_start, timing_start, size in matcher.get_matching_blocks():
        for offset in range(size):
            mapping[timing_start + offset] = expected_start + offset
    for index in range(1, len(timed)):
        if timed[index] == timed[index - 1] and index - 1 not in mapping and index in mapping:
            mapping[index - 1] = mapping.pop(index)
    return mapping


def _paired_timing_indices(
    ideal_timings: Sequence[WordTiming],
    participant_timings: Sequence[WordTiming],
    max_repeat_words: int,
) -> dict[int, int]:
    """Map participant word timings to same-token ideal timings monotonically."""
    ideal_words = [_normalized_word(timing.word) for timing in ideal_timings]
    participant_words = [_normalized_word(timing.word) for timing in participant_timings]
    mapping: dict[int, int] = {}
    matcher = SequenceMatcher(a=ideal_words, b=participant_words, autojunk=False)
    for ideal_start, participant_start, size in matcher.get_matching_blocks():
        for offset in range(size):
            mapping[participant_start + offset] = ideal_start + offset
    mapping = _prefer_earlier_repeated_sequence(
        participant_words, mapping, max_repeat_words
    )
    return mapping


def _prefer_earlier_repeated_sequence(
    words: Sequence[str], mapping: Mapping[int, int], max_repeat_words: int
) -> dict[int, int]:
    """Prefer the first occurrence when only one adjacent repeated n-gram maps."""
    adjusted = dict(mapping)
    for width in range(1, max_repeat_words + 1):
        for start in range(len(words) - 2 * width + 1):
            left = list(range(start, start + width))
            right = list(range(start + width, start + 2 * width))
            if list(words[start : start + width]) != list(
                words[start + width : start + 2 * width]
            ):
                continue
            left_unmapped = all(index not in adjusted for index in left)
            right_mapped = all(index in adjusted for index in right)
            if not left_unmapped or not right_mapped:
                continue
            targets = [adjusted[index] for index in right]
            if any(later != earlier + 1 for earlier, later in zip(targets, targets[1:])):
                continue
            for right_index, target in zip(right, targets):
                del adjusted[right_index]
            adjusted.update(zip(left, targets))
    return adjusted


def _is_unmatched_repeat(
    index: int,
    word_timings: Sequence[WordTiming],
    matched_indices: set[int],
    max_repeat_words: int,
) -> bool:
    """Identify a repeated adjacent timed token left unmatched by alignment."""
    words = [_normalized_word(timing.word) for timing in word_timings]
    for width in range(1, max_repeat_words + 1):
        for start in range(len(words) - 2 * width + 1):
            left = list(range(start, start + width))
            right = list(range(start + width, start + 2 * width))
            if words[start : start + width] != words[start + width : start + 2 * width]:
                continue
            unmatched_side = (
                left if all(item not in matched_indices for item in left)
                else right if all(item not in matched_indices for item in right)
                else []
            )
            if index in unmatched_side:
                return True
    return False


def load_detection_config(path: str | Path = _CONFIG_PATH) -> dict:
    """Load the frozen, explicit thresholds for detection and classification."""
    with Path(path).open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def _position_classes(transcript_text: str, word_count: int) -> list[str]:
    """Assign each word to first, middle, or last position in its phrase."""
    words = transcript_text.split()[:word_count]
    classes = ["middle"] * len(words)
    phrase_start = 0
    for index, word in enumerate(words):
        phrase_end = bool(_PUNCTUATION.search(word)) or index == len(words) - 1
        if phrase_end:
            phrase_indices = list(range(phrase_start, index + 1))
            if phrase_indices:
                classes[phrase_indices[0]] = "first"
                classes[phrase_indices[-1]] = "last"
            phrase_start = index + 1
    return classes


def word_feature_rows(
    bundle: FeatureBundle,
    word_timings: Sequence[WordTiming],
    transcript_text: str,
) -> list[dict[str, float | int | str]]:
    """Build feature and phrase-position rows from an extracted recording."""
    frame_table = bundle.frames
    positions = _position_classes(transcript_text, len(transcript_text.split()))
    transcript_words = transcript_text.split()
    timing_to_transcript = _timing_to_transcript_indices(word_timings, transcript_text)
    rows: list[dict[str, float | int | str]] = []
    for index, timing in enumerate(word_timings):
        mask = frame_table["word_index"].to_numpy() == index
        active = mask & frame_table["speech_activity"].to_numpy(dtype=bool)
        voiced = mask & frame_table["voiced"].to_numpy(dtype=bool)
        local_intensity = frame_table.loc[active, "intensity_db"].to_numpy(dtype=np.float64)
        local_f0 = frame_table.loc[voiced, "f0_semitones"].to_numpy(dtype=np.float64)
        next_gap_s = (
            max(0.0, word_timings[index + 1].start_s - timing.end_s)
            if index + 1 < len(word_timings)
            else 0.0
        )
        word_values = bundle.words.iloc[index]
        transcript_index = timing_to_transcript.get(index)
        if transcript_index is None and timing_to_transcript:
            nearest_timing_index = min(
                timing_to_transcript,
                key=lambda candidate: abs(candidate - index),
            )
            transcript_index = timing_to_transcript[nearest_timing_index]
        phrase_values = bundle.phrases.loc[
            (bundle.phrases["first_word_index"] <= index)
            & (bundle.phrases["end_word_index_exclusive"] > index)
        ]
        phrase = phrase_values.iloc[0] if not phrase_values.empty else {}
        rows.append(
            {
                "word_index": index,
                "word": timing.word,
                "position_class": positions[transcript_index]
                if transcript_index is not None and transcript_index < len(positions)
                else "middle",
                "start_s": timing.start_s,
                "end_s": timing.end_s,
                "rate_sps": float(word_values["articulation_rate_sps"]),
                "f0_std": float(np.std(local_f0)) if local_f0.size else 0.0,
                "intensity_db": float(np.median(local_intensity)) if local_intensity.size else 0.0,
                "pause_after_s": next_gap_s,
                "phrase_rate_sps": float(phrase.get("articulation_rate_sps", 0.0)),
                "phrase_f0_std": float(phrase.get("f0_std_semitones", 0.0)),
                "phrase_pause_ratio": float(phrase.get("pause_ratio", 0.0)),
                "alignment_confidence": float(timing.confidence)
                if timing.confidence is not None
                else 1.0,
            }
        )
    return rows


def fit_reference_free(
    ideal_word_rows: Sequence[Sequence[Mapping[str, float | int | str]]],
) -> dict:
    """Fit median/MAD word and phrase distributions from IDEAL rows only."""
    values_by_position: dict[str, dict[str, list[float]]] = {}
    phrase_features = ("rate_sps", "f0_std", "intensity_db", "pause_after_s")
    for rows in ideal_word_rows:
        for row in rows:
            position = str(row["position_class"])
            group = values_by_position.setdefault(
                position, {feature: [] for feature in phrase_features}
            )
            for feature in phrase_features:
                value = float(row[feature])
                if np.isfinite(value):
                    group[feature].append(value)
    statistics: dict[str, dict[str, dict[str, float]]] = {}
    for position, feature_values in sorted(values_by_position.items()):
        statistics[position] = {}
        for feature, values in sorted(feature_values.items()):
            median = float(np.median(values)) if values else 0.0
            mad = float(np.median(np.abs(np.asarray(values) - median))) if values else 0.0
            statistics[position][feature] = {"median": median, "mad": mad}

    phrase_statistics: dict[str, dict[str, dict[str, float]]] = {}
    for position in sorted(values_by_position):
        phrase_statistics[position] = {}
        rows = [row for recording_rows in ideal_word_rows for row in recording_rows
                if str(row["position_class"]) == position]
        for feature in ("phrase_rate_sps", "phrase_f0_std", "phrase_pause_ratio"):
            values = np.asarray([float(row[feature]) for row in rows], dtype=np.float64)
            median = float(np.median(values)) if values.size else 0.0
            mad = float(np.median(np.abs(values - median))) if values.size else 0.0
            phrase_statistics[position][feature] = {"median": median, "mad": mad}
    return {
        "schema_version": 1,
        "fit_source": "DEV IDEAL recordings only",
        "position_statistics": statistics,
        "phrase_statistics": phrase_statistics,
    }


def save_reference_artifact(reference: Mapping, path: str | Path) -> None:
    """Write a stable, compact UTF-8 JSON reference artifact."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(reference, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _robust_z(value: float, statistics: Mapping[str, float], floor: float) -> float:
    """Compute a normal-consistent robust z-score using median and MAD."""
    scale = max(1.4826 * float(statistics["mad"]), floor)
    return (value - float(statistics["median"])) / scale


def reference_free_deviation_table(
    bundle: FeatureBundle,
    word_timings: Sequence[WordTiming],
    transcript_text: str,
    reference: Mapping,
    config: Mapping,
) -> pd.DataFrame:
    """Score participant frames against the fitted DEV-IDEAL robust reference."""
    rows = word_feature_rows(bundle, word_timings, transcript_text)
    frame_table = bundle.frames.copy()
    defaults: dict[str, object] = {
        "rate_z": 0.0,
        "intensity_z": 0.0,
        "f0_std_ratio": 1.0,
        "f0_z": 0.0,
        "pause_excess_s": 0.0,
        "repeat_score": 0.0,
        "nonlexical_voiced": False,
        "explicit_filler": False,
        "expected_rate_sps": 0.0,
        "observed_rate_sps": 0.0,
        "expected_intensity_db": 0.0,
        "observed_intensity_db": 0.0,
        "expected_f0_std": 0.0,
        "observed_f0_std": 0.0,
        "phrase_rate_z": 0.0,
        "phrase_f0_std_z": 0.0,
        "phrase_pause_ratio_z": 0.0,
        "alignment_confidence": 1.0,
        "position_class": "unassigned",
    }
    for column, value in defaults.items():
        frame_table[column] = value

    position_statistics = reference["position_statistics"]
    floor = float(config["robust_scale_floor"])
    aligned_timing_indices = _prefer_earlier_repeated_sequence(
        [_normalized_word(timing.word) for timing in word_timings],
        _timing_to_transcript_indices(word_timings, transcript_text),
        int(config["repeat_sequence_max_words"]),
    )
    for row in rows:
        position = str(row["position_class"])
        stats = position_statistics.get(position, position_statistics["middle"])
        index_mask = frame_table["word_index"] == int(row["word_index"])
        expected_rate = float(stats["rate_sps"]["median"])
        expected_f0 = float(stats["f0_std"]["median"])
        observed_rate = float(row["rate_sps"])
        observed_f0 = float(row["f0_std"])
        observed_intensity = float(row["intensity_db"])
        rate_z = _robust_z(observed_rate, stats["rate_sps"], floor)
        intensity_z = _robust_z(observed_intensity, stats["intensity_db"], floor)
        f0_z = _robust_z(observed_f0, stats["f0_std"], floor)
        phrase_stats = reference["phrase_statistics"].get(
            position, reference["phrase_statistics"]["middle"]
        )
        phrase_rate_z = _robust_z(float(row["phrase_rate_sps"]), phrase_stats["phrase_rate_sps"], floor)
        phrase_f0_z = _robust_z(float(row["phrase_f0_std"]), phrase_stats["phrase_f0_std"], floor)
        phrase_pause_z = _robust_z(
            float(row["phrase_pause_ratio"]), phrase_stats["phrase_pause_ratio"], floor
        )
        f0_ratio = observed_f0 / max(expected_f0, floor)
        punctuation_boundary = bool(_PUNCTUATION.search(str(row["word"])))
        expected_pause = float(stats["pause_after_s"]["median"])
        pause_excess = 0.0
        if not punctuation_boundary and expected_pause < float(config["natural_pause_threshold_s"]):
            pause_excess = max(
                0.0,
                float(row["pause_after_s"]) - expected_pause
                - float(config["minimum_pause_excess_s"]),
            )
        frame_table.loc[index_mask, [
            "rate_z", "intensity_z", "f0_std_ratio", "f0_z", "pause_excess_s",
            "expected_rate_sps", "observed_rate_sps", "expected_intensity_db",
            "observed_intensity_db", "expected_f0_std", "observed_f0_std",
            "alignment_confidence", "position_class", "phrase_rate_z",
            "phrase_f0_std_z", "phrase_pause_ratio_z",
        ]] = [
            rate_z, intensity_z, f0_ratio, f0_z, pause_excess, expected_rate,
            observed_rate, float(stats["intensity_db"]["median"]), observed_intensity,
            expected_f0, observed_f0, float(row["alignment_confidence"]), position,
            phrase_rate_z, phrase_f0_z, phrase_pause_z,
        ]
        word_index = int(row["word_index"])
        if _is_unmatched_repeat(
            word_index,
            word_timings,
            set(aligned_timing_indices),
            int(config["repeat_sequence_max_words"]),
        ):
            frame_table.loc[index_mask, "repeat_score"] = 1.0

    pause_mask = frame_table["word_index"] < 0
    frame_table.loc[pause_mask, "deviation"] = 0.0
    nonlexical = (
        frame_table["speech_activity"].to_numpy(dtype=bool)
        & frame_table["voiced"].to_numpy(dtype=bool)
        & pause_mask.to_numpy()
    )
    filler_tokens = {_normalized_word(token) for token in config["filler_tokens"]}
    explicit_filler = np.zeros(len(frame_table), dtype=bool)
    for word_index, timing in enumerate(word_timings):
        if word_index in aligned_timing_indices or _normalized_word(timing.word) not in filler_tokens:
            continue
        word_frames = frame_table["word_index"].to_numpy() == word_index
        explicit_filler |= (
            word_frames
            & frame_table["speech_activity"].to_numpy(dtype=bool)
            & frame_table["voiced"].to_numpy(dtype=bool)
        )
    frame_table["explicit_filler"] = explicit_filler
    frame_table["nonlexical_voiced"] = nonlexical | explicit_filler
    frame_table["deviation"] = np.maximum.reduce(
        [
            np.abs(frame_table["rate_z"].to_numpy(dtype=np.float64)),
            np.abs(frame_table["intensity_z"].to_numpy(dtype=np.float64)),
            np.maximum(0.0, -frame_table["f0_z"].to_numpy(dtype=np.float64)),
            np.abs(frame_table["phrase_rate_z"].to_numpy(dtype=np.float64)),
            np.maximum(0.0, -frame_table["phrase_f0_std_z"].to_numpy(dtype=np.float64)),
            frame_table["repeat_score"].to_numpy(dtype=np.float64)
            * float(config["repeat_score_strength"]),
            explicit_filler.astype(np.float64)
            * float(config["explicit_filler_strength"]),
            nonlexical.astype(np.float64)
            * float(config["nonlexical_voiced_strength"]),
            frame_table["pause_excess_s"].to_numpy(dtype=np.float64)
            / float(config["pause_score_scale_s"]),
        ]
    )
    return frame_table


def _dtw_frame_map(
    ideal_frames: pd.DataFrame,
    participant_frames: pd.DataFrame,
    ideal_word_index: int,
    participant_word_index: int,
) -> dict[int, int]:
    """Map participant frames to ideal frames with MFCC DTW inside one word."""
    ideal_indices = np.flatnonzero(ideal_frames["word_index"].to_numpy() == ideal_word_index)
    participant_indices = np.flatnonzero(
        participant_frames["word_index"].to_numpy() == participant_word_index
    )
    if not ideal_indices.size or not participant_indices.size:
        return {}
    ideal_mfcc = ideal_frames.loc[:, MFCC_COLUMNS].to_numpy(dtype=np.float64)[ideal_indices].T
    participant_mfcc = participant_frames.loc[:, MFCC_COLUMNS].to_numpy(dtype=np.float64)[participant_indices].T
    _, path = librosa.sequence.dtw(X=ideal_mfcc, Y=participant_mfcc, metric="cosine")
    mapping: dict[int, int] = {}
    for ideal_local, participant_local in path:
        mapping[int(participant_indices[participant_local])] = int(ideal_indices[ideal_local])
    for participant_local, participant_frame in enumerate(participant_indices):
        if int(participant_frame) not in mapping:
            nearest = min(
                path.tolist(), key=lambda pair: abs(int(pair[1]) - participant_local)
            )
            mapping[int(participant_frame)] = int(ideal_indices[int(nearest[0])])
    return mapping


def paired_deviation_table(
    participant_bundle: FeatureBundle,
    ideal_bundle: FeatureBundle,
    participant_timings: Sequence[WordTiming],
    ideal_timings: Sequence[WordTiming],
    transcript_text: str,
    config: Mapping,
) -> pd.DataFrame:
    """Compare word-anchored speaker-normalized features using MFCC DTW."""
    participant = participant_bundle.frames.copy()
    ideal = ideal_bundle.frames
    participant_rows = word_feature_rows(
        participant_bundle, participant_timings, transcript_text
    )
    ideal_rows = word_feature_rows(ideal_bundle, ideal_timings, transcript_text)
    participant_to_ideal = _paired_timing_indices(
        ideal_timings,
        participant_timings,
        int(config["repeat_sequence_max_words"]),
    )
    participant["rate_z"] = 0.0
    participant["intensity_z"] = 0.0
    participant["f0_std_ratio"] = 1.0
    participant["f0_z"] = 0.0
    participant["pause_excess_s"] = 0.0
    participant["repeat_score"] = 0.0
    participant["nonlexical_voiced"] = False
    participant["alignment_confidence"] = 1.0
    participant["expected_rate_sps"] = 0.0
    participant["observed_rate_sps"] = 0.0
    participant["expected_intensity_db"] = 0.0
    participant["observed_intensity_db"] = 0.0
    participant["expected_f0_std"] = 0.0
    participant["observed_f0_std"] = 0.0
    threshold = config["classification"]
    scale = float(config["paired_rate_scale"])
    intensity_scale = float(config["paired_intensity_scale_db"])
    natural_pause_s = float(config["natural_pause_threshold_s"])

    for participant_index, ideal_index in sorted(participant_to_ideal.items()):
        participant_row = participant_rows[participant_index]
        ideal_row = ideal_rows[ideal_index]
        rate_ratio = float(participant_row["rate_sps"]) / max(float(ideal_row["rate_sps"]), 1e-8)
        rate_z = float(np.log(max(rate_ratio, 1e-8)) / scale)
        f0_ratio = float(participant_row["f0_std"]) / max(float(ideal_row["f0_std"]), 1e-8)
        intensity_delta = float(participant_row["intensity_db"]) - float(ideal_row["intensity_db"])
        confidence = float(participant_row["alignment_confidence"])
        word_mask = participant["word_index"] == participant_index
        participant.loc[word_mask, [
            "rate_z", "intensity_z", "f0_std_ratio", "alignment_confidence",
            "expected_rate_sps", "observed_rate_sps", "expected_intensity_db",
            "observed_intensity_db", "expected_f0_std", "observed_f0_std",
        ]] = [
            rate_z, intensity_delta / intensity_scale, f0_ratio, confidence,
            float(ideal_row["rate_sps"]), float(participant_row["rate_sps"]),
            float(ideal_row["intensity_db"]), float(participant_row["intensity_db"]),
            float(ideal_row["f0_std"]), float(participant_row["f0_std"]),
        ]
        mapping = _dtw_frame_map(ideal, participant, ideal_index, participant_index)
        for participant_frame, ideal_frame in mapping.items():
            ideal_pitch = float(ideal.iloc[ideal_frame]["f0_semitones"])
            participant_pitch = float(participant.iloc[participant_frame]["f0_semitones"])
            if np.isfinite(ideal_pitch) and np.isfinite(participant_pitch):
                participant.loc[participant_frame, "f0_z"] = (
                    participant_pitch - ideal_pitch
                ) / float(config["paired_pitch_scale_semitones"])

        if participant_index + 1 >= len(participant_timings):
            continue
        next_ideal_index = participant_to_ideal.get(participant_index + 1)
        if next_ideal_index != ideal_index + 1:
            continue
        participant_gap = max(
            0.0,
            participant_timings[participant_index + 1].start_s
            - participant_timings[participant_index].end_s,
        )
        ideal_gap = max(
            0.0,
            ideal_timings[ideal_index + 1].start_s - ideal_timings[ideal_index].end_s,
        )
        words = transcript_text.split()
        boundary_type = (
            "punctuation_boundary"
            if ideal_index < len(words) and _PUNCTUATION.search(words[ideal_index])
            else "mid_phrase_boundary"
        )
        if pause_is_flaggable(
            boundary_type,
            participant_gap,
            ideal_gap,
            float(config["minimum_pause_excess_s"]),
            natural_pause_s,
        ):
            excess = participant_gap - ideal_gap
            pause_mask = participant["time_s"].between(
                participant_timings[participant_index].end_s,
                participant_timings[participant_index + 1].start_s,
                inclusive="both",
            )
            participant.loc[pause_mask, "pause_excess_s"] = excess
            participant.loc[pause_mask, "alignment_confidence"] = min(
                confidence,
                float(participant_rows[participant_index + 1]["alignment_confidence"]),
            )

    for participant_index, timing in enumerate(participant_timings):
        if not _is_unmatched_repeat(
            participant_index,
            participant_timings,
            set(participant_to_ideal),
            int(config["repeat_sequence_max_words"]),
        ):
            continue
        repeated_word = participant["word_index"] == participant_index
        participant.loc[repeated_word, "repeat_score"] = 1.0

    unassigned_active = (
        (participant["word_index"] < 0)
        & participant["speech_activity"].astype(bool)
        & participant["voiced"].astype(bool)
    )
    filler_tokens = {_normalized_word(token) for token in config["filler_tokens"]}
    explicit_filler = np.zeros(len(participant), dtype=bool)
    for participant_index, timing in enumerate(participant_timings):
        if (
            participant_index in participant_to_ideal
            or _normalized_word(timing.word) not in filler_tokens
        ):
            continue
        word_frames = participant["word_index"] == participant_index
        explicit_filler |= (
            word_frames.to_numpy()
            & participant["speech_activity"].to_numpy(dtype=bool)
            & participant["voiced"].to_numpy(dtype=bool)
        )
    generic_unassigned = unassigned_active.to_numpy()
    participant["explicit_filler"] = explicit_filler
    participant["nonlexical_voiced"] = generic_unassigned | explicit_filler
    monotone_score = np.maximum(
        0.0,
        float(threshold["monotone_f0_std_ratio"])
        - participant["f0_std_ratio"].to_numpy(dtype=np.float64),
    ) / float(config["monotone_score_scale"])
    participant["deviation"] = np.maximum.reduce(
        [
            np.abs(participant["rate_z"].to_numpy(dtype=np.float64)),
            np.abs(participant["intensity_z"].to_numpy(dtype=np.float64)),
            np.abs(participant["f0_z"].to_numpy(dtype=np.float64)),
            monotone_score,
            participant["repeat_score"].to_numpy(dtype=np.float64)
            * float(config["repeat_score_strength"]),
            explicit_filler.astype(np.float64)
            * float(config["explicit_filler_strength"]),
            generic_unassigned.astype(np.float64)
            * float(config["nonlexical_voiced_strength"]),
            participant["pause_excess_s"].to_numpy(dtype=np.float64)
            / float(config["pause_score_scale_s"]),
        ]
    )
    return participant


def detect_deviation_regions(
    deviation_table: pd.DataFrame,
    word_timings: Sequence[WordTiming],
    config: Mapping,
) -> list[dict]:
    """Compatibility wrapper using independent thresholds for every type."""
    from speechlens.detection.measurements import detect_typed_regions

    return detect_typed_regions(deviation_table, word_timings, config)


def hysteresis_regions(
    values: Sequence[float],
    times_s: Sequence[float],
    enter_threshold: float,
    exit_threshold: float,
    minimum_duration_s: float,
    merge_gap_s: float,
) -> list[tuple[float, float]]:
    """Return smoothed-score regions using enter/exit hysteresis and merging.

    Values are expected to have been smoothed by the caller. The interval end
    is exclusive and extends one frame beyond the final selected sample.
    """
    scores = np.asarray(values, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64)
    if scores.ndim != 1 or times.ndim != 1 or scores.size != times.size:
        raise ValueError("values and times_s must be equal-length one-dimensional arrays")
    if not scores.size:
        return []
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("times_s must be finite and strictly increasing")
    if exit_threshold > enter_threshold:
        raise ValueError("exit_threshold must not exceed enter_threshold")
    if minimum_duration_s < 0 or merge_gap_s < 0:
        raise ValueError("durations must be non-negative")

    frame_step_s = float(np.median(np.diff(times))) if times.size > 1 else 0.01
    raw_regions: list[tuple[float, float]] = []
    start_index: int | None = None
    for index, score in enumerate(scores):
        if start_index is None and score >= enter_threshold:
            start_index = index
        elif start_index is not None and score < exit_threshold:
            raw_regions.append((float(times[start_index]), float(times[index])))
            start_index = None
    if start_index is not None:
        raw_regions.append((float(times[start_index]), float(times[-1] + frame_step_s)))

    merged: list[list[float]] = []
    for start_s, end_s in raw_regions:
        if merged and start_s - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = end_s
        else:
            merged.append([start_s, end_s])
    return [
        (start_s, end_s)
        for start_s, end_s in merged
        if end_s - start_s >= minimum_duration_s
    ]


def snap_interval_to_words(
    start_s: float,
    end_s: float,
    word_boundaries_s: Sequence[float],
    tolerance_s: float,
) -> tuple[float, float]:
    """Snap each interval edge independently to a nearby word boundary."""
    if start_s >= end_s:
        raise ValueError("start_s must be less than end_s")
    if tolerance_s < 0:
        raise ValueError("tolerance_s must be non-negative")
    boundaries = np.asarray(word_boundaries_s, dtype=np.float64)
    if not boundaries.size:
        return start_s, end_s
    if not np.isfinite(boundaries).all():
        raise ValueError("word boundaries must be finite")

    def nearest(value: float) -> float:
        candidate = float(boundaries[np.argmin(np.abs(boundaries - value))])
        return candidate if abs(candidate - value) <= tolerance_s else value

    snapped_start, snapped_end = nearest(start_s), nearest(end_s)
    if snapped_start >= snapped_end:
        return start_s, end_s
    return snapped_start, snapped_end


def pause_is_flaggable(
    boundary_type: str,
    participant_pause_s: float,
    ideal_pause_s: float,
    minimum_excess_s: float,
    natural_pause_threshold_s: float,
) -> bool:
    """Exclude punctuation pauses and pauses already present in the ideal."""
    if boundary_type == "punctuation_boundary":
        return False
    if ideal_pause_s >= natural_pause_threshold_s:
        return False
    return participant_pause_s - ideal_pause_s >= minimum_excess_s


def classify_feature_row(
    row: Mapping[str, float | int | bool | str],
    thresholds: Mapping[str, float],
) -> FlawType:
    """Classify an aggregated deviation row with ordered, explicit rules."""
    if float(row.get("repeat_score", 0.0)) >= thresholds["repeat_score"]:
        return "stumble_repeat"
    if float(row.get("pause_excess_s", 0.0)) >= thresholds["pause_excess_s"]:
        return "long_pause"
    if bool(row.get("nonlexical_voiced", False)):
        return "filler"
    if float(row.get("rate_z", 0.0)) >= thresholds["rate_z"]:
        if float(row.get("pause_excess_s", 0.0)) < thresholds["pause_excess_s"]:
            return "pace_fast"
    if float(row.get("rate_z", 0.0)) <= -thresholds["rate_z"]:
        return "pace_slow"
    if float(row.get("f0_std_ratio", 1.0)) <= thresholds["monotone_f0_std_ratio"]:
        return "monotone"
    if float(row.get("intensity_z", 0.0)) <= -thresholds["intensity_drop_z"]:
        return "volume_dropoff"
    return "other_deviation"
