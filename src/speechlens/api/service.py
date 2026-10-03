"""Offline alignment, detection, scoring, and plot-series assembly for the API."""

from __future__ import annotations

import io
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from scripts.eval_detection import _cached_plain_alignment
from speechlens.alignment.align import SAMPLE_RATE_HZ, align, has_cached_mms_fa_model
from speechlens.detection.core import load_detection_config
from speechlens.detection.measurements import build_typed_deviation_table, detect_typed_regions
from speechlens.explain import explanation_record
from speechlens.features import FeatureBundle, extract_recording_features
from speechlens.schema import WordTiming
from speechlens.scoring import aggregate_reference_folds, score_recording

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_PATH = PROJECT_ROOT / "eval" / "results" / "reference_free_ideal_stats.json"
TEST_METRICS_PATH = PROJECT_ROOT / "eval" / "results" / "test_detection_type_metrics.csv"
DEMO_DIR = PROJECT_ROOT / "data" / "processed" / "demo"
_PUNCTUATION = re.compile(r"[.!?;:,][\"')\]]*$")


def alignment_model_available() -> bool:
    """Check local MMS_FA weight availability without loading or downloading it."""
    return has_cached_mms_fa_model()


def decode_audio_bytes(audio_bytes: bytes) -> np.ndarray:
    """Decode an upload into finite mono float32 audio at the project sample rate."""
    if not audio_bytes:
        raise ValueError("audio upload is empty")
    samples, sample_rate_hz = sf.read(
        io.BytesIO(audio_bytes), dtype="float32", always_2d=True
    )
    waveform = samples.mean(axis=1, dtype=np.float32)
    if sample_rate_hz != SAMPLE_RATE_HZ:
        waveform = librosa.resample(
            waveform,
            orig_sr=sample_rate_hz,
            target_sr=SAMPLE_RATE_HZ,
        ).astype(np.float32, copy=False)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("decoded audio must be finite and non-empty")
    return np.ascontiguousarray(waveform, dtype=np.float32)


def load_test_detection_metrics(project_root: Path = PROJECT_ROOT) -> dict[str, dict[str, dict[str, float]]]:
    """Load held-out IoU 0.3 precision/recall required for region reliability."""
    path = project_root / "eval" / "results" / "test_detection_type_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(
            "held-out detector metrics are missing; run the frozen TEST evaluator first"
        )
    frame = pd.read_csv(path)
    required = {"mode", "timing_source", "flaw_type", "iou_threshold", "precision", "recall"}
    if not required.issubset(frame.columns):
        raise ValueError(f"held-out metrics file is missing columns: {sorted(required - set(frame.columns))}")
    selected = frame.loc[
        (frame["timing_source"] == "realistic")
        & np.isclose(frame["iou_threshold"].astype(float), 0.3)
    ]
    result: dict[str, dict[str, dict[str, float]]] = {}
    for row in selected.to_dict(orient="records"):
        result.setdefault(str(row["mode"]), {})[str(row["flaw_type"])] = {
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
        }
    return result


def load_reference_statistics(project_root: Path = PROJECT_ROOT) -> dict:
    """Load and aggregate only the persisted DEV-IDEAL LOO reference folds."""
    path = project_root / "eval" / "results" / "reference_free_ideal_stats.json"
    if not path.is_file():
        raise FileNotFoundError("DEV-IDEAL reference artifact is missing; run the DEV evaluator first")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    return aggregate_reference_folds(artifact)


def _frame_series(bundle: FeatureBundle, column: str, mask_column: str | None = None) -> list[dict]:
    frames = bundle.frames
    values = frames[column].to_numpy(dtype=np.float64)
    selected = (
        frames[mask_column].to_numpy(dtype=bool)
        if mask_column is not None
        else np.ones(len(frames), dtype=bool)
    )
    times = frames["time_s"].to_numpy(dtype=np.float64)
    return [
        {"time_s": float(time), "value": float(value) if selected[index] and np.isfinite(value) else None}
        for index, (time, value) in enumerate(zip(times, values))
    ]


def _word_series(bundle: FeatureBundle) -> list[dict]:
    if bundle.words.empty:
        return []
    return [
        {
            "start_s": float(row["start_s"]),
            "end_s": float(row["end_s"]),
            "value": float(row["articulation_rate_sps"]),
            "word": str(row["word"]),
        }
        for row in bundle.words.to_dict(orient="records")
    ]


def _reference_word_rate_series(
    bundle: FeatureBundle,
    timings: Sequence[WordTiming],
    transcript: str,
    reference: Mapping,
) -> list[dict]:
    """Render expected speech rates from DEV position duration medians."""
    words = transcript.split()
    positions = ["middle"] * len(words)
    start = 0
    for index, word in enumerate(words):
        if _PUNCTUATION.search(word) or index == len(words) - 1:
            if start <= index:
                positions[start] = "first"
                positions[index] = "last"
            start = index + 1
    stats_by_position = reference.get("position_statistics", {})
    durations = [timing.end_s - timing.start_s for timing in timings]
    passage_duration = float(np.median(durations)) if durations else 0.0
    result = []
    for index, timing in enumerate(timings):
        row = bundle.words.loc[bundle.words["word_index"] == index]
        nuclei = float(row.iloc[0]["syllable_nuclei"]) if not row.empty else 0.0
        position = positions[min(index, len(positions) - 1)] if positions else "middle"
        stats = stats_by_position.get(position, stats_by_position.get("middle", {}))
        expected_ratio = float(stats.get("duration_ratio", {}).get("median", 1.0))
        expected_duration = max(passage_duration * expected_ratio, 1e-6)
        result.append({
            "start_s": float(timing.start_s),
            "end_s": float(timing.end_s),
            "value": nuclei / expected_duration,
            "word": timing.word,
        })
    return result


def _series_pair(
    participant_bundle: FeatureBundle,
    baseline_bundle: FeatureBundle | None,
    participant_timings: Sequence[WordTiming],
    baseline_timings: Sequence[WordTiming] | None,
    transcript: str,
    reference: Mapping | None,
) -> dict:
    participant = {
        "f0_semitones": _frame_series(participant_bundle, "f0_semitones", "voiced"),
        "energy_db": _frame_series(participant_bundle, "intensity_db", "speech_activity"),
        "speech_rate_sps": _word_series(participant_bundle),
    }
    if baseline_bundle is not None and baseline_timings is not None:
        baseline = {
            "f0_semitones": _frame_series(baseline_bundle, "f0_semitones", "voiced"),
            "energy_db": _frame_series(baseline_bundle, "intensity_db", "speech_activity"),
            "speech_rate_sps": _word_series(baseline_bundle),
            "source": "provided ideal audio",
        }
    else:
        baseline = {
            "f0_semitones": [
                {"time_s": point["time_s"], "value": 0.0 if point["value"] is not None else None}
                for point in participant["f0_semitones"]
            ],
            "energy_db": [
                {"time_s": point["time_s"], "value": 0.0 if point["value"] is not None else None}
                for point in participant["energy_db"]
            ],
            "speech_rate_sps": _reference_word_rate_series(
                participant_bundle,
                participant_timings,
                transcript,
                reference or {},
            ),
            "source": "speaker-relative DEV-IDEAL reference",
        }
    return {"participant": participant, "baseline": baseline}


def _region_reliability(
    flaw_type: str,
    mode: str,
    metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict:
    held_out = metrics.get(mode, {}).get(flaw_type)
    if held_out is None:
        return {
            "level": "low",
            "test_precision": None,
            "test_recall": None,
            "iou_threshold": 0.3,
            "reason": "held-out precision/recall row is unavailable",
        }
    precision = float(held_out["precision"])
    level = "high" if precision >= 0.8 else "medium" if precision >= 0.4 else "low"
    return {
        "level": level,
        "test_precision": precision,
        "test_recall": float(held_out["recall"]),
        "iou_threshold": 0.3,
    }


def analyze_audio_bytes(
    audio_bytes: bytes,
    transcript: str,
    ideal_audio_bytes: bytes | None = None,
    *,
    project_root: Path = PROJECT_ROOT,
    test_metrics: Mapping[str, Mapping[str, Mapping[str, float]]] | None = None,
    alignment_fn=None,
) -> dict:
    """Analyze one upload in paired mode or DEV-reference mode without network access."""
    transcript = transcript.strip()
    if not transcript:
        raise ValueError("transcript must not be empty")
    if alignment_fn is None:
        alignment_fn = align
    root = project_root.resolve()
    config = load_detection_config(root / "config" / "detection.yaml")
    audio = decode_audio_bytes(audio_bytes)
    timings = alignment_fn(audio, transcript)
    if not timings:
        raise ValueError("plain forced alignment returned no word timings")
    participant_bundle = extract_recording_features(audio, timings, transcript)
    ideal_bundle = None
    ideal_timings = None
    reference = None
    if ideal_audio_bytes is not None:
        ideal_audio = decode_audio_bytes(ideal_audio_bytes)
        ideal_timings = alignment_fn(ideal_audio, transcript)
        if not ideal_timings:
            raise ValueError("plain forced alignment returned no ideal word timings")
        ideal_bundle = extract_recording_features(ideal_audio, ideal_timings, transcript)
        mode = "paired"
    else:
        reference = load_reference_statistics(root)
        mode = "reference_free"
    table = build_typed_deviation_table(
        participant_bundle,
        timings,
        transcript,
        config,
        ideal_bundle=ideal_bundle,
        ideal_timings=ideal_timings,
        reference=reference,
    )
    regions = detect_typed_regions(table, timings, config, mode=mode)
    score = score_recording(
        participant_bundle,
        timings,
        transcript,
        mode,
        regions,
        config,
        baseline_bundle=ideal_bundle,
        baseline_timings=ideal_timings,
        reference=reference,
    )
    metrics = (
        dict(test_metrics)
        if test_metrics is not None
        else load_test_detection_metrics(root)
    )
    explained_regions = []
    for region in regions:
        explanation = explanation_record(region)
        explanation["reliability"] = _region_reliability(
            str(region["type"]), mode, metrics
        )
        explained_regions.append(explanation)
    enabled = [
        flaw_type
        for flaw_type, detector in config["detectors"].items()
        if detector.get("enabled_modes", {}).get(mode, detector.get("enabled", True))
    ]
    disabled = {
        flaw_type: detector.get("disabled_reasons_by_mode", {}).get(
            mode, detector.get("disabled_reason", "disabled by frozen config")
        )
        for flaw_type, detector in config["detectors"].items()
        if flaw_type not in enabled
    }
    return {
        "mode": mode,
        "scores": score["scores"],
        "total_score": score["total"],
        "dimension_reliability": score["dimension_reliability"],
        "score_details": {
            "weights": score["weights"],
            "penalties": score["penalties"],
            "detector_penalties": score["detector_penalties"],
            "features": score["features"],
        },
        "enabled_detector_types": enabled,
        "disabled_detector_types": disabled,
        "regions": explained_regions,
        "series": _series_pair(
            participant_bundle,
            ideal_bundle,
            timings,
            ideal_timings,
            transcript,
            reference,
        ),
    }
