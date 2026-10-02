"""Shared dev-only manifest loading and feature-delta calculations."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from speechlens.alignment.align import SAMPLE_RATE_HZ
from speechlens.features.extract import FeatureBundle, _config, _energy_syllable_peaks, extract_recording_features
from speechlens.schema import WordTiming


EXPECTED_METRICS = {
    "pace_fast": ("articulation_rate_sps", "increase"),
    "pace_slow": ("articulation_rate_sps", "decrease"),
    "long_pause": ("boundary_pause_s", "increase"),
    "monotone": ("f0_std_semitones", "decrease"),
    "volume_dropoff": ("intensity_mean_db", "decrease"),
    "filler": ("speech_activity_duration_s", "increase"),
    "stumble_repeat": ("repeated_word_occurrences", "increase"),
}
DELTA_METRICS = (
    "articulation_rate_sps",
    "words_per_second",
    "f0_std_semitones",
    "f0_range_semitones",
    "intensity_mean_db",
    "intensity_range_db",
    "voiced_fraction",
    "speech_activity_fraction",
    "speech_activity_duration_s",
    "mean_spectral_flux",
    "median_hnr_db",
    "interval_duration_s",
    "boundary_pause_s",
    "repeated_word_occurrences",
)


def load_dev_manifest_rows(project_root: Path) -> list[dict[str, str]]:
    """Read manifest rows but expose only dev rows for downstream file access."""
    manifest_path = project_root / "data" / "labels" / "manifest.csv"
    with manifest_path.open(encoding="utf-8", newline="") as manifest_file:
        return [
            row
            for row in csv.DictReader(manifest_file)
            if row.get("split") == "dev"
        ]


def load_dev_recording(
    project_root: Path,
    row: dict[str, str],
) -> tuple[np.ndarray, list[WordTiming], str, dict[str, Any]]:
    """Load one already-filtered dev recording and its transcript/timings."""
    if row.get("split") != "dev":
        raise ValueError("feature report attempted to open a non-dev recording")
    recording_id = row["recording_id"]
    sidecar_path = project_root / "data" / "labels" / f"{recording_id}.json"
    passage_path = project_root / "data" / "labels" / "passages" / f"{row['passage_id']}.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    passage = json.loads(passage_path.read_text(encoding="utf-8"))
    if sidecar.get("split") != "dev" or passage.get("split") != "dev":
        raise ValueError(f"{recording_id}: sidecar is not explicitly marked dev")
    audio_path = project_root / Path(row["path"])
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    if sample_rate != SAMPLE_RATE_HZ or audio.ndim != 1:
        raise ValueError(f"{recording_id}: expected mono {SAMPLE_RATE_HZ} Hz audio")
    timings_data = sidecar.get("original_word_timings") or sidecar["word_timings"]
    timings = [WordTiming.model_validate(value) for value in timings_data]
    return np.ascontiguousarray(audio, dtype=np.float32), timings, passage["transcript"], sidecar


def analyze_dev_recording(
    project_root: Path,
    row: dict[str, str],
) -> FeatureBundle:
    """Extract features for one dev row only."""
    audio, timings, transcript, _ = load_dev_recording(project_root, row)
    return extract_recording_features(audio, timings, transcript)


def _interval_mask(bundle: FeatureBundle, start_s: float, end_s: float) -> np.ndarray:
    """Select frame centers within a half-open time interval."""
    times = bundle.frames["time_s"].to_numpy(dtype=np.float64)
    return (times >= start_s) & (times < end_s)


def _interval_metrics(
    bundle: FeatureBundle,
    start_s: float,
    end_s: float,
    word_count: int,
) -> dict[str, float]:
    """Summarize the acoustic metrics used in paired dev flaw comparisons."""
    mask = _interval_mask(bundle, start_s, end_s)
    frames = bundle.frames.loc[mask]
    if frames.empty:
        return {
            "articulation_rate_sps": 0.0,
            "words_per_second": 0.0,
            "f0_std_semitones": 0.0,
            "f0_range_semitones": 0.0,
            "intensity_mean_db": 0.0,
            "intensity_range_db": 0.0,
            "voiced_fraction": 0.0,
            "speech_activity_fraction": 0.0,
            "speech_activity_duration_s": 0.0,
            "speech_duration_s": 0.0,
            "syllable_nuclei": 0.0,
            "mean_spectral_flux": 0.0,
            "median_hnr_db": 0.0,
            "interval_duration_s": max(end_s - start_s, 0.0),
            "boundary_pause_s": max(end_s - start_s, 0.0),
        }
    active = frames["speech_activity"].to_numpy(dtype=bool)
    voiced = frames["voiced"].to_numpy(dtype=bool)
    speech_duration = max(float(np.count_nonzero(active)) * float(_config()["hop_s"]), 0.01)
    f0 = frames.loc[voiced, "f0_semitones"].to_numpy(dtype=np.float64)
    intensity = frames.loc[active, "intensity_db"].to_numpy(dtype=np.float64)
    flux = frames["spectral_flux"].to_numpy(dtype=np.float64)
    hnr = frames["hnr_db"].dropna().to_numpy(dtype=np.float64)
    nuclei = _energy_syllable_peaks(
        frames["intensity_raw_dbfs"].to_numpy(dtype=np.float64),
        active,
        1.0 / float(_config()["hop_s"]),
    )
    duration = max(end_s - start_s, 0.01)
    return {
        "articulation_rate_sps": nuclei / speech_duration,
        "words_per_second": word_count / duration,
        "f0_std_semitones": float(np.std(f0)) if f0.size else 0.0,
        "f0_range_semitones": float(np.ptp(f0)) if f0.size else 0.0,
        "intensity_mean_db": float(np.mean(intensity)) if intensity.size else 0.0,
        "intensity_range_db": float(np.ptp(intensity)) if intensity.size else 0.0,
        "voiced_fraction": float(np.count_nonzero(voiced) / max(1, np.count_nonzero(active))),
        "speech_activity_fraction": float(np.count_nonzero(active) / max(1, len(frames))),
        "speech_activity_duration_s": float(np.count_nonzero(active)) * float(_config()["hop_s"]),
        "speech_duration_s": speech_duration,
        "syllable_nuclei": float(nuclei),
        "mean_spectral_flux": float(np.mean(flux)) if flux.size else 0.0,
        "median_hnr_db": float(np.median(hnr)) if hnr.size else 0.0,
        "interval_duration_s": duration,
        "boundary_pause_s": duration,
    }


def _source_interval(
    timings: list[WordTiming],
    word_indices: list[int],
) -> tuple[float, float]:
    """Return the ideal interval covered by source word indices."""
    return timings[min(word_indices)].start_s, timings[max(word_indices)].end_s


def _pause_gap_s(timings: list[WordTiming], word_indices: list[int]) -> float:
    """Return the natural gap between the two source words around a pause."""
    first = min(word_indices)
    second = max(word_indices)
    if second <= first:
        second = first + 1
    return max(0.0, timings[second].start_s - timings[first].end_s)


def _flaw_delta_row(
    flaw: dict,
    ideal_bundle: FeatureBundle,
    flawed_bundle: FeatureBundle,
    ideal_timings: list[WordTiming],
    flawed_timings: list[WordTiming],
    flaw_sidecar: dict,
) -> dict[str, object]:
    """Compute the expected feature delta for one explicitly labeled flaw."""
    flaw_type = flaw["flaw_type"]
    indices = [int(index) for index in flaw["word_indices"]]
    expected_feature, expected_direction = EXPECTED_METRICS[flaw_type]
    original_start = float(flaw["original_start_s"])
    original_end = float(flaw["original_end_s"])
    rendered_start = float(flaw["rendered_start_s"])
    rendered_end = float(flaw["rendered_end_s"])

    source_start, source_end = _source_interval(ideal_timings, indices)
    baseline = _interval_metrics(ideal_bundle, source_start, source_end, len(indices))
    injected_word_count = len(indices)
    if flaw_type == "stumble_repeat":
        injected_word_count = sum(
            float(item["start_s"]) >= rendered_start
            and float(item["end_s"]) <= rendered_end
            for item in flaw_sidecar["word_timings"]
        )
    injected = _interval_metrics(
        flawed_bundle,
        rendered_start,
        rendered_end,
        injected_word_count,
    )
    baseline_value = baseline.get(expected_feature, 0.0)
    injected_value = injected.get(expected_feature, 0.0)
    if flaw_type in {"pace_fast", "pace_slow"}:
        paired_nuclei = baseline["syllable_nuclei"]
        baseline_value = paired_nuclei / max(baseline["speech_duration_s"], 0.01)
        injected_value = paired_nuclei / max(injected["speech_duration_s"], 0.01)
    if flaw_type == "long_pause":
        baseline_value = _pause_gap_s(ideal_timings, indices)
        injected_value = rendered_end - rendered_start
    elif flaw_type == "filler":
        baseline_gap_start = ideal_timings[min(indices)].end_s
        following_index = min(indices) + 1
        baseline_gap_end = ideal_timings[following_index].start_s if following_index < len(ideal_timings) else baseline_gap_start
        baseline_gap_metrics = _interval_metrics(
            ideal_bundle,
            baseline_gap_start,
            baseline_gap_end,
            0,
        )
        baseline_value = baseline_gap_metrics["speech_activity_duration_s"]
        injected_value = injected["speech_activity_duration_s"]
    elif flaw_type == "stumble_repeat":
        baseline_value = 0.0
        injected_value = float(injected_word_count)

    delta = float(injected_value - baseline_value)
    moved_as_expected = (
        delta > 0 if expected_direction == "increase" else delta < 0
    )
    row: dict[str, object] = {
        "recording_id": flaw_sidecar["recording"]["id"],
        "passage_id": flaw_sidecar["passage_id"],
        "split": flaw_sidecar["split"],
        "severity_level": flaw_sidecar["recording"]["severity_level"],
        "flaw_type": flaw_type,
        "severity": flaw["severity"],
        "feature": expected_feature,
        "baseline_value": baseline_value,
        "injected_value": injected_value,
        "delta": delta,
        "direction": "increase" if delta > 0 else "decrease" if delta < 0 else "unchanged",
        "expected_direction": expected_direction,
        "expected_moved": bool(moved_as_expected),
        "original_start_s": original_start,
        "original_end_s": original_end,
        "rendered_start_s": rendered_start,
        "rendered_end_s": rendered_end,
    }
    row[f"baseline_{expected_feature}"] = baseline_value
    row[f"injected_{expected_feature}"] = injected_value
    row[f"delta_{expected_feature}"] = delta
    for metric in DELTA_METRICS:
        if metric == expected_feature:
            continue
        if metric == "repeated_word_occurrences":
            baseline[metric] = 0.0
            injected[metric] = float(injected_word_count) if flaw_type == "stumble_repeat" else 0.0
        row[f"baseline_{metric}"] = baseline[metric]
        row[f"injected_{metric}"] = injected[metric]
        row[f"delta_{metric}"] = injected[metric] - baseline[metric]
    return row


def _bundle_summary(bundle: FeatureBundle) -> dict[str, float]:
    """Return whole-recording metrics used to estimate control false-positive drift."""
    frames = bundle.frames
    voiced = frames[frames["voiced"]]
    speech = frames[frames["speech_activity"]]
    return {
        "median_f0_hz": float(voiced["f0_hz"].median()) if not voiced.empty else 0.0,
        "f0_std_semitones": float(voiced["f0_semitones"].std(ddof=0)) if not voiced.empty else 0.0,
        "median_intensity_raw_dbfs": float(speech["intensity_raw_dbfs"].median()) if not speech.empty else 0.0,
        "median_intensity_db": float(speech["intensity_db"].median()) if not speech.empty else 0.0,
        "voiced_fraction": float(frames["voiced"].sum() / max(1, frames["speech_activity"].sum())),
        "mean_spectral_flux": float(frames["spectral_flux"].mean()),
        "median_hnr_db": float(frames["hnr_db"].median()) if frames["hnr_db"].notna().any() else 0.0,
        "energy_modulation_power_3_6_hz": bundle.modulation["energy_power_3_6_hz"],
    }


def passage_runtime_timer() -> float:
    """Return a monotonic timer value for per-passage reporting."""
    return time.perf_counter()