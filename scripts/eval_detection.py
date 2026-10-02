"""Tune and evaluate detection on DEV recordings only."""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from speechlens.alignment.align import align
from speechlens.detection.core import load_detection_config
from speechlens.detection.asr import cached_inserted_word_spans
from speechlens.detection.measurements import (
    DETECTOR_TYPES,
    build_typed_deviation_table,
    detect_typed_regions,
    fit_reference_free,
    normalize_token,
)
from speechlens.explain import explanation_record
from speechlens.features import FeatureBundle, extract_recording_features
from speechlens.schema import WordTiming


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
REFERENCE_PATH = RESULTS_DIR / "reference_free_ideal_stats.json"
IOU_THRESHOLDS = (0.3, 0.5)
CONTROL_GROUPS = ("ideal", "identity", "mp3", "gain", "noise")
TIMING_SOURCES = ("oracle", "realistic")
MODES = ("paired", "reference_free")


def _transcript_with_wildcards(transcript: str, words_per_slot: int) -> str:
    """Add wildcard slots between transcript groups for untranscribed speech."""
    if words_per_slot < 1:
        raise ValueError("words_per_slot must be positive")
    words = transcript.split()
    groups = [words[index : index + words_per_slot] for index in range(0, len(words), words_per_slot)]
    return " * ".join(" ".join(group) for group in groups)


def _cached_forced_alignment(
    audio: np.ndarray,
    transcript: str,
    cache_dir: Path,
    words_per_slot: int,
) -> list[WordTiming]:
    """Forced-align actual audio and transcript; cache the exact output."""
    alignable_text = _transcript_with_wildcards(transcript, words_per_slot)
    waveform = np.ascontiguousarray(audio, dtype=np.float32)
    digest = hashlib.sha256(waveform.tobytes() + alignable_text.encode("utf-8")).hexdigest()
    path = cache_dir / f"{digest}.json"
    if path.is_file():
        values = json.loads(path.read_text(encoding="utf-8"))
        return [WordTiming.model_validate(item) for item in values]
    timings = align(waveform, alignable_text)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [item.model_dump(mode="json") for item in timings],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return timings


def _cached_plain_alignment(
    audio: np.ndarray,
    transcript: str,
    cache_dir: Path,
) -> list[WordTiming]:
    """Forced-align the unmodified transcript and cache the exact word timings."""
    waveform = np.ascontiguousarray(audio, dtype=np.float32)
    digest = hashlib.sha256(b"plain\0" + waveform.tobytes() + transcript.encode("utf-8")).hexdigest()
    path = cache_dir / f"plain_{digest}.json"
    if path.is_file():
        values = json.loads(path.read_text(encoding="utf-8"))
        return [WordTiming.model_validate(item) for item in values]
    timings = align(waveform, transcript)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [item.model_dump(mode="json") for item in timings],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n",
        encoding="utf-8",
    )
    return timings


def _cached_recording_features(
    audio: np.ndarray,
    timings: list[WordTiming],
    transcript: str,
    cache_dir: Path,
) -> FeatureBundle:
    """Cache acoustic feature bundles keyed by the exact waveform and timing payload."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    waveform = np.ascontiguousarray(audio, dtype=np.float32)
    payload = {
        "audio_sha256": hashlib.sha256(waveform.tobytes()).hexdigest(),
        "transcript": transcript,
        "timings": [item.model_dump(mode="json") for item in timings],
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    path = cache_dir / f"{digest}.pkl"
    if path.is_file():
        with path.open("rb") as cache_file:
            return pickle.load(cache_file)
    bundle = extract_recording_features(audio, timings, transcript)
    with path.open("wb") as cache_file:
        pickle.dump(bundle, cache_file, protocol=pickle.HIGHEST_PROTOCOL)
    return bundle


def _load_dev_rows(project_root: Path) -> list[dict[str, str]]:
    """Filter the manifest to DEV before opening any recording sidecar/audio."""
    path = project_root / "data" / "labels" / "manifest.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        return [row for row in csv.DictReader(stream) if row.get("split") == "dev"]


def _preserve_legacy_baseline() -> None:
    """Keep the prior oracle detector summary for a direct before/after report."""
    target = RESULTS_DIR / "detection_legacy_before.csv"
    source = RESULTS_DIR / "detection_summary.csv"
    if target.is_file() or not source.is_file():
        return
    with source.open(newline="", encoding="utf-8") as stream:
        previous_rows = list(csv.DictReader(stream))
    columns = (
        "mode", "precision_iou_0.3", "recall_iou_0.3", "f1_iou_0.3",
        "precision_iou_0.5", "recall_iou_0.5", "f1_iou_0.5",
        "mean_boundary_error_ms", "false_regions_per_minute_all_controls_and_ideals",
    )
    preserved = [
        {"baseline": "previous sidecar-timing detector", **{
            column: row.get(column, "") for column in columns
        }}
        for row in previous_rows
    ]
    if preserved:
        pd.DataFrame(preserved).to_csv(target, index=False, lineterminator="\n")


def _load_dev_entries(project_root: Path, config: dict) -> list[dict]:
    """Load DEV data; sidecar timings are retained solely for oracle metrics."""
    entries: list[dict] = []
    transcript_cache: dict[str, str] = {}
    cache_dir = project_root / ".speechlens_cache" / "detection_alignments"
    dev_rows = _load_dev_rows(project_root)
    for recording_index, row in enumerate(dev_rows, start=1):
        recording_id = row["recording_id"]
        print(
            f"Aligning DEV recording {recording_index}/{len(dev_rows)}: {recording_id}",
            flush=True,
        )
        sidecar_path = project_root / "data" / "labels" / f"{recording_id}.json"
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar.get("split") != "dev":
            raise ValueError(f"DEV manifest row has non-DEV sidecar: {recording_id}")
        passage_id = row["passage_id"]
        if passage_id not in transcript_cache:
            passage_path = project_root / "data" / "labels" / "passages" / f"{passage_id}.json"
            passage = json.loads(passage_path.read_text(encoding="utf-8"))
            if passage.get("split") != "dev":
                raise ValueError(f"DEV row references non-DEV transcript: {passage_id}")
            transcript_cache[passage_id] = str(passage["transcript"])
        transcript = transcript_cache[passage_id]
        audio, sample_rate_hz = sf.read(project_root / row["path"], dtype="float32", always_2d=False)
        if audio.ndim != 1 or sample_rate_hz != 16000:
            raise ValueError(f"expected mono 16 kHz DEV audio: {recording_id}")

        oracle_timings = [WordTiming.model_validate(item) for item in sidecar["word_timings"]]
        realistic_timings = _cached_plain_alignment(
            audio,
            transcript,
            cache_dir,
        )
        wildcard_timings = _cached_forced_alignment(
            audio, transcript, cache_dir, int(config["wildcard_words_per_slot"])
        )
        stumble_intervals = (
            cached_inserted_word_spans(
                audio,
                transcript,
                project_root / ".speechlens_cache" / "detection_asr",
                project_root / ".speechlens_cache" / "asr_torch_hub",
                float(config["asr"]["chunk_duration_s"]),
                float(config["asr"]["overlap_s"]),
            )
            if row["kind"] != "ideal"
            else []
        )
        timing_bundles: dict[str, dict] = {}
        for source, timings in (
            ("oracle", oracle_timings),
            ("realistic", realistic_timings),
        ):
            feature_cache_dir = project_root / ".speechlens_cache" / "detection_feature_bundles"
            timing_bundles[source] = {
                "timings": timings,
                "bundle": _cached_recording_features(
                    audio,
                    timings,
                    transcript,
                    feature_cache_dir,
                ),
                "wildcard_spans": [
                    (float(timing.start_s), float(timing.end_s))
                    for timing in timings
                    if timing.word.strip() == "*"
                ],
            }
        pair = sidecar.get("pair")
        entries.append({
            "manifest": row,
            "sidecar": sidecar,
            "transcript": transcript,
            "timing_bundles": timing_bundles,
            "wildcard_timings": wildcard_timings,
            "stumble_intervals": stumble_intervals,
            "ideal_recording_id": (
                pair["ideal_recording_id"] if pair else
                sidecar["recording"].get("parent_recording_id") or recording_id
            ),
            "ground_truth": [
                {
                    "start_s": float(flaw["rendered_start_s"]),
                    "end_s": float(flaw["rendered_end_s"]),
                    "type": str(flaw["flaw_type"]),
                    "severity": float(flaw["severity"]),
                }
                for flaw in (pair["flaws"] if pair else [])
            ],
        })
    return entries


def _interval_iou(predicted: Mapping, actual: Mapping) -> float:
    intersection = max(
        0.0,
        min(float(predicted["end_s"]), float(actual["end_s"]))
        - max(float(predicted["start_s"]), float(actual["start_s"])),
    )
    union = max(float(predicted["end_s"]), float(actual["end_s"])) - min(
        float(predicted["start_s"]), float(actual["start_s"])
    )
    return intersection / union if union > 0 else 0.0


def _match_regions(
    predicted: list[dict], actual: list[dict], threshold: float
) -> list[tuple[int, int, float]]:
    candidates = [
        (pi, ai, _interval_iou(prediction, label))
        for pi, prediction in enumerate(predicted)
        for ai, label in enumerate(actual)
    ]
    candidates.sort(key=lambda item: (-item[2], item[0], item[1]))
    used_predicted: set[int] = set()
    used_actual: set[int] = set()
    matches = []
    for pi, ai, iou in candidates:
        if iou < threshold or pi in used_predicted or ai in used_actual:
            continue
        used_predicted.add(pi)
        used_actual.add(ai)
        matches.append((pi, ai, iou))
    return matches


def _control_group(entry: dict) -> str | None:
    row = entry["manifest"]
    if row["kind"] == "ideal":
        return "ideal"
    if row["kind"] != "control":
        return None
    recording_id = row["recording_id"].lower()
    if "identity" in recording_id:
        return "identity"
    if "lossy" in recording_id or "mp3" in recording_id:
        return "mp3"
    if "gain" in recording_id:
        return "gain"
    if "noise" in recording_id:
        return "noise"
    return None


def _fit_references_leave_one_passage_out(
    entries: list[dict], source: str, window_sizes: list[int]
) -> dict[str, dict]:
    """Fit one DEV-IDEAL reference per held-out passage, excluding that passage."""
    ideal_entries = [entry for entry in entries if entry["manifest"]["kind"] == "ideal"]
    passage_ids = sorted({str(entry["manifest"]["passage_id"]) for entry in ideal_entries})
    if len(passage_ids) < 2:
        raise ValueError("leave-one-passage-out references require at least two DEV IDEAL passages")
    references: dict[str, dict] = {}
    for held_out in passage_ids:
        fit_entries = [
            entry for entry in ideal_entries
            if str(entry["manifest"]["passage_id"]) != held_out
        ]
        reference = fit_reference_free(
            [
                {
                    "bundle": entry["timing_bundles"][source]["bundle"],
                    "timings": entry["timing_bundles"][source]["timings"],
                    "transcript": entry["transcript"],
                }
                for entry in fit_entries
            ],
            window_sizes,
        )
        reference["fit_passage_ids"] = sorted(
            str(entry["manifest"]["passage_id"]) for entry in fit_entries
        )
        reference["held_out_passage_id"] = held_out
        references[held_out] = reference
    return references


def _alignment_boundary_differences(entries: list[dict]) -> list[dict]:
    """Compare matched plain and wildcard word boundaries for DEV IDEALs."""
    differences: list[dict] = []
    for entry in entries:
        if entry["manifest"]["kind"] != "ideal":
            continue
        plain = entry["timing_bundles"]["realistic"]["timings"]
        wildcard = entry["wildcard_timings"]
        plain_lexical = [item for item in plain if item.word.strip() != "*"]
        wildcard_lexical = [item for item in wildcard if item.word.strip() != "*"]
        matcher = SequenceMatcher(
            a=[normalize_token(item.word) for item in plain_lexical],
            b=[normalize_token(item.word) for item in wildcard_lexical],
            autojunk=False,
        )
        for plain_start, wildcard_start, size in matcher.get_matching_blocks():
            for offset in range(size):
                plain_word = plain_lexical[plain_start + offset]
                wildcard_word = wildcard_lexical[wildcard_start + offset]
                differences.append({
                    "recording_id": entry["manifest"]["recording_id"],
                    "word": plain_word.word,
                    "start_boundary_abs_difference_ms": abs(
                        plain_word.start_s - wildcard_word.start_s
                    ) * 1000.0,
                    "end_boundary_abs_difference_ms": abs(
                        plain_word.end_s - wildcard_word.end_s
                    ) * 1000.0,
                    "duration_abs_difference_ms": abs(
                        (plain_word.end_s - plain_word.start_s)
                        - (wildcard_word.end_s - wildcard_word.start_s)
                    ) * 1000.0,
                })
    return differences


def _ideal_score_distributions(entries: list[dict], tables: dict) -> list[dict]:
    """Summarize per-type frame scores pooled over DEV IDEAL recordings."""
    ideal_ids = [
        entry["manifest"]["recording_id"]
        for entry in entries
        if entry["manifest"]["kind"] == "ideal"
    ]
    rows: list[dict] = []
    for source in TIMING_SOURCES:
        for mode in MODES:
            for flaw_type in DETECTOR_TYPES:
                values = np.concatenate([
                    tables[source][mode][recording_id][f"score_{flaw_type}"].to_numpy(dtype=np.float64)
                    for recording_id in ideal_ids
                ])
                rows.append({
                    "timing_source": source,
                    "mode": mode,
                    "flaw_type": flaw_type,
                    "frames": int(values.size),
                    "p50": float(np.percentile(values, 50)),
                    "p90": float(np.percentile(values, 90)),
                    "p99": float(np.percentile(values, 99)),
                    "max": float(np.max(values)) if values.size else 0.0,
                })
    return rows


def _run_pre_tuning_diagnostics(
    entries: list[dict], tables: dict, config: dict
) -> list[dict]:
    """Write alignment/score diagnostics and enforce zero paired IDEAL regions."""
    boundary_rows = _alignment_boundary_differences(entries)
    boundary_values = [
        value
        for row in boundary_rows
        for value in (
            row["start_boundary_abs_difference_ms"],
            row["end_boundary_abs_difference_ms"],
        )
    ]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(RESULTS_DIR / "alignment_plain_vs_wildcard_dev_ideal.csv", boundary_rows)
    if boundary_values:
        print(
            "Plain/wildcard DEV IDEAL median absolute boundary difference: "
            f"{float(np.median(boundary_values)):.2f} ms",
            flush=True,
        )
    score_rows = _ideal_score_distributions(entries, tables)
    _write_csv(RESULTS_DIR / "detection_ideal_score_distributions.csv", score_rows)
    for row in score_rows:
        print(
            f"IDEAL scores {row['timing_source']}/{row['mode']}/{row['flaw_type']}: "
            f"p50={row['p50']:.6g} p90={row['p90']:.6g} "
            f"p99={row['p99']:.6g} max={row['max']:.6g}",
            flush=True,
        )

    diagnostic_rows: list[dict] = []
    paired, _, _, paired_controls, paired_predictions = _evaluate_variant(
        "paired", "realistic", entries, tables["realistic"]["paired"],
        config, False,
    )
    ideal_false_regions = {
        entry["manifest"]["recording_id"]: paired_predictions[entry["manifest"]["recording_id"]]
        for entry in entries
        if entry["manifest"]["kind"] == "ideal"
        and paired_predictions[entry["manifest"]["recording_id"]]
    }
    if ideal_false_regions:
        raise RuntimeError(
            "paired IDEAL self-comparison produced regions before tuning: "
            + ", ".join(sorted(ideal_false_regions))
        )
    diagnostic_rows.extend(paired_controls)

    reference_free, _, _, reference_controls, _ = _evaluate_variant(
        "reference_free", "realistic", entries, tables["realistic"]["reference_free"],
        config, False,
    )
    diagnostic_rows.extend(reference_controls)
    _write_csv(RESULTS_DIR / "detection_pre_tuning_control_rates.csv", diagnostic_rows)
    print(
        "Pre-tuning DEV IDEAL/control false regions per minute: "
        f"paired={paired['false_regions_per_minute_all_controls_and_ideals']:.3f}, "
        f"reference_free={reference_free['false_regions_per_minute_all_controls_and_ideals']:.3f}",
        flush=True,
    )
    return diagnostic_rows


def _prepare_tables(entries: list[dict], config: dict, references: dict[str, dict]) -> dict:
    by_id = {entry["manifest"]["recording_id"]: entry for entry in entries}
    result = {source: {mode: {} for mode in MODES} for source in TIMING_SOURCES}
    for entry_index, entry in enumerate(entries, 1):
        row = entry["manifest"]
        recording_id = row["recording_id"]
        ideal = by_id[entry["ideal_recording_id"]]
        for source in TIMING_SOURCES:
            participant = entry["timing_bundles"][source]
            ideal_data = ideal["timing_bundles"][source]
            reference = references[source][str(row["passage_id"])]
            cache_root = PROJECT_ROOT / ".speechlens_cache" / "detection_tables"
            print(f"Preparing {source} tables {entry_index}/{len(entries)}: {recording_id}", flush=True)
            shared_key = {
                "cache_version": "plain-alignment-typed-frame-table-v2",
                "recording_id": recording_id,
                "ideal_recording_id": entry["ideal_recording_id"],
                "timing_source": source,
                "transcript": entry["transcript"],
                "participant_timings": [timing.model_dump(mode="json") for timing in participant["timings"]],
                "ideal_timings": [timing.model_dump(mode="json") for timing in ideal_data["timings"]],
                "measurement_config": config["measurements"],
                "wildcard_spans": participant["wildcard_spans"],
                "stumble_intervals": entry["stumble_intervals"],
            }
            for mode in MODES:
                cache_material = {
                    **shared_key,
                    "mode": mode,
                    "reference": reference if mode == "reference_free" else None,
                }
                cache_hash = hashlib.sha256(
                    json.dumps(cache_material, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                cache_path = cache_root / f"{cache_hash}.pkl"
                if cache_path.is_file():
                    with cache_path.open("rb") as cache_file:
                        result[source][mode][recording_id] = pickle.load(cache_file)
                    continue
                if mode == "paired":
                    table = build_typed_deviation_table(
                        participant["bundle"], participant["timings"], entry["transcript"], config,
                        ideal_bundle=ideal_data["bundle"], ideal_timings=ideal_data["timings"],
                        stumble_intervals=entry["stumble_intervals"],
                    )
                else:
                    table = build_typed_deviation_table(
                        participant["bundle"], participant["timings"], entry["transcript"], config,
                        reference=reference,
                        stumble_intervals=entry["stumble_intervals"],
                    )
                cache_root.mkdir(parents=True, exist_ok=True)
                with cache_path.open("wb") as cache_file:
                    pickle.dump(table, cache_file, protocol=pickle.HIGHEST_PROTOCOL)
                result[source][mode][recording_id] = table
    return result


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _bootstrap_f1(
    entries: list[dict], predictions: dict[str, list[dict]], count: int, seed: int
) -> tuple[float, float]:
    by_passage: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        by_passage[entry["manifest"]["passage_id"]].append(entry)
    passage_ids = sorted(by_passage)
    rng = np.random.default_rng(seed)
    f1_samples = []
    for _ in range(count):
        selected = rng.choice(passage_ids, size=len(passage_ids), replace=True)
        tp = fp = fn = 0
        for passage_id in selected:
            for entry in by_passage[str(passage_id)]:
                prediction = predictions[entry["manifest"]["recording_id"]]
                actual = entry["ground_truth"]
                matches = _match_regions(prediction, actual, 0.3)
                tp += len(matches)
                fp += len(prediction) - len({item[0] for item in matches})
                fn += len(actual) - len({item[1] for item in matches})
        f1_samples.append(_prf(tp, fp, fn)[2])
    low, high = np.percentile(f1_samples, [2.5, 97.5])
    return float(low), float(high)


def _evaluate_variant(
    mode: str,
    source: str,
    entries: list[dict],
    tables: dict[str, pd.DataFrame],
    config: dict,
    include_bootstrap: bool = True,
) -> tuple[dict, list[dict], list[dict], list[dict], dict[str, list[dict]]]:
    predictions: dict[str, list[dict]] = {}
    controls: Counter = Counter()
    minutes: Counter = Counter()
    type_counts: dict[str, Counter] = defaultdict(Counter)
    severity_counts: dict[str, Counter] = defaultdict(Counter)
    confusion: Counter = Counter()
    boundary_errors = []
    for entry in entries:
        row = entry["manifest"]
        recording_id = row["recording_id"]
        timings = entry["timing_bundles"][source]["timings"]
        predicted = detect_typed_regions(tables[recording_id], timings, config)
        predictions[recording_id] = predicted
        actual = entry["ground_truth"]
        group = _control_group(entry)
        if not actual and group:
            controls[group] += len(predicted)
            minutes[group] += float(row["duration_s"]) / 60.0
        for flaw_type in DETECTOR_TYPES:
            pred_type = [region for region in predicted if region["type"] == flaw_type]
            actual_type = [region for region in actual if region["type"] == flaw_type]
            matches = _match_regions(pred_type, actual_type, 0.3)
            type_counts[flaw_type]["tp"] += len(matches)
            type_counts[flaw_type]["fp"] += len(pred_type) - len({item[0] for item in matches})
            type_counts[flaw_type]["fn"] += len(actual_type) - len({item[1] for item in matches})
        matches = _match_regions(predicted, actual, 0.3)
        for pi, ai, _ in matches:
            confusion[(actual[ai]["type"], predicted[pi]["type"])] += 1
            boundary_errors.extend([
                abs(predicted[pi]["start_s"] - actual[ai]["start_s"]) * 1000,
                abs(predicted[pi]["end_s"] - actual[ai]["end_s"]) * 1000,
            ])
        if row["kind"] == "injected":
            level = f"L{int(row['severity_level'])}"
            severity_matches = _match_regions(predicted, actual, 0.3)
            severity_counts[level]["actual"] += len(actual)
            severity_counts[level]["matched"] += len({item[1] for item in severity_matches})

    summary = {"mode": mode, "timing_source": source, "confusion": confusion}
    for threshold in IOU_THRESHOLDS:
        tp = fp = fn = 0
        for entry in entries:
            predicted = predictions[entry["manifest"]["recording_id"]]
            actual = entry["ground_truth"]
            matches = _match_regions(predicted, actual, threshold)
            tp += len(matches)
            fp += len(predicted) - len({item[0] for item in matches})
            fn += len(actual) - len({item[1] for item in matches})
        precision, recall, f1 = _prf(tp, fp, fn)
        summary[f"precision_iou_{threshold:.1f}"] = precision
        summary[f"recall_iou_{threshold:.1f}"] = recall
        summary[f"f1_iou_{threshold:.1f}"] = f1
    summary["mean_boundary_error_ms"] = float(np.mean(boundary_errors)) if boundary_errors else 0.0
    summary["false_regions_per_minute_all_controls_and_ideals"] = sum(controls.values()) / max(sum(minutes.values()), 1e-12)
    types = []
    for flaw_type in DETECTOR_TYPES:
        for threshold in IOU_THRESHOLDS:
            typed_tp = typed_fp = typed_fn = 0
            for entry in entries:
                recording_id = entry["manifest"]["recording_id"]
                predicted = [
                    region for region in predictions[recording_id]
                    if region["type"] == flaw_type
                ]
                actual = [
                    region for region in entry["ground_truth"]
                    if region["type"] == flaw_type
                ]
                matches = _match_regions(predicted, actual, threshold)
                typed_tp += len(matches)
                typed_fp += len(predicted) - len({match[0] for match in matches})
                typed_fn += len(actual) - len({match[1] for match in matches})
            precision, recall, f1 = _prf(typed_tp, typed_fp, typed_fn)
            types.append({
                "mode": mode, "timing_source": source, "flaw_type": flaw_type,
                "iou_threshold": threshold,
                "tp": typed_tp, "fp": typed_fp, "fn": typed_fn,
                "precision": precision, "recall": recall, "f1": f1,
            })
    severity = [
        {
            "mode": mode, "timing_source": source, "severity_level": f"L{level}",
            "actual": severity_counts[f"L{level}"]["actual"],
            "matched": severity_counts[f"L{level}"]["matched"],
            "recall": severity_counts[f"L{level}"]["matched"] / max(1, severity_counts[f"L{level}"]["actual"]),
        }
        for level in range(1, 6)
    ]
    control_rows = [
        {
            "mode": mode, "timing_source": source, "control_type": group,
            "false_regions": controls[group], "minutes": minutes[group],
            "false_regions_per_minute": controls[group] / max(minutes[group], 1e-12),
        }
        for group in CONTROL_GROUPS
    ]
    if include_bootstrap:
        low, high = _bootstrap_f1(
            entries,
            predictions,
            int(config["evaluation"]["bootstrap_samples"]),
            int(config["evaluation"]["bootstrap_seed"]),
        )
        summary["f1_iou_0.3_ci95_low"] = low
        summary["f1_iou_0.3_ci95_high"] = high
    return summary, types, severity, control_rows, predictions


def _write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")
    else:
        path.write_text("\n", encoding="utf-8")


def _tune_thresholds(entries: list[dict], tables: dict, config: dict) -> tuple[dict, list[dict]]:
    """Tune only thresholds that meet their share and enforce the combined budget."""
    from copy import deepcopy

    realistic = tables["realistic"]
    budget = float(config["evaluation"]["false_region_budget_per_minute"])
    type_budget = float(config["evaluation"]["false_region_budget_per_type_per_minute"])
    tuning_rows: list[dict] = []
    baseline_thresholds = {
        flaw_type: {
            "enter": float(config["detectors"][flaw_type]["enter_threshold"]),
            "exit": float(config["detectors"][flaw_type]["exit_threshold"]),
        }
        for flaw_type in DETECTOR_TYPES
    }
    for flaw_type in DETECTOR_TYPES:
        config["detectors"][flaw_type]["enabled"] = True
        config["detectors"][flaw_type].pop("disabled_reason", None)

    current_predictions = {
        mode: {
            entry["manifest"]["recording_id"]: detect_typed_regions(
                realistic[mode][entry["manifest"]["recording_id"]],
                entry["timing_bundles"]["realistic"]["timings"],
                config,
            )
            for entry in entries
        }
        for mode in MODES
    }

    def measure_predictions(
        mode_predictions: dict[str, list[dict]], target: str
    ) -> tuple[float, float, float, float]:
        target_tp = target_fp = target_fn = 0
        all_tp = all_fp = all_fn = 0
        false_regions = 0
        target_false_regions = 0
        control_minutes = 0.0
        for entry in entries:
            recording_id = entry["manifest"]["recording_id"]
            predicted = mode_predictions[recording_id]
            actual = entry["ground_truth"]
            matches = _match_regions(predicted, actual, 0.3)
            all_tp += len(matches)
            all_fp += len(predicted) - len({match[0] for match in matches})
            all_fn += len(actual) - len({match[1] for match in matches})
            predicted_target = [item for item in predicted if item["type"] == target]
            actual_target = [item for item in actual if item["type"] == target]
            typed_matches = _match_regions(predicted_target, actual_target, 0.3)
            target_tp += len(typed_matches)
            target_fp += len(predicted_target) - len({match[0] for match in typed_matches})
            target_fn += len(actual_target) - len({match[1] for match in typed_matches})
            if _control_group(entry) is not None:
                false_regions += len(predicted)
                target_false_regions += len(predicted_target)
                control_minutes += float(entry["manifest"]["duration_s"]) / 60.0
        type_f1 = _prf(target_tp, target_fp, target_fn)[2]
        overall_f1 = _prf(all_tp, all_fp, all_fn)[2]
        return (
            type_f1,
            overall_f1,
            false_regions / max(control_minutes, 1e-12),
            target_false_regions / max(control_minutes, 1e-12),
        )

    def combined_control_rate(mode_predictions: dict[str, list[dict]]) -> float:
        false_regions = 0
        control_minutes = 0.0
        for entry in entries:
            if _control_group(entry) is None:
                continue
            recording_id = entry["manifest"]["recording_id"]
            false_regions += len(mode_predictions[recording_id])
            control_minutes += float(entry["manifest"]["duration_s"]) / 60.0
        return false_regions / max(control_minutes, 1e-12)

    baseline_type_f1 = {
        mode: {
            flaw_type: measure_predictions(current_predictions[mode], flaw_type)[0]
            for flaw_type in DETECTOR_TYPES
        }
        for mode in MODES
    }

    for flaw_type in DETECTOR_TYPES:
        before_metrics = {
            mode: measure_predictions(current_predictions[mode], flaw_type)
            for mode in MODES
        }
        before_fp = max(metrics[2] for metrics in before_metrics.values())
        score_column = f"score_{flaw_type}"
        maximum_observed_score = max(
            float(realistic[mode][entry["manifest"]["recording_id"]][score_column].max())
            for mode in MODES
            for entry in entries
        )
        thresholds = sorted({
            float(value) for value in config["tuning_candidates"][flaw_type]
        } | {float(np.nextafter(maximum_observed_score, np.inf))})
        config["tuning_candidates"][flaw_type] = thresholds
        best_score = -1.0
        best_thresholds = None
        best_predictions = None
        for threshold in thresholds:
            print(f"Tuning {flaw_type}: enter_threshold={float(threshold):g}", flush=True)
            candidate = deepcopy(config)
            detector = candidate["detectors"][flaw_type]
            detector["enabled"] = True
            detector["enter_threshold"] = float(threshold)
            detector["exit_threshold"] = min(
                float(detector["exit_threshold"]), float(threshold) * 0.5
            )
            candidate_predictions = {}
            for mode in MODES:
                candidate_predictions[mode] = {}
                for entry in entries:
                    recording_id = entry["manifest"]["recording_id"]
                    timings = entry["timing_bundles"]["realistic"]["timings"]
                    other_regions = [
                        region for region in current_predictions[mode][recording_id]
                        if region["type"] != flaw_type
                    ]
                    own_regions = detect_typed_regions(
                        realistic[mode][recording_id],
                        timings,
                        candidate,
                        flaw_types=(flaw_type,),
                    )
                    candidate_predictions[mode][recording_id] = sorted(
                        other_regions + own_regions,
                        key=lambda region: (region["start_s"], region["type"]),
                    )
            metrics = {
                mode: measure_predictions(candidate_predictions[mode], flaw_type)
                for mode in MODES
            }
            score = float(np.mean([metrics[mode][0] for mode in MODES]))
            worst_type_fp = max(metrics[mode][3] for mode in MODES)
            share_met = worst_type_fp <= type_budget
            tuning_rows.append({
                "stage": "candidate", "flaw_type": flaw_type,
                "enter_threshold": float(threshold),
                "exit_threshold": detector["exit_threshold"],
                "paired_f1_iou_0_3": metrics["paired"][1],
                "reference_free_f1_iou_0_3": metrics["reference_free"][1],
                "mean_target_type_f1": score,
                "worst_mode_combined_fp_per_minute": max(metrics[mode][2] for mode in MODES),
                "worst_mode_type_fp_per_minute": worst_type_fp,
                "type_budget_per_minute": type_budget,
                "combined_budget_per_minute": budget,
                "budget_met": share_met,
            })
            print(
                f"  {flaw_type} type-FP/min paired={metrics['paired'][3]:.3f} "
                f"reference_free={metrics['reference_free'][3]:.3f} share_met={share_met}",
                flush=True,
            )
            if share_met and score > 0.0 and score > best_score:
                best_score = score
                best_thresholds = (detector["enter_threshold"], detector["exit_threshold"])
                best_predictions = candidate_predictions

        if best_thresholds is None:
            enabled = False
            disabled_reason = (
                f"No configured DEV-only threshold met the per-type false-region "
                f"budget of {type_budget:.3f}/minute with positive target F1 in both modes."
            )
            config["detectors"][flaw_type]["enabled"] = False
            config["detectors"][flaw_type]["disabled_reason"] = disabled_reason
            for mode in MODES:
                for recording_id, regions in current_predictions[mode].items():
                    current_predictions[mode][recording_id] = [
                        region for region in regions if region["type"] != flaw_type
                    ]
        else:
            enabled = True
            disabled_reason = ""
            config["detectors"][flaw_type]["enabled"] = True
            config["detectors"][flaw_type].pop("disabled_reason", None)
            config["detectors"][flaw_type]["enter_threshold"] = best_thresholds[0]
            config["detectors"][flaw_type]["exit_threshold"] = best_thresholds[1]
            current_predictions = best_predictions

        after_metrics = {
            mode: measure_predictions(current_predictions[mode], flaw_type)
            for mode in MODES
        }
        tuning_rows.append({
            "stage": "before_after", "flaw_type": flaw_type,
            "enter_threshold_before": baseline_thresholds[flaw_type]["enter"],
            "exit_threshold_before": baseline_thresholds[flaw_type]["exit"],
            "enter_threshold_after": config["detectors"][flaw_type]["enter_threshold"],
            "exit_threshold_after": config["detectors"][flaw_type]["exit_threshold"],
            "paired_f1_before": baseline_type_f1["paired"][flaw_type],
            "reference_free_f1_before": baseline_type_f1["reference_free"][flaw_type],
            "paired_f1_after": after_metrics["paired"][0],
            "reference_free_f1_after": after_metrics["reference_free"][0],
            "type_fp_per_minute_after": max(after_metrics[mode][3] for mode in MODES),
            "type_budget_per_minute": type_budget,
            "enabled": enabled,
            "disabled_reason": disabled_reason,
            "budget_met": (
                not enabled
                or max(after_metrics[mode][3] for mode in MODES) <= type_budget
            ),
        })

    combined_rate = max(combined_control_rate(current_predictions[mode]) for mode in MODES)
    while combined_rate > budget:
        enabled_types = [
            flaw_type for flaw_type in DETECTOR_TYPES
            if config["detectors"][flaw_type].get("enabled", True)
        ]
        if not enabled_types:
            raise RuntimeError(
                f"Combined DEV false-region rate {combined_rate:.3f}/minute exceeds "
                f"the {budget:.3f}/minute budget with all types disabled."
            )
        contributions = {
            flaw_type: max(
                measure_predictions(current_predictions[mode], flaw_type)[3]
                for mode in MODES
            )
            for flaw_type in enabled_types
        }
        disabled_type = max(enabled_types, key=lambda item: (contributions[item], item))
        reason = (
            f"Disabled to enforce the combined DEV false-region budget of "
            f"{budget:.3f}/minute; remaining contribution was "
            f"{contributions[disabled_type]:.3f}/minute."
        )
        config["detectors"][disabled_type]["enabled"] = False
        config["detectors"][disabled_type]["disabled_reason"] = reason
        for mode in MODES:
            for recording_id, regions in current_predictions[mode].items():
                current_predictions[mode][recording_id] = [
                    region for region in regions if region["type"] != disabled_type
                ]
        tuning_rows.append({
            "stage": "combined_budget_disable",
            "flaw_type": disabled_type,
            "enabled": False,
            "disabled_reason": reason,
            "type_fp_per_minute_before_disable": contributions[disabled_type],
            "combined_budget_per_minute": budget,
        })
        combined_rate = max(combined_control_rate(current_predictions[mode]) for mode in MODES)

    if combined_rate > budget:
        raise RuntimeError(
            f"Tuner left combined DEV false-region rate {combined_rate:.3f}/minute "
            f"above the {budget:.3f}/minute budget."
        )
    print(
        f"Final combined DEV false-region rate: {combined_rate:.3f}/minute "
        f"(budget {budget:.3f})",
        flush=True,
    )
    return config, tuning_rows


def _write_reports(
    summaries: list[dict],
    type_rows: list[dict],
    severity_rows: list[dict],
    control_rows: list[dict],
    tuning_rows: list[dict],
    predictions: dict[tuple[str, str], dict[str, list[dict]]],
    config: dict,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(
        RESULTS_DIR / "detection_summary.csv",
        [{key: value for key, value in row.items() if key != "confusion"} for row in summaries],
    )
    _write_csv(RESULTS_DIR / "detection_type_metrics.csv", type_rows)
    _write_csv(RESULTS_DIR / "detection_severity_recall.csv", severity_rows)
    _write_csv(RESULTS_DIR / "detection_control_false_positives.csv", control_rows)
    _write_csv(RESULTS_DIR / "detection_threshold_tuning.csv", tuning_rows)
    for summary in summaries:
        matrix = pd.DataFrame(0, index=DETECTOR_TYPES, columns=DETECTOR_TYPES)
        for (actual, predicted), count in summary["confusion"].items():
            if actual in matrix.index and predicted in matrix.columns:
                matrix.loc[actual, predicted] = count
        matrix.to_csv(
            RESULTS_DIR / f"detection_confusion_{summary['timing_source']}_{summary['mode']}.csv",
            lineterminator="\n",
        )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for source in TIMING_SOURCES:
        for mode in MODES:
            selected = [
                row for row in type_rows
                if row["timing_source"] == source
                and row["mode"] == mode
                and row["iou_threshold"] == 0.3
            ]
            axes[0].plot(
                [row["flaw_type"] for row in selected], [row["recall"] for row in selected],
                marker="o", label=f"{source}/{mode}",
            )
            if source == "realistic":
                severity = [row for row in severity_rows if row["timing_source"] == source and row["mode"] == mode]
                axes[1].plot(
                    [row["severity_level"] for row in severity], [row["recall"] for row in severity],
                    marker="o", label=mode,
                )
            records = []
            for recording_id, regions in sorted(predictions[(source, mode)].items()):
                for region in regions:
                    record = explanation_record(region)
                    record.update({"recording_id": recording_id, "timing_source": source, "mode": mode})
                    records.append(record)
            (RESULTS_DIR / f"detection_explanations_{source}_{mode}.json").write_text(
                json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    axes[0].set_title("DEV per-type recall")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].set_title("Severity recall, realistic timings")
    axes[1].set_ylim(0.0, 1.0)
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend()
    figure.savefig(RESULTS_DIR / "detection_recall.png", dpi=160)
    plt.close(figure)
    headline = [row for row in summaries if row["timing_source"] == "realistic"]
    weak_types = [
        {
            "mode": row["mode"], "flaw_type": row["flaw_type"],
            "iou_threshold": row["iou_threshold"],
            "precision": row["precision"], "recall": row["recall"],
        }
        for row in type_rows
        if row["timing_source"] == "realistic"
        and row["iou_threshold"] == 0.3
        and row["recall"] < 0.5
    ]
    limitations = {
        "headline_timing_source": "realistic",
        "enabled_types": [
            flaw_type for flaw_type in DETECTOR_TYPES
            if config["detectors"][flaw_type].get("enabled", True)
        ],
        "disabled_types": {
            flaw_type: config["detectors"][flaw_type].get("disabled_reason", "")
            for flaw_type in DETECTOR_TYPES
            if not config["detectors"][flaw_type].get("enabled", True)
        },
        "weak_types_from_dev_metrics": weak_types,
        "fillers": "Only stable voiced segments in inter-word gaps of the plain alignment are considered. Acoustic evidence cannot prove lexical content; controls and IDEAL rates are reported for review.",
        "stumbles": "Greedy Wav2Vec2 ASR on CPU compares inserted words against the reference transcript; recognition and forced timing errors can still miss or misplace repeats.",
        "reference_free": "Each DEV passage is scored against an IDEAL reference fitted on all other DEV IDEAL passages. No held-out passage contributes to its own reference.",
        "alignment": "Pace, pause, monotone, and volume measurements use plain forced alignment. Wildcard alignment is diagnostic only.",
        "bootstrap_ci95": {
            row["mode"]: [row["f1_iou_0.3_ci95_low"], row["f1_iou_0.3_ci95_high"]]
            for row in headline
        },
    }
    (RESULTS_DIR / "what_detector_cannot_do.json").write_text(
        json.dumps(limitations, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config_path = PROJECT_ROOT / "config" / "detection.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    (RESULTS_DIR / "frozen_config.sha256").write_text(
        f"{digest}  config/detection.yaml\n", encoding="ascii"
    )


def run_evaluation(project_root: Path = PROJECT_ROOT) -> list[dict]:
    """Run oracle and realistic evaluation for paired and reference-free modes."""
    global PROJECT_ROOT, RESULTS_DIR, REFERENCE_PATH
    PROJECT_ROOT = project_root.resolve()
    RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
    REFERENCE_PATH = RESULTS_DIR / "reference_free_ideal_stats.json"
    _preserve_legacy_baseline()
    config = load_detection_config(PROJECT_ROOT / "config" / "detection.yaml")
    entries = _load_dev_entries(PROJECT_ROOT, config)
    references = {
        source: _fit_references_leave_one_passage_out(
            entries, source, config["measurements"]["pace"]["window_words"]
        )
        for source in TIMING_SOURCES
    }
    REFERENCE_PATH.write_text(
        json.dumps(references, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    tables = _prepare_tables(entries, config, references)
    _run_pre_tuning_diagnostics(entries, tables, config)
    config, tuning_rows = _tune_thresholds(entries, tables, config)
    summaries: list[dict] = []
    type_rows: list[dict] = []
    severity_rows: list[dict] = []
    control_rows: list[dict] = []
    predictions: dict[tuple[str, str], dict[str, list[dict]]] = {}
    for source in TIMING_SOURCES:
        for mode in MODES:
            summary, types, severity, controls, mode_predictions = _evaluate_variant(
                mode, source, entries, tables[source][mode], config
            )
            summaries.append(summary)
            type_rows.extend(types)
            severity_rows.extend(severity)
            control_rows.extend(controls)
            predictions[(source, mode)] = mode_predictions
    _write_reports(
        summaries, type_rows, severity_rows, control_rows, tuning_rows, predictions, config
    )
    print(f"DEV recordings evaluated: {len(entries)}")
    for row in summaries:
        print(
            f"{row['timing_source']}/{row['mode']}: "
            f"F1@0.3={row['f1_iou_0.3']:.3f} "
            f"F1@0.5={row['f1_iou_0.5']:.3f} "
            f"boundary MAE={row['mean_boundary_error_ms']:.1f} ms "
            f"combined FP/min={row['false_regions_per_minute_all_controls_and_ideals']:.3f}"
        )
    print(f"Reference artifact: {REFERENCE_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    print("Frozen config checksum: eval/results/frozen_config.sha256")
    disabled = {
        flaw_type: config["detectors"][flaw_type].get("disabled_reason", "")
        for flaw_type in DETECTOR_TYPES
        if not config["detectors"][flaw_type].get("enabled", True)
    }
    print(f"Disabled detector types: {json.dumps(disabled, sort_keys=True)}")
    return summaries


def main() -> None:
    """Run deterministic DEV-only evaluation."""
    run_evaluation()


if __name__ == "__main__":
    main()