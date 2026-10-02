"""Evaluate and tune flaw-region detection using DEV recordings only."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from speechlens.detection.core import (
    detect_deviation_regions,
    fit_reference_free,
    paired_deviation_table,
    reference_free_deviation_table,
    save_reference_artifact,
    word_feature_rows,
    load_detection_config,
)
from speechlens.explain import explanation_record
from speechlens.features import extract_recording_features
from speechlens.schema import WordTiming


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
REFERENCE_PATH = RESULTS_DIR / "reference_free_ideal_stats.json"
FLAW_TYPES = (
    "pace_fast",
    "pace_slow",
    "long_pause",
    "monotone",
    "volume_dropoff",
    "filler",
    "stumble_repeat",
    "other_deviation",
)
IOU_THRESHOLDS = (0.3, 0.5)
FALSE_REGION_BUDGET_PER_MINUTE = 1.0


def _load_dev_rows(project_root: Path) -> list[dict[str, str]]:
    """Read only DEV manifest rows; never resolve a held-out asset path."""
    manifest_path = project_root / "data" / "labels" / "manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        return [row for row in csv.DictReader(manifest_file) if row.get("split") == "dev"]


def _load_dev_recordings(project_root: Path) -> tuple[list[dict], dict[str, Any]]:
    """Load DEV sidecars, passage text, audio, and features only."""
    dev_rows = _load_dev_rows(project_root)
    entries: list[dict] = []
    bundles: dict[str, Any] = {}
    passage_texts: dict[str, str] = {}
    for row in dev_rows:
        recording_id = row["recording_id"]
        label_path = project_root / "data" / "labels" / f"{recording_id}.json"
        sidecar = json.loads(label_path.read_text(encoding="utf-8"))
        if sidecar.get("split") != "dev":
            raise ValueError(f"DEV manifest row has non-DEV sidecar: {recording_id}")
        passage_id = row["passage_id"]
        if passage_id not in passage_texts:
            passage_path = project_root / "data" / "labels" / "passages" / f"{passage_id}.json"
            passage = json.loads(passage_path.read_text(encoding="utf-8"))
            if passage.get("split") != "dev":
                raise ValueError(f"DEV recording references a non-DEV passage: {passage_id}")
            passage_texts[passage_id] = str(passage["transcript"])
        audio_path = project_root / row["path"]
        audio, sample_rate_hz = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio.ndim != 1 or sample_rate_hz != 16000:
            raise ValueError(f"DEV recording must be mono 16 kHz audio: {recording_id}")
        timings = [WordTiming.model_validate(item) for item in sidecar["word_timings"]]
        bundle = extract_recording_features(audio, timings, passage_texts[passage_id])
        bundles[recording_id] = bundle
        pair = sidecar.get("pair")
        entries.append(
            {
                "manifest": row,
                "sidecar": sidecar,
                "timings": timings,
                "transcript": passage_texts[passage_id],
                "bundle": bundle,
                "ground_truth": [
                    {
                        "start_s": float(flaw["rendered_start_s"]),
                        "end_s": float(flaw["rendered_end_s"]),
                        "type": str(flaw["flaw_type"]),
                        "severity": float(flaw["severity"]),
                    }
                    for flaw in (pair["flaws"] if pair else [])
                ],
                "ideal_recording_id": (
                    pair["ideal_recording_id"]
                    if pair
                    else sidecar["recording"].get("parent_recording_id")
                    or recording_id
                ),
            }
        )
    return entries, bundles


def _interval_iou(first: dict, second: dict) -> float:
    """Compute one-dimensional temporal intersection-over-union."""
    intersection = max(
        0.0,
        min(float(first["end_s"]), float(second["end_s"]))
        - max(float(first["start_s"]), float(second["start_s"])),
    )
    union = max(float(first["end_s"]), float(second["end_s"])) - min(
        float(first["start_s"]), float(second["start_s"])
    )
    return intersection / union if union > 0 else 0.0


def _match_regions(
    predicted: list[dict], actual: list[dict], threshold: float
) -> list[tuple[int, int, float]]:
    """Greedily match region pairs by descending IoU without reusing either."""
    candidates = [
        (predicted_index, actual_index, _interval_iou(prediction, label))
        for predicted_index, prediction in enumerate(predicted)
        for actual_index, label in enumerate(actual)
    ]
    candidates.sort(key=lambda item: (-item[2], item[0], item[1]))
    used_predictions: set[int] = set()
    used_actual: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for predicted_index, actual_index, iou in candidates:
        if iou < threshold:
            continue
        if predicted_index in used_predictions or actual_index in used_actual:
            continue
        used_predictions.add(predicted_index)
        used_actual.add(actual_index)
        matches.append((predicted_index, actual_index, iou))
    return matches


def _control_group(entry: dict) -> str | None:
    """Map a DEV control ID to the required false-positive group."""
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
    return "other_control"


def _prepare_tables(entries: list[dict], config: dict, reference: dict) -> dict[str, dict[str, pd.DataFrame]]:
    """Construct paired and reference-free per-frame tables for DEV rows."""
    by_id = {entry["manifest"]["recording_id"]: entry for entry in entries}
    tables: dict[str, dict[str, pd.DataFrame]] = {"paired": {}, "reference_free": {}}
    for entry_index, entry in enumerate(entries, start=1):
        row = entry["manifest"]
        recording_id = row["recording_id"]
        ideal = by_id[entry["ideal_recording_id"]]
        if recording_id == entry["ideal_recording_id"]:
            paired = entry["bundle"].frames.copy()
            paired["deviation"] = 0.0
            for column, default in (
                ("rate_z", 0.0), ("intensity_z", 0.0), ("f0_std_ratio", 1.0),
                ("f0_z", 0.0), ("pause_excess_s", 0.0), ("repeat_score", 0.0),
                ("nonlexical_voiced", False), ("alignment_confidence", 1.0),
                ("expected_rate_sps", 0.0), ("observed_rate_sps", 0.0),
                ("expected_intensity_db", 0.0), ("observed_intensity_db", 0.0),
                ("expected_f0_std", 0.0), ("observed_f0_std", 0.0),
            ):
                paired[column] = default
        else:
            print(
                f"Preparing paired table {entry_index}/{len(entries)}: {recording_id}",
                flush=True,
            )
            paired = paired_deviation_table(
                entry["bundle"], ideal["bundle"], entry["timings"], ideal["timings"],
                entry["transcript"], config,
            )
        tables["paired"][recording_id] = paired
        tables["reference_free"][recording_id] = reference_free_deviation_table(
            entry["bundle"], entry["timings"], entry["transcript"], reference, config
        )
    return tables


def _evaluate_mode(
    mode: str,
    entries: list[dict],
    tables: dict[str, pd.DataFrame],
    config: dict,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """Compute region, classification, severity, and false-positive metrics."""
    predictions: dict[str, list[dict]] = {}
    true_positive: Counter = Counter()
    false_positive: Counter = Counter()
    false_negative: Counter = Counter()
    boundaries: list[float] = []
    type_counts: Counter = Counter()
    severity_counts: dict[str, Counter] = defaultdict(Counter)
    confusion: Counter = Counter()
    control_counts: Counter = Counter()
    control_minutes: Counter = Counter()
    for entry in entries:
        row = entry["manifest"]
        recording_id = row["recording_id"]
        prediction = detect_deviation_regions(tables[recording_id], entry["timings"], config)
        predictions[recording_id] = prediction
        actual = entry["ground_truth"]
        if not actual:
            group = _control_group(entry)
            if group is not None:
                control_counts[group] += len(prediction)
                control_minutes[group] += float(row["duration_s"]) / 60.0

        matches = _match_regions(prediction, actual, 0.3)
        matched_prediction = {item[0] for item in matches}
        matched_actual = {item[1] for item in matches}
        true_positive["all"] += len(matches)
        false_positive["all"] += len(prediction) - len(matched_prediction)
        false_negative["all"] += len(actual) - len(matched_actual)
        for predicted_index, actual_index, _ in matches:
            predicted_region, actual_region = prediction[predicted_index], actual[actual_index]
            boundaries.extend(
                [
                    abs(float(predicted_region["start_s"]) - float(actual_region["start_s"])) * 1000.0,
                    abs(float(predicted_region["end_s"]) - float(actual_region["end_s"])) * 1000.0,
                ]
            )
            confusion[(actual_region["type"], predicted_region["type"])] += 1
        for flaw_type in FLAW_TYPES[:-1]:
            typed_actual = [flaw for flaw in actual if flaw["type"] == flaw_type]
            type_counts[(flaw_type, "actual")] += len(typed_actual)
            type_counts[(flaw_type, "matched")] += sum(
                actual[index]["type"] == flaw_type
                and prediction[predicted_index]["type"] == flaw_type
                for predicted_index, index, _ in matches
            )
        severity_level = int(row["severity_level"])
        if row["kind"] == "injected":
            key = f"L{severity_level}"
            severity_counts[key]["actual"] += len(actual)
            severity_counts[key]["matched"] += len(matched_actual)
    summary: dict[str, Any] = {"mode": mode}
    for threshold in IOU_THRESHOLDS:
        tp = fp = fn = 0
        for entry in entries:
            predicted = predictions[entry["manifest"]["recording_id"]]
            actual = entry["ground_truth"]
            matched = _match_regions(predicted, actual, threshold)
            tp += len(matched)
            fp += len(predicted) - len({item[0] for item in matched})
            fn += len(actual) - len({item[1] for item in matched})
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        summary[f"precision_iou_{threshold:.1f}"] = precision
        summary[f"recall_iou_{threshold:.1f}"] = recall
        summary[f"f1_iou_{threshold:.1f}"] = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    summary["mean_boundary_error_ms"] = float(np.mean(boundaries)) if boundaries else 0.0
    total_actual = sum(type_counts[(flaw_type, "actual")] for flaw_type in FLAW_TYPES[:-1])
    summary["actual_regions"] = total_actual
    type_rows = [
        {
            "mode": mode,
            "flaw_type": flaw_type,
            "actual": type_counts[(flaw_type, "actual")],
            "matched": type_counts[(flaw_type, "matched")],
            "recall": type_counts[(flaw_type, "matched")]
            / max(1, type_counts[(flaw_type, "actual")]),
        }
        for flaw_type in FLAW_TYPES[:-1]
    ]
    severity_rows = [
        {
            "mode": mode,
            "severity_level": level,
            "actual": severity_counts[level]["actual"],
            "matched": severity_counts[level]["matched"],
            "recall": severity_counts[level]["matched"]
            / max(1, severity_counts[level]["actual"]),
        }
        for level in (f"L{value}" for value in range(1, 6))
    ]
    control_rows = [
        {
            "mode": mode,
            "control_type": group,
            "false_regions": control_counts[group],
            "minutes": control_minutes[group],
            "false_regions_per_minute": control_counts[group]
            / max(control_minutes[group], 1e-12),
        }
        for group in ("ideal", "identity", "mp3", "gain", "noise")
    ]
    summary["false_regions_per_minute_all_controls_and_ideals"] = sum(
        row["false_regions"] for row in control_rows
    ) / max(sum(row["minutes"] for row in control_rows), 1e-12)
    summary["confusion"] = confusion
    return summary, type_rows, severity_rows, control_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows as deterministic CSV with a stable field order."""
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    pd.DataFrame(rows, columns=fields).to_csv(path, index=False, lineterminator="\n")


def _tune_thresholds(
    entries: list[dict],
    tables_by_mode: dict[str, dict[str, pd.DataFrame]],
    config: dict,
) -> tuple[dict, list[dict]]:
    """Select a shared enter threshold by DEV F1 under the false-region budget."""
    original = dict(config)
    candidates = (
        1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0, 10.0,
        12.0, 16.0, 20.0,
    )
    rows: list[dict] = []
    best: tuple[float, float, float, dict] | None = None
    baseline_summaries = {
        mode: _evaluate_mode(mode, entries, tables, original)[0]
        for mode, tables in tables_by_mode.items()
    }
    baseline_controls = {
        mode: _evaluate_mode(mode, entries, tables, original)[3]
        for mode, tables in tables_by_mode.items()
    }
    baseline_rates = [
        float(row["false_regions_per_minute"])
        for groups in baseline_controls.values()
        for row in groups
    ]
    rows.append(
        {
            "stage": "before_tuning",
            "enter_threshold": original["enter_threshold"],
            "exit_threshold": original["exit_threshold"],
            "paired_dev_f1_iou_0_3": baseline_summaries["paired"]["f1_iou_0.3"],
            "reference_free_dev_f1_iou_0_3": baseline_summaries["reference_free"]["f1_iou_0.3"],
            "mean_false_regions_per_minute": float(np.mean(baseline_rates)),
            "maximum_group_false_regions_per_minute": max(baseline_rates, default=0.0),
            "budget_per_minute": FALSE_REGION_BUDGET_PER_MINUTE,
            "budget_met": max(baseline_rates, default=0.0)
            <= FALSE_REGION_BUDGET_PER_MINUTE,
        }
    )
    for enter in candidates:
        candidate_config = dict(config)
        candidate_config["enter_threshold"] = enter
        candidate_config["exit_threshold"] = min(float(config["exit_threshold"]), enter * 0.5)
        f1_values: list[float] = []
        false_rates: list[float] = []
        group_rates: list[float] = []
        for mode, tables in tables_by_mode.items():
            summary, _, _, controls = _evaluate_mode(
                mode, entries, tables, candidate_config
            )
            f1_values.append(float(summary["f1_iou_0.3"]))
            false_rates.append(
                float(summary["false_regions_per_minute_all_controls_and_ideals"])
            )
            group_rates.extend(
                float(row["false_regions_per_minute"]) for row in controls
            )
        mean_f1 = float(np.mean(f1_values))
        mean_false_rate = float(np.mean(false_rates))
        maximum_group_false_rate = max(group_rates, default=0.0)
        budget_met = maximum_group_false_rate <= FALSE_REGION_BUDGET_PER_MINUTE
        rows.append(
            {
                "stage": "candidate",
                "enter_threshold": enter,
                "exit_threshold": candidate_config["exit_threshold"],
                "paired_dev_f1_iou_0_3": f1_values[0],
                "reference_free_dev_f1_iou_0_3": f1_values[1],
                "mean_false_regions_per_minute": mean_false_rate,
                "maximum_group_false_regions_per_minute": maximum_group_false_rate,
                "budget_per_minute": FALSE_REGION_BUDGET_PER_MINUTE,
                "budget_met": budget_met,
            }
        )
        if budget_met and (best is None or mean_f1 > best[0]):
            best = (mean_f1, mean_false_rate, enter, candidate_config)
    if best is None:
        candidate = min(
            (row for row in rows if row.get("stage") == "candidate"),
            key=lambda row: row["maximum_group_false_regions_per_minute"],
        )
        chosen_enter = float(candidate["enter_threshold"])
        config["enter_threshold"] = chosen_enter
        config["exit_threshold"] = float(candidate["exit_threshold"])
    else:
        config["enter_threshold"] = best[2]
        config["exit_threshold"] = best[3]["exit_threshold"]
    rows.append(
        {
            "stage": "after_tuning",
            "enter_threshold": config["enter_threshold"],
            "exit_threshold": config["exit_threshold"],
            "paired_dev_f1_iou_0_3": np.nan,
            "reference_free_dev_f1_iou_0_3": np.nan,
            "mean_false_regions_per_minute": np.nan,
            "maximum_group_false_regions_per_minute": np.nan,
            "budget_per_minute": FALSE_REGION_BUDGET_PER_MINUTE,
            "budget_met": "pending_final_metrics",
        }
    )
    return config, rows


def _write_reports(
    summaries: list[dict],
    type_rows: list[dict],
    severity_rows: list[dict],
    control_rows: list[dict],
    tuning_rows: list[dict],
    entries: list[dict],
    tables_by_mode: dict[str, dict[str, pd.DataFrame]],
    config: dict,
) -> None:
    """Write CSV/PNG reports, deterministic examples, and frozen config hash."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(RESULTS_DIR / "detection_summary.csv", summaries)
    _write_csv(RESULTS_DIR / "detection_type_recall.csv", type_rows)
    _write_csv(RESULTS_DIR / "detection_severity_recall.csv", severity_rows)
    _write_csv(RESULTS_DIR / "detection_control_false_positives.csv", control_rows)
    _write_csv(RESULTS_DIR / "detection_threshold_tuning.csv", tuning_rows)
    for summary in summaries:
        mode = summary["mode"]
        confusion = summary.pop("confusion")
        matrix = pd.DataFrame(0, index=FLAW_TYPES[:-1], columns=FLAW_TYPES)
        for (actual_type, predicted_type), count in confusion.items():
            matrix.loc[actual_type, predicted_type] = count
        matrix.to_csv(RESULTS_DIR / f"detection_confusion_{mode}.csv", lineterminator="\n")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for mode in ("paired", "reference_free"):
        selected = [row for row in type_rows if row["mode"] == mode]
        axes[0].plot(
            [row["flaw_type"] for row in selected],
            [row["recall"] for row in selected],
            marker="o",
            label=mode,
        )
        selected_severity = [row for row in severity_rows if row["mode"] == mode]
        axes[1].plot(
            [row["severity_level"] for row in selected_severity],
            [row["recall"] for row in selected_severity],
            marker="o",
            label=mode,
        )
    axes[0].set_title("DEV recall by flaw type")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].set_title("DEV sensitivity by severity")
    axes[1].set_ylim(0.0, 1.0)
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend()
    figure.savefig(RESULTS_DIR / "detection_recall.png", dpi=160)
    plt.close(figure)

    examples: list[dict] = []
    for mode in ("paired", "reference_free"):
        all_records: list[dict] = []
        for entry in entries:
            regions = detect_deviation_regions(
                tables_by_mode[mode][entry["manifest"]["recording_id"]],
                entry["timings"],
                config,
            )
            for region in regions:
                record = explanation_record(region)
                record["recording_id"] = entry["manifest"]["recording_id"]
                all_records.append(record)
                if mode == "paired" and len(examples) < 10:
                    examples.append(record)
        (RESULTS_DIR / f"detection_explanations_{mode}.json").write_text(
            json.dumps(all_records, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (RESULTS_DIR / "detection_explanation_examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config_path = PROJECT_ROOT / "config" / "detection.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    (RESULTS_DIR / "frozen_config.sha256").write_text(
        f"{digest}  config/detection.yaml\n", encoding="ascii"
    )


def run_evaluation(project_root: Path = PROJECT_ROOT) -> list[dict]:
    """Run both detector modes on DEV only, tune, freeze, and report metrics."""
    global PROJECT_ROOT, RESULTS_DIR, REFERENCE_PATH
    PROJECT_ROOT = project_root.resolve()
    RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
    REFERENCE_PATH = RESULTS_DIR / "reference_free_ideal_stats.json"
    entries, _ = _load_dev_recordings(PROJECT_ROOT)
    ideal_rows = [
        word_feature_rows(entry["bundle"], entry["timings"], entry["transcript"])
        for entry in entries
        if entry["manifest"]["kind"] == "ideal"
    ]
    reference = fit_reference_free(ideal_rows)
    save_reference_artifact(reference, REFERENCE_PATH)
    config = load_detection_config(PROJECT_ROOT / "config" / "detection.yaml")
    tables_by_mode = _prepare_tables(entries, config, reference)
    config, tuning_rows = _tune_thresholds(entries, tables_by_mode, config)
    summaries: list[dict] = []
    type_rows: list[dict] = []
    severity_rows: list[dict] = []
    control_rows: list[dict] = []
    for mode, tables in tables_by_mode.items():
        summary, mode_types, mode_severity, mode_controls = _evaluate_mode(
            mode, entries, tables, config
        )
        summaries.append(summary)
        type_rows.extend(mode_types)
        severity_rows.extend(mode_severity)
        control_rows.extend(mode_controls)
    final_control_rates = [
        float(row["false_regions_per_minute"]) for row in control_rows
    ]
    after_row = next(row for row in tuning_rows if row.get("stage") == "after_tuning")
    after_row.update(
        {
            "paired_dev_f1_iou_0_3": summaries[0]["f1_iou_0.3"],
            "reference_free_dev_f1_iou_0_3": summaries[1]["f1_iou_0.3"],
            "mean_false_regions_per_minute": float(
                np.mean([
                    summary["false_regions_per_minute_all_controls_and_ideals"]
                    for summary in summaries
                ])
            ),
            "maximum_group_false_regions_per_minute": max(final_control_rates, default=0.0),
            "budget_met": max(final_control_rates, default=0.0)
            <= FALSE_REGION_BUDGET_PER_MINUTE,
        }
    )
    _write_reports(
        summaries,
        type_rows,
        severity_rows,
        control_rows,
        tuning_rows,
        entries,
        tables_by_mode,
        config,
    )
    for summary in summaries:
        print(
            f"{summary['mode']}: F1@0.3={summary['f1_iou_0.3']:.3f}, "
            f"recall@0.3={summary['recall_iou_0.3']:.3f}, "
            f"F1@0.5={summary['f1_iou_0.5']:.3f}, "
            f"recall@0.5={summary['recall_iou_0.5']:.3f}, "
            f"boundary MAE={summary['mean_boundary_error_ms']:.1f} ms, "
            f"control/ideal FP/min="
            f"{summary['false_regions_per_minute_all_controls_and_ideals']:.3f}"
        )
    print(f"DEV recordings evaluated: {len(entries)}")
    print(f"Reference artifact: {REFERENCE_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    print("Frozen config checksum: eval/results/frozen_config.sha256")
    return summaries


def main() -> None:
    """Run the deterministic DEV-only detection evaluation."""
    run_evaluation()


if __name__ == "__main__":
    main()