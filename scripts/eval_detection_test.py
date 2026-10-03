"""Evaluate frozen DEV-tuned detection thresholds once on TEST data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from speechlens.detection.core import load_detection_config
from speechlens.detection.measurements import fit_reference_free
from scripts import eval_detection as evaluator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"


def _verify_frozen_config(project_root: Path) -> dict:
    """Fail closed unless the configured thresholds match the DEV freeze."""
    config_path = project_root / "config" / "detection.yaml"
    checksum_path = project_root / "eval" / "results" / "frozen_config.sha256"
    if not checksum_path.is_file():
        raise RuntimeError("DEV-frozen config checksum is missing; run `python run.py eval` first")
    checksum_parts = checksum_path.read_text(encoding="ascii").strip().split()
    actual = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if len(checksum_parts) < 2 or checksum_parts[0] != actual:
        raise RuntimeError("config/detection.yaml does not match eval/results/frozen_config.sha256")
    return load_detection_config(config_path)


def _test_references(dev_entries: list[dict], test_entries: list[dict], config: dict) -> dict:
    """Fit reference-free statistics from DEV IDEALs and reuse them on TEST."""
    ideal_records = [
        {
            "bundle": entry["timing_bundles"]["realistic"]["bundle"],
            "timings": entry["timing_bundles"]["realistic"]["timings"],
            "transcript": entry["transcript"],
        }
        for entry in dev_entries
        if entry["manifest"]["kind"] == "ideal"
    ]
    reference = fit_reference_free(
        ideal_records,
        config["measurements"]["pace"]["window_words"],
    )
    passages = {str(entry["manifest"]["passage_id"]) for entry in test_entries}
    return {source: {passage: reference for passage in passages} for source in evaluator.TIMING_SOURCES}


def _boundary_rows(entries: list[dict], predictions: dict[str, list[dict]]) -> list[dict]:
    """Return signed boundary errors for regions matched at IoU 0.3."""
    rows: list[dict] = []
    for entry in entries:
        recording_id = entry["manifest"]["recording_id"]
        predicted = predictions[recording_id]
        actual = entry["ground_truth"]
        for predicted_index, actual_index, iou in evaluator._match_regions(predicted, actual, 0.3):
            prediction = predicted[predicted_index]
            label = actual[actual_index]
            rows.append({
                "recording_id": recording_id,
                "type": label["type"],
                "iou": iou,
                "start_error_ms": (prediction["start_s"] - label["start_s"]) * 1000.0,
                "end_error_ms": (prediction["end_s"] - label["end_s"]) * 1000.0,
                "start_absolute_error_ms": abs(prediction["start_s"] - label["start_s"]) * 1000.0,
                "end_absolute_error_ms": abs(prediction["end_s"] - label["end_s"]) * 1000.0,
            })
    return rows


def run_test_evaluation(project_root: Path = PROJECT_ROOT) -> list[dict]:
    """Evaluate frozen thresholds on TEST recordings without tuning or config writes."""
    project_root = project_root.resolve()
    config = _verify_frozen_config(project_root)
    dev_entries = evaluator._load_entries(project_root, config, "dev")
    test_entries = evaluator._load_entries(project_root, config, "test")
    if not test_entries:
        raise RuntimeError("TEST split is empty; refusing to emit evaluation reports")
    references = _test_references(dev_entries, test_entries, config)
    tables = evaluator._prepare_tables(test_entries, config, references)

    summaries: list[dict] = []
    type_rows: list[dict] = []
    severity_rows: list[dict] = []
    control_rows: list[dict] = []
    boundary_rows: list[dict] = []
    for mode in evaluator.MODES:
        summary, types, severity, controls, predictions = evaluator._evaluate_variant(
            mode,
            "realistic",
            test_entries,
            tables["realistic"][mode],
            config,
        )
        summaries.append(summary)
        type_rows.extend(types)
        severity_rows.extend(severity)
        control_rows.extend(controls)
        boundary_rows.extend(_boundary_rows(test_entries, predictions))

    results_dir = project_root / "eval" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).drop(columns=["confusion"], errors="ignore").to_csv(
        results_dir / "test_detection_summary.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(type_rows).to_csv(
        results_dir / "test_detection_type_metrics.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(severity_rows).to_csv(
        results_dir / "test_detection_severity_recall.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(control_rows).to_csv(
        results_dir / "test_detection_control_false_positives.csv", index=False,
        lineterminator="\n",
    )
    pd.DataFrame(boundary_rows, columns=[
        "recording_id", "type", "iou", "start_error_ms", "end_error_ms",
        "start_absolute_error_ms", "end_absolute_error_ms",
    ]).to_csv(results_dir / "test_detection_boundary_error.csv", index=False, lineterminator="\n")
    ci_rows = [{
        "mode": row["mode"],
        "f1_iou_0.3": row["f1_iou_0.3"],
        "bootstrap_95_ci_low": row.get("f1_iou_0.3_ci95_low", 0.0),
        "bootstrap_95_ci_high": row.get("f1_iou_0.3_ci95_high", 0.0),
        "bootstrap_samples": config["evaluation"]["bootstrap_samples"],
        "bootstrap_seed": config["evaluation"]["bootstrap_seed"],
    } for row in summaries]
    pd.DataFrame(ci_rows).to_csv(
        results_dir / "test_detection_bootstrap_ci.csv", index=False, lineterminator="\n"
    )
    print(f"TEST recordings evaluated: {len(test_entries)}")
    for row in summaries:
        print(
            f"{row['mode']}: F1@0.3={row['f1_iou_0.3']:.3f} "
            f"F1@0.5={row['f1_iou_0.5']:.3f} "
            f"boundary MAE={row['mean_boundary_error_ms']:.1f} ms "
            f"combined FP/min={row['false_regions_per_minute_all_controls_and_ideals']:.3f}"
        )
    return summaries


def main() -> None:
    """Run the frozen TEST evaluation."""
    run_test_evaluation()


if __name__ == "__main__":
    main()
