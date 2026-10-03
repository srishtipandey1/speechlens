"""Measure frame-score separation on DEV labels using existing detection caches."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts import eval_detection as evaluator
from scripts.eval_detection import _match_regions
from speechlens.detection.core import load_detection_config
from speechlens.detection.measurements import DETECTOR_TYPES, detect_typed_regions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "eval" / "results" / "score_diagnostics.csv"


def _frame_masks(table: pd.DataFrame, entry: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return rendered-timeline flaw and injected-target masks for table frames."""
    times = table["time_s"].to_numpy(dtype=np.float64)
    any_flaw = np.zeros(len(table), dtype=bool)
    for flaw in entry["ground_truth"]:
        any_flaw |= (times >= flaw["start_s"]) & (times < flaw["end_s"])
    return any_flaw, times


def _scores_for_mode(
    entries: list[dict], tables: dict[str, pd.DataFrame], mode: str, flaw_type: str
) -> tuple[np.ndarray, np.ndarray, list[tuple[dict, dict]]]:
    """Collect positive flaw frames and clean/IDEAL/control negative frames."""
    positive: list[np.ndarray] = []
    negative: list[np.ndarray] = []
    target_labels: list[tuple[dict, dict]] = []
    score_column = f"score_{flaw_type}"
    for entry in entries:
        recording_id = entry["manifest"]["recording_id"]
        table = tables[recording_id]
        scores = table[score_column].fillna(0.0).to_numpy(dtype=np.float64)
        any_flaw, times = _frame_masks(table, entry)
        if entry["manifest"]["kind"] == "injected":
            target_mask = np.zeros(len(table), dtype=bool)
            for label in entry["ground_truth"]:
                if label["type"] != flaw_type:
                    continue
                region_mask = (times >= label["start_s"]) & (times < label["end_s"])
                target_mask |= region_mask
                target_labels.append((entry, label))
            if target_mask.any():
                positive.append(scores[target_mask])
            clean_mask = ~any_flaw
            if clean_mask.any():
                negative.append(scores[clean_mask])
        elif entry["manifest"]["kind"] in {"ideal", "control"}:
            negative.append(scores)
    positives = np.concatenate(positive) if positive else np.asarray([], dtype=np.float64)
    negatives = np.concatenate(negative) if negative else np.asarray([], dtype=np.float64)
    return positives, negatives, target_labels


def _low_threshold_region_recall(
    entries: list[dict],
    tables: dict[str, pd.DataFrame],
    config: dict,
    mode: str,
    flaw_type: str,
    labels: list[tuple[dict, dict]],
) -> float:
    """Recall labeled regions after detection at the configured near-zero threshold."""
    low_config = deepcopy(config)
    settings = low_config["detectors"][flaw_type]
    threshold = float(config["evaluation"]["diagnostic_low_threshold"])
    settings["enabled"] = True
    settings["enter_threshold"] = threshold
    settings["exit_threshold"] = threshold * 0.5
    settings.setdefault("enabled_modes", {})[mode] = True
    settings.setdefault("thresholds_by_mode", {})[mode] = {
        "enter_threshold": threshold,
        "exit_threshold": threshold * 0.5,
    }
    by_recording: dict[str, list[dict]] = {}
    entry_by_id = {
        entry["manifest"]["recording_id"]: entry
        for entry in entries
        if entry["manifest"]["kind"] == "injected"
    }
    for recording_id, entry in entry_by_id.items():
        by_recording[recording_id] = detect_typed_regions(
            tables[recording_id],
            entry["timing_bundles"]["realistic"]["timings"],
            low_config,
            flaw_types=(flaw_type,),
            mode=mode,
        )
    matched = 0
    for entry, label in labels:
        recording_id = entry["manifest"]["recording_id"]
        candidates = [region for region in by_recording[recording_id] if region["type"] == flaw_type]
        if _match_regions(candidates, [label], 0.3):
            matched += 1
    return matched / len(labels) if labels else float("nan")


def diagnose(stage: str, *, refresh_scores: bool = False) -> list[dict]:
    """Write per-type/mode separation metrics without recomputing alignments."""
    config = load_detection_config(PROJECT_ROOT / "config" / "detection.yaml")
    entries = evaluator._load_entries(
        PROJECT_ROOT, config, "dev", require_cached_inputs=True
    )
    references = {
        source: evaluator._fit_references_leave_one_passage_out(
            entries, source, config["measurements"]["pace"]["window_words"]
        )
        for source in evaluator.TIMING_SOURCES
    }
    tables = evaluator._prepare_tables(
        entries,
        config,
        references,
        require_cached=not refresh_scores,
    )
    rows: list[dict] = []
    for mode in evaluator.MODES:
        for flaw_type in DETECTOR_TYPES:
            positives, negatives, labels = _scores_for_mode(
                entries, tables["realistic"][mode], mode, flaw_type
            )
            auc = (
                float(roc_auc_score(
                    np.r_[np.ones(len(positives)), np.zeros(len(negatives))],
                    np.r_[positives, negatives],
                ))
                if positives.size and negatives.size
                else float("nan")
            )
            recall = _low_threshold_region_recall(
                entries,
                tables["realistic"][mode],
                config,
                mode,
                flaw_type,
                labels,
            )
            rows.append({
                "stage": stage,
                "mode": mode,
                "flaw_type": flaw_type,
                "positive_frames": int(positives.size),
                "negative_frames": int(negatives.size),
                "median_inside": float(np.median(positives)) if positives.size else float("nan"),
                "median_outside": float(np.median(negatives)) if negatives.size else float("nan"),
                "frame_auc": auc,
                "low_threshold_region_recall": recall,
                "labeled_regions": len(labels),
            })
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict] = []
    if RESULTS_PATH.is_file():
        with RESULTS_PATH.open(newline="", encoding="utf-8") as stream:
            existing_rows = [
                row for row in csv.DictReader(stream) if row.get("stage") != stage
            ]
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(existing_rows + rows)
    print(f"Score diagnostics ({stage})")
    print("mode             detector          inside    outside   AUC     low-threshold recall")
    for row in rows:
        print(
            f"{row['mode']:<16} {row['flaw_type']:<17} "
            f"{row['median_inside']:>8.3f} {row['median_outside']:>9.3f} "
            f"{row['frame_auc']:>7.3f} {row['low_threshold_region_recall']:>10.3f}"
        )
    print(f"Wrote {RESULTS_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    return rows


def main() -> None:
    """Run cache-only DEV score diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("before", "after"), default="before")
    parser.add_argument("--refresh-scores", action="store_true")
    args = parser.parse_args()
    diagnose(args.stage, refresh_scores=args.refresh_scores)


if __name__ == "__main__":
    main()
