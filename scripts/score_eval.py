"""Score DEV and TEST recordings with frozen detection settings and rubric weights."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from scripts import eval_detection as evaluator
from speechlens.detection.core import load_detection_config
from speechlens.detection.measurements import detect_typed_regions
from speechlens.scoring import DIMENSION_NAMES, aggregate_reference_folds, score_recording


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
SPLITS = ("dev", "test")
MODES = ("paired", "reference_free")


def _load_dev_reference(project_root: Path) -> dict[str, dict]:
    """Load the already-produced DEV-only robust reference artifact."""
    reference_path = project_root / "eval" / "results" / "reference_free_ideal_stats.json"
    if not reference_path.is_file():
        raise FileNotFoundError(
            "DEV-IDEAL reference artifact is missing; run the frozen DEV evaluation first"
        )
    artifact = json.loads(reference_path.read_text(encoding="utf-8"))
    if "realistic" not in artifact:
        raise ValueError("DEV-IDEAL reference artifact has no realistic alignment statistics")
    return artifact


def _references_for_split(
    split: str,
    entries: list[dict],
    dev_reference: dict[str, dict],
) -> dict[str, dict[str, dict]]:
    """Use passage-specific LOO folds for DEV and aggregate DEV folds for TEST."""
    source_reference = dev_reference["realistic"]
    if split == "dev":
        missing = sorted({
            str(entry["manifest"]["passage_id"])
            for entry in entries
            if str(entry["manifest"]["passage_id"]) not in source_reference
        })
        if missing:
            raise ValueError(f"DEV-IDEAL reference folds are missing passages: {missing}")
        passage_refs = {
            str(entry["manifest"]["passage_id"]): source_reference[
                str(entry["manifest"]["passage_id"])
            ]
            for entry in entries
        }
    else:
        aggregate = aggregate_reference_folds(dev_reference)
        passage_refs = {
            str(entry["manifest"]["passage_id"]): aggregate
            for entry in entries
        }
    return {source: passage_refs for source in evaluator.TIMING_SOURCES}


def _score_split(
    split: str,
    entries: list[dict],
    tables: dict,
    references: dict,
    config: dict,
) -> list[dict]:
    """Score every recording in both frozen detector modes."""
    entries_by_id = {
        entry["manifest"]["recording_id"]: entry
        for entry in entries
    }
    scored: list[dict] = []
    for mode in MODES:
        for entry in entries:
            manifest = entry["manifest"]
            recording_id = manifest["recording_id"]
            timing_bundle = entry["timing_bundles"]["realistic"]
            ideal_entry = entries_by_id.get(entry["ideal_recording_id"])
            ideal_timing_bundle = (
                ideal_entry["timing_bundles"]["realistic"]
                if ideal_entry is not None
                else None
            )
            if mode == "paired" and ideal_timing_bundle is None:
                raise ValueError(f"paired IDEAL is missing for {recording_id}")
            reference = references["realistic"][str(manifest["passage_id"])]
            table = tables["realistic"][mode][recording_id]
            regions = detect_typed_regions(
                table,
                timing_bundle["timings"],
                config,
                mode=mode,
            )
            result = score_recording(
                timing_bundle["bundle"],
                timing_bundle["timings"],
                entry["transcript"],
                mode,
                regions,
                config,
                baseline_bundle=(
                    ideal_timing_bundle["bundle"] if ideal_timing_bundle else None
                ),
                baseline_timings=(
                    ideal_timing_bundle["timings"] if ideal_timing_bundle else None
                ),
                reference=reference,
            )
            scored.append({
                "split": split,
                "mode": mode,
                "recording_id": recording_id,
                "passage_id": str(manifest["passage_id"]),
                "kind": manifest["kind"],
                "severity_level": int(manifest["severity_level"]),
                **result["scores"],
                "total": result["total"],
                "enabled_detectors": ";".join(
                    flaw_type
                    for flaw_type, detector in config["detectors"].items()
                    if detector.get("enabled_modes", {}).get(
                        mode, detector.get("enabled", True)
                    )
                ),
            })
    return scored


def _correlation_metrics(
    rows: list[dict],
    split: str,
    mode: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict]:
    """Compute severity Spearman and ideal/flawed AUC with passage bootstrap CI."""
    eligible = [
        row for row in rows
        if row["split"] == split
        and row["mode"] == mode
        and row["kind"] in {"ideal", "injected", "human_flawed"}
    ]
    severity_rho = float("nan")
    if len({row["severity_level"] for row in eligible}) > 1 and len(eligible) > 1:
        severity_rho = float(spearmanr(
            [row["severity_level"] for row in eligible],
            [row["total"] for row in eligible],
        ).statistic)

    by_passage: dict[str, list[dict]] = defaultdict(list)
    for row in eligible:
        by_passage[row["passage_id"]].append(row)
    passage_ids = sorted(by_passage)
    random = np.random.default_rng(bootstrap_seed)
    bootstrap_values: list[float] = []
    if len(passage_ids) >= 2:
        for _ in range(bootstrap_samples):
            selected = random.choice(passage_ids, size=len(passage_ids), replace=True)
            sample = [
                row
                for passage_id in selected
                for row in by_passage[str(passage_id)]
            ]
            if len({row["severity_level"] for row in sample}) < 2:
                continue
            rho = float(spearmanr(
                [row["severity_level"] for row in sample],
                [row["total"] for row in sample],
            ).statistic)
            if np.isfinite(rho):
                bootstrap_values.append(rho)
    ci_low, ci_high = (
        (float(value) for value in np.percentile(bootstrap_values, [2.5, 97.5]))
        if bootstrap_values
        else (float("nan"), float("nan"))
    )

    binary_rows = [row for row in eligible if row["kind"] in {"ideal", "injected", "human_flawed"}]
    labels = [0 if row["kind"] == "ideal" else 1 for row in binary_rows]
    auc = (
        float(roc_auc_score(labels, [-float(row["total"]) for row in binary_rows]))
        if len(set(labels)) == 2
        else float("nan")
    )
    return [
        {
            "split": split,
            "mode": mode,
            "metric": "spearman_total_vs_severity",
            "value": severity_rho,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "recordings": len(eligible),
            "passages": len(passage_ids),
            "bootstrap_samples": len(bootstrap_values),
            "bootstrap_seed": bootstrap_seed,
        },
        {
            "split": split,
            "mode": mode,
            "metric": "auc_ideal_vs_flawed",
            "value": auc,
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "recordings": len(binary_rows),
            "passages": len({row["passage_id"] for row in binary_rows}),
            "bootstrap_samples": 0,
            "bootstrap_seed": bootstrap_seed,
        },
    ]


def run_scoring_evaluation(project_root: Path = PROJECT_ROOT) -> tuple[list[dict], list[dict]]:
    """Evaluate fixed weighted scores on DEV and TEST without threshold tuning."""
    project_root = project_root.resolve()
    config = evaluator.load_detection_config(project_root / "config" / "detection.yaml")
    dev_reference = _load_dev_reference(project_root)
    scored_rows: list[dict] = []
    for split in SPLITS:
        entries = evaluator._load_entries(project_root, config, split)
        if not entries:
            raise RuntimeError(f"{split.upper()} split is empty")
        references = _references_for_split(split, entries, dev_reference)
        tables = evaluator._prepare_tables(entries, config, references)
        scored_rows.extend(_score_split(split, entries, tables, references, config))

    metrics: list[dict] = []
    for split in SPLITS:
        for mode in MODES:
            metrics.extend(_correlation_metrics(
                scored_rows,
                split,
                mode,
                int(config["evaluation"]["bootstrap_samples"]),
                int(config["evaluation"]["bootstrap_seed"]),
            ))

    results_dir = project_root / "eval" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    recording_columns = [
        "split", "mode", "recording_id", "passage_id", "kind", "severity_level",
        *DIMENSION_NAMES, "total", "enabled_detectors",
    ]
    pd.DataFrame(scored_rows, columns=recording_columns).sort_values(
        ["split", "mode", "passage_id", "recording_id"], kind="stable"
    ).to_csv(
        results_dir / "scoring_recordings.csv",
        index=False,
        lineterminator="\n",
        float_format="%.6f",
    )
    pd.DataFrame(metrics).to_csv(
        results_dir / "scoring_metrics.csv",
        index=False,
        lineterminator="\n",
        float_format="%.6f",
    )
    print("split mode metric value ci95_low ci95_high")
    for row in metrics:
        print(
            f"{row['split']} {row['mode']} {row['metric']} {row['value']:.4f} "
            f"{row['ci95_low']:.4f} {row['ci95_high']:.4f}"
        )
    return scored_rows, metrics


def main() -> None:
    """Run the fixed-weight DEV/TEST scoring evaluation."""
    run_scoring_evaluation()


if __name__ == "__main__":
    main()
