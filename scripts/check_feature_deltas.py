"""Measure labeled flaw deltas and control drift using dev recordings only."""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__:
    from scripts.dev_feature_reports import (
        DELTA_METRICS,
        EXPECTED_METRICS,
        _bundle_summary,
        _flaw_delta_row,
        analyze_dev_recording,
        load_dev_manifest_rows,
        load_dev_recording,
        passage_runtime_timer,
    )
else:
    from dev_feature_reports import (
        DELTA_METRICS,
        EXPECTED_METRICS,
        _bundle_summary,
        _flaw_delta_row,
        analyze_dev_recording,
        load_dev_manifest_rows,
        load_dev_recording,
        passage_runtime_timer,
    )


CONTROL_METRICS = (
    "median_f0_hz",
    "f0_std_semitones",
    "median_intensity_raw_dbfs",
    "median_intensity_db",
    "voiced_fraction",
    "mean_spectral_flux",
    "median_hnr_db",
    "energy_modulation_power_3_6_hz",
)
REPORT_METRICS = (*DELTA_METRICS, "repeated_word_occurrences")


def _rank_most_moved(rows: pd.DataFrame, flaw_type: str) -> tuple[str, float, float]:
    """Rank features by median relative absolute change within a flaw type."""
    selected = rows[rows["flaw_type"] == flaw_type]
    candidates: list[tuple[float, str, float]] = []
    for feature in REPORT_METRICS:
        delta_column = f"delta_{feature}"
        baseline_column = f"baseline_{feature}"
        if delta_column not in selected:
            continue
        deltas = selected[delta_column].to_numpy(dtype=np.float64)
        baseline = selected[baseline_column].to_numpy(dtype=np.float64)
        if not deltas.size:
            continue
        baseline_scale = max(float(np.median(np.abs(baseline))), 0.01)
        median_delta = float(np.median(deltas))
        candidates.append((abs(median_delta) / baseline_scale, feature, median_delta))
    if not candidates:
        return "none", 0.0, 0.0
    score, feature, delta = max(candidates)
    return feature, delta, score


def _write_flaw_plot(rows: pd.DataFrame, output_path: Path) -> None:
    """Plot per-label delta in its configured expected feature."""
    flaw_types = list(EXPECTED_METRICS)
    figure, axis = plt.subplots(figsize=(12, 6))
    samples = [
        rows.loc[rows["flaw_type"] == flaw_type, "delta"].to_numpy(dtype=np.float64)
        for flaw_type in flaw_types
    ]
    axis.boxplot(samples, tick_labels=flaw_types, showfliers=False)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("Target feature delta (feature-specific units)")
    axis.set_title("Dev injected recordings: labeled flaw feature deltas")
    axis.tick_params(axis="x", labelrotation=25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _write_control_plot(rows: pd.DataFrame, output_path: Path) -> None:
    """Plot median absolute percent feature drift by control variant."""
    controls = list(rows["control_type"].drop_duplicates())
    values = [
        rows.loc[rows["control_type"] == control, "absolute_percent_drift"].to_numpy(dtype=np.float64)
        for control in controls
    ]
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.boxplot(values, tick_labels=controls, showfliers=False)
    axis.set_ylabel("Absolute drift (%)")
    axis.set_title("Dev control variants: whole-recording feature drift")
    axis.tick_params(axis="x", labelrotation=20)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    """Extract only dev data and write paired flaw/control delta reports."""
    project_root = Path(__file__).resolve().parents[1]
    dev_rows = load_dev_manifest_rows(project_root)
    ideals = {row["passage_id"]: row for row in dev_rows if row["kind"] == "ideal"}
    injections: dict[str, list[dict[str, str]]] = defaultdict(list)
    controls: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in dev_rows:
        if row["kind"] == "injected":
            injections[row["passage_id"]].append(row)
        elif row["kind"] == "control":
            controls[row["passage_id"]].append(row)

    flaw_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    passage_runtimes: list[float] = []
    for passage_id, ideal_row in sorted(ideals.items()):
        started = passage_runtime_timer()
        ideal_audio, ideal_timings, _, ideal_sidecar = load_dev_recording(project_root, ideal_row)
        ideal_bundle = analyze_dev_recording(project_root, ideal_row)
        ideal_summary = _bundle_summary(ideal_bundle)

        for row in injections.get(passage_id, []):
            _, source_timings, _, sidecar = load_dev_recording(project_root, row)
            flawed_audio, _, transcript, _ = load_dev_recording(project_root, row)
            flawed_bundle = analyze_dev_recording(project_root, row)
            for flaw in sidecar["pair"]["flaws"]:
                flaw_rows.append(
                    _flaw_delta_row(
                        flaw,
                        ideal_bundle,
                        flawed_bundle,
                        ideal_timings,
                        source_timings,
                        sidecar,
                    )
                )

        for row in controls.get(passage_id, []):
            control_bundle = analyze_dev_recording(project_root, row)
            control_summary = _bundle_summary(control_bundle)
            for feature in CONTROL_METRICS:
                baseline_value = ideal_summary[feature]
                control_value = control_summary[feature]
                delta = control_value - baseline_value
                scale = max(abs(baseline_value), 1e-8)
                control_rows.append(
                    {
                        "recording_id": row["recording_id"],
                        "passage_id": passage_id,
                        "gender": row["gender"],
                        "control_type": row["recording_id"].split("__", 1)[1],
                        "feature": feature,
                        "ideal_value": baseline_value,
                        "control_value": control_value,
                        "delta": delta,
                        "absolute_percent_drift": abs(delta) / scale * 100.0,
                    }
                )
        passage_runtimes.append(passage_runtime_timer() - started)
        print(f"{passage_id}: dev feature analysis {passage_runtimes[-1]:.2f} s")

    output_dir = project_root / "eval" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    flaw_frame = pd.DataFrame(flaw_rows)
    flaw_csv = output_dir / "feature_deltas_dev.csv"
    flaw_frame.to_csv(flaw_csv, index=False, float_format="%.8f", lineterminator="\n")
    _write_flaw_plot(flaw_frame, output_dir / "feature_deltas_dev.png")

    control_frame = pd.DataFrame(control_rows)
    control_csv = output_dir / "control_drift_dev.csv"
    control_frame.to_csv(control_csv, index=False, float_format="%.8f", lineterminator="\n")
    _write_control_plot(control_frame, output_dir / "control_drift_dev.png")

    print("DEV FLAW SUMMARY (median expected-target delta)")
    failed_expectations: list[str] = []
    for flaw_type, (expected_feature, expected_direction) in EXPECTED_METRICS.items():
        selected = flaw_frame[flaw_frame["flaw_type"] == flaw_type]
        if selected.empty:
            print(f"{flaw_type}: no dev labels")
            failed_expectations.append(flaw_type)
            continue
        median_delta = float(selected["delta"].median())
        expected_moved = median_delta > 0 if expected_direction == "increase" else median_delta < 0
        most_feature, most_delta, relative_effect = _rank_most_moved(flaw_frame, flaw_type)
        print(
            f"{flaw_type}: expected {expected_feature} {expected_direction}; "
            f"median delta={median_delta:+.5f}; strongest={most_feature} "
            f"({most_delta:+.5f}, relative effect {relative_effect:.3f})"
        )
        if not expected_moved:
            failed_expectations.append(flaw_type)
    print(f"Flaw types with expected feature not moving: {failed_expectations or 'none'}")

    print("DEV CONTROL DRIFT (median absolute percent by control)")
    for control_type, group in control_frame.groupby("control_type", sort=True):
        print(f"{control_type}: {group['absolute_percent_drift'].median():.3f}%")
    print(f"Mean feature analysis runtime per dev ideal passage: {float(np.mean(passage_runtimes)):.2f} s")
    print(f"Wrote {flaw_csv.relative_to(project_root).as_posix()}")
    print(f"Wrote {control_csv.relative_to(project_root).as_posix()}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error