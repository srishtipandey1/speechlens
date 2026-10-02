"""Plot raw and speaker-normalized distributions for dev ideal recordings only."""

from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from scripts.dev_feature_reports import analyze_dev_recording, load_dev_manifest_rows
else:
    from dev_feature_reports import analyze_dev_recording, load_dev_manifest_rows


def _speaker_medians(groups: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    """Return equal-weight per-recording medians grouped by speaker gender."""
    medians = {
        gender: np.asarray([np.median(values) for values in speaker_values], dtype=np.float64)
        for gender, speaker_values in groups.items()
        if speaker_values
    }
    if set(medians) != {"M", "F"}:
        raise ValueError("dev ideals must contain both genders")
    return medians


def _standardized_gap(groups: dict[str, list[np.ndarray]]) -> tuple[float, float, float]:
    """Return female-minus-male median gap, pooled SD, and Cohen's d."""
    medians = _speaker_medians(groups)
    gap = float(np.median(medians["F"]) - np.median(medians["M"]))
    pooled_std = float(np.sqrt((np.var(medians["M"]) + np.var(medians["F"])) / 2.0))
    effect_size = gap / pooled_std if pooled_std > np.finfo(np.float64).eps else 0.0
    return gap, pooled_std, effect_size


def _effect_shrink_percent(before: float, after: float) -> float:
    """Return percentage reduction in absolute dimensionless standardized gap."""
    if abs(before) <= np.finfo(np.float64).eps:
        return 0.0
    return 100.0 * (1.0 - abs(after) / abs(before))


def main() -> None:
    """Analyze and plot dev-only ideal feature distributions."""
    project_root = Path(__file__).resolve().parents[1]
    dev_ideals = [
        row
        for row in load_dev_manifest_rows(project_root)
        if row["kind"] == "ideal"
    ]
    metrics = {
        "raw_f0_hz": {"M": [], "F": []},
        "f0_semitones": {"M": [], "F": []},
        "raw_intensity_dbfs": {"M": [], "F": []},
        "intensity_db": {"M": [], "F": []},
    }
    passage_runtimes: list[float] = []

    for row in dev_ideals:
        started = time.perf_counter()
        bundle = analyze_dev_recording(project_root, row)
        passage_runtimes.append(time.perf_counter() - started)
        frames = bundle.frames
        voiced = frames[frames["voiced"]]
        speech = frames[frames["speech_activity"]]
        gender = row["gender"]
        metrics["raw_f0_hz"][gender].append(voiced["f0_hz"].to_numpy(dtype=np.float64))
        metrics["f0_semitones"][gender].append(
            voiced["f0_semitones"].to_numpy(dtype=np.float64)
        )
        metrics["raw_intensity_dbfs"][gender].append(
            speech["intensity_raw_dbfs"].to_numpy(dtype=np.float64)
        )
        metrics["intensity_db"][gender].append(
            speech["intensity_db"].to_numpy(dtype=np.float64)
        )
        print(f"{row['recording_id']}: {passage_runtimes[-1]:.2f} s")

    panels = (
        ("raw_f0_hz", "Raw voiced F0 (Hz)"),
        ("f0_semitones", "Speaker-relative F0 (semitones)"),
        ("raw_intensity_dbfs", "Raw speech intensity (dBFS)"),
        ("intensity_db", "Speaker-relative intensity (dB)"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = ("tab:blue", "tab:orange")
    for axis, (metric, title) in zip(axes.flat, panels):
        male = np.concatenate(metrics[metric]["M"])
        female = np.concatenate(metrics[metric]["F"])
        axis.boxplot([male, female], tick_labels=["Male", "Female"], showfliers=False)
        rng = np.random.default_rng(2401)
        for index, values in enumerate((male, female), start=1):
            stride = max(1, values.size // 3000)
            sampled = values[::stride]
            horizontal = index + rng.uniform(-0.08, 0.08, size=sampled.size)
            axis.scatter(horizontal, sampled, s=2, alpha=0.12, color=colors[index - 1])
        axis.set_title(title)
        axis.set_ylabel(title)

    figure.suptitle("Dev ideal recordings: raw features and per-recording normalization")
    figure.tight_layout()
    output_path = project_root / "eval" / "results" / "normalization_dev.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    f0_before, _, f0_effect_before = _standardized_gap(metrics["raw_f0_hz"])
    f0_after, _, f0_effect_after = _standardized_gap(metrics["f0_semitones"])
    intensity_before, _, intensity_effect_before = _standardized_gap(metrics["raw_intensity_dbfs"])
    intensity_after, _, intensity_effect_after = _standardized_gap(metrics["intensity_db"])
    print(
        f"F0 between-gender median gap: {f0_before:.3f} Hz -> "
        f"{f0_after:.3f} semitones; Cohen's d {f0_effect_before:.3f} -> "
        f"{f0_effect_after:.3f}; effect-size shrink "
        f"{_effect_shrink_percent(f0_effect_before, f0_effect_after):.2f}%"
    )
    print(
        f"Intensity between-gender median gap: {intensity_before:.3f} dBFS -> "
        f"{intensity_after:.3f} dB; Cohen's d {intensity_effect_before:.3f} -> "
        f"{intensity_effect_after:.3f}; effect-size shrink "
        f"{_effect_shrink_percent(intensity_effect_before, intensity_effect_after):.2f}%"
    )
    print(f"Mean runtime per dev ideal passage: {float(np.mean(passage_runtimes)):.2f} s")
    print(f"Saved {output_path.relative_to(project_root).as_posix()}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error