"""Generate WAV and waveform-label demos for every flaw injection type."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import parselmouth
import soundfile as sf
import yaml

from speechlens.alignment.align import SAMPLE_RATE_HZ, align, load_audio
from speechlens.injection import FlawSpec, apply_flaws


def f0_semitone_series(
    waveform: np.ndarray,
    start_s: float,
    end_s: float,
    reference_f0_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return voiced pitch times, semitones, and the used reference median F0."""
    config_path = Path("config/injection.yaml")
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    praat_config = config["praat"]
    sound = parselmouth.Sound(waveform, sampling_frequency=SAMPLE_RATE_HZ)
    pitch = sound.to_pitch_ac(
        time_step=float(praat_config["time_step_s"]),
        pitch_floor=float(praat_config["pitch_floor_hz"]),
        pitch_ceiling=float(praat_config["pitch_ceiling_hz"]),
    )
    frequencies = pitch.selected_array["frequency"]
    times = pitch.x1 + np.arange(frequencies.size) * pitch.dx
    voiced = (frequencies > 0.0) & (times >= start_s) & (times <= end_s)
    reference = frequencies[voiced]
    if reference.size == 0:
        return np.empty(0), np.empty(0), 0.0
    median_f0 = float(np.median(reference)) if reference_f0_hz is None else reference_f0_hz
    context = (frequencies > 0.0) & (times >= start_s - 0.5) & (times <= end_s + 0.5)
    semitones = 12.0 * np.log2(frequencies[context] / median_f0)
    return times[context] - start_s, semitones, median_f0


def _plot_monotone_f0(
    original_audio: np.ndarray,
    flawed_audio: np.ndarray,
    label,
    output_path: Path,
    severity: float,
) -> float:
    """Plot original/flawed local F0 in semitones and return std ratio."""
    original_times, original_st, original_median_f0 = f0_semitone_series(
        original_audio,
        label.original_start_s,
        label.original_end_s,
    )
    flawed_times, flawed_st, _ = f0_semitone_series(
        flawed_audio,
        label.rendered_start_s,
        label.rendered_end_s,
        reference_f0_hz=original_median_f0,
    )
    duration_s = label.original_end_s - label.original_start_s
    original_region = original_st[(original_times >= 0) & (original_times <= duration_s)]
    flawed_region = flawed_st[(flawed_times >= 0) & (flawed_times <= duration_s)]
    if original_region.size == 0 or flawed_region.size == 0:
        raise ValueError("monotone comparison region contains no voiced F0 frames")
    original_std = float(np.std(original_region))
    flawed_std = float(np.std(flawed_region))
    ratio = flawed_std / original_std if original_std else 0.0

    figure, axis = plt.subplots(figsize=(12, 4))
    axis.axvspan(0.0, duration_s, color="tab:red", alpha=0.12, label="flaw region")
    axis.plot(original_times, original_st, color="tab:blue", linewidth=1.0, label="ideal F0")
    axis.plot(flawed_times, flawed_st, color="tab:orange", linewidth=1.0, label="monotone F0")
    axis.set_xlabel("Seconds relative to flaw start")
    axis.set_ylabel("F0 (semitones from ideal-region median)")
    axis.set_title(f"Monotone F0, severity {severity:.1f}; std ratio {ratio:.3f}")
    axis.legend(loc="upper right")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return ratio


FLAW_TYPES = (
    "pace_fast",
    "pace_slow",
    "long_pause",
    "monotone",
    "volume_dropoff",
    "filler",
    "stumble_repeat",
)
SEVERITIES = (0.3, 0.9)


def _choose_region(word_count: int) -> tuple[int, int]:
    """Choose a reproducible central region with preceding words for stumbles."""
    if word_count < 5:
        raise ValueError("demo clip needs at least five aligned words")
    start = max(1, word_count // 3)
    end = min(word_count, start + 4)
    return start, end


def _plot_result(
    waveform: np.ndarray,
    timings,
    labels,
    output_path: Path,
    title: str,
) -> None:
    """Plot waveform, word boundaries, and rendered ground-truth intervals."""
    times = np.arange(waveform.size, dtype=np.float32) / SAMPLE_RATE_HZ
    figure, axis = plt.subplots(figsize=(16, 5))
    axis.plot(times, waveform, color="black", linewidth=0.45)
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Amplitude")
    axis.set_title(title)

    upper_limit = max(float(np.max(np.abs(waveform))), 1e-3)
    for timing in timings:
        axis.axvline(timing.start_s, color="tab:blue", linewidth=0.55, alpha=0.42)
        axis.text(
            (timing.start_s + timing.end_s) / 2,
            upper_limit * 0.95,
            timing.word,
            rotation=90,
            ha="center",
            va="top",
            fontsize=6,
        )
    for label in labels:
        axis.axvspan(
            label.rendered_start_s,
            label.rendered_end_s,
            color="tab:red",
            alpha=0.25,
            label=f"{label.flaw_type} ground truth",
        )
    if labels:
        axis.legend(loc="upper right")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    """Align the fixture, generate each requested injection, and save artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio",
        type=Path,
        default=Path("data/raw/test_clip.wav"),
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=Path("data/raw/test_clip.txt"),
    )
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    ideal_audio = load_audio(arguments.audio)
    word_timings = align(
        ideal_audio,
        arguments.transcript.read_text(encoding="utf-8"),
    )
    region = _choose_region(len(word_timings))
    wav_root = Path("data/processed/demo")
    plot_root = Path("eval/results")
    wav_root.mkdir(parents=True, exist_ok=True)
    plot_root.mkdir(parents=True, exist_ok=True)

    for flaw_index, flaw_type in enumerate(FLAW_TYPES):
        for severity_index, severity in enumerate(SEVERITIES):
            spec = FlawSpec(flaw_type, region, severity)
            audio, timings, labels = apply_flaws(
                ideal_audio,
                word_timings,
                [spec],
                seed=arguments.seed + flaw_index * len(SEVERITIES) + severity_index,
            )
            stem = f"{flaw_type}_severity_{severity:.1f}"
            wav_path = wav_root / f"{stem}.wav"
            plot_path = plot_root / f"{stem}.png"
            sf.write(wav_path, audio, SAMPLE_RATE_HZ, subtype="PCM_24")
            _plot_result(
                audio,
                timings,
                labels,
                plot_path,
                f"{flaw_type}, severity {severity:.1f}",
            )
            print(f"{flaw_type} severity {severity:.1f}: {wav_path}, {plot_path}")
            if flaw_type == "monotone":
                f0_plot_path = plot_root / f"monotone_f0_severity_{severity:.1f}.png"
                ratio = _plot_monotone_f0(
                    ideal_audio,
                    audio,
                    labels[0],
                    f0_plot_path,
                    severity,
                )
                print(f"F0 standard-deviation ratio: {ratio:.4f}; plot: {f0_plot_path}")


if __name__ == "__main__":
    main()