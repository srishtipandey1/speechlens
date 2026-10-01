"""Generate WAV and waveform-label demos for every flaw injection type."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from speechlens.alignment.align import SAMPLE_RATE_HZ, align, load_audio
from speechlens.injection import FlawSpec, apply_flaws


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
            sf.write(wav_path, audio, SAMPLE_RATE_HZ, subtype="PCM_16")
            _plot_result(
                audio,
                timings,
                labels,
                plot_path,
                f"{flaw_type}, severity {severity:.1f}",
            )
            print(f"{flaw_type} severity {severity:.1f}: {wav_path}, {plot_path}")


if __name__ == "__main__":
    main()