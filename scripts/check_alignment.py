"""Render an aligned waveform for manual timing inspection."""

import argparse
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from speechlens.alignment.align import SAMPLE_RATE_HZ, align, load_audio


def main() -> None:
    """Align an input clip and save a waveform/timing visualization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("transcript", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("eval/results/alignment_check.png"),
    )
    arguments = parser.parse_args()

    try:
        waveform = load_audio(arguments.audio)
        timings = align(waveform, arguments.transcript.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    times = np.arange(waveform.size) / SAMPLE_RATE_HZ
    figure, axis = plt.subplots(figsize=(16, 5))
    axis.plot(times, waveform, linewidth=0.5, color="black")
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Amplitude")
    axis.set_title("MMS_FA word alignment")

    upper_limit = float(np.max(np.abs(waveform))) or 1.0
    for timing in timings:
        axis.axvline(timing.start_s, color="tab:red", linewidth=0.7, alpha=0.7)
        axis.axvline(timing.end_s, color="tab:red", linewidth=0.7, alpha=0.7)
        axis.text(
            (timing.start_s + timing.end_s) / 2,
            upper_limit * 0.9,
            timing.word,
            rotation=90,
            ha="center",
            va="top",
            fontsize=7,
        )

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(arguments.out, dpi=160)
    plt.close(figure)
    print(f"Saved alignment plot to {arguments.out}")
    confidences = [timing.confidence for timing in timings if timing.confidence is not None]
    print(f"Aligned words: {len(timings)}")
    if confidences:
        print(f"Median word confidence: {statistics.median(confidences):.4f}")
        least_confident = sorted(
            (timing for timing in timings if timing.confidence is not None),
            key=lambda timing: timing.confidence,
        )[:5]
        print(
            "Lowest-confidence words: "
            + ", ".join(
                f"{timing.word} ({timing.confidence:.4f})" for timing in least_confident
            )
        )


if __name__ == "__main__":
    main()