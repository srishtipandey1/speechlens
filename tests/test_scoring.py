"""Deterministic rubric tests on small synthetic acoustic signals."""

import json

import numpy as np

from speechlens.detection.core import load_detection_config
from speechlens.features import FeatureBundle, extract_recording_features
from speechlens.schema import WordTiming
from speechlens.scoring import SCORING_WEIGHTS, score_recording


def _tone_bundle() -> tuple[FeatureBundle, list[WordTiming], str]:
    sample_rate_hz = 16000
    duration_s = 0.6
    times = np.arange(round(sample_rate_hz * duration_s), dtype=np.float64) / sample_rate_hz
    audio = (0.2 * np.sin(2.0 * np.pi * 180.0 * times)).astype(np.float32)
    transcript = "tone one two"
    timings = [
        WordTiming(word="tone", start_s=0.0, end_s=0.2),
        WordTiming(word="one", start_s=0.2, end_s=0.4),
        WordTiming(word="two", start_s=0.4, end_s=0.6),
    ]
    return extract_recording_features(audio, timings, transcript), timings, transcript


def test_paired_identical_audio_scores_are_byte_deterministic() -> None:
    bundle, timings, transcript = _tone_bundle()
    config = load_detection_config()

    first = score_recording(
        bundle, timings, transcript, "paired", [], config,
        baseline_bundle=bundle, baseline_timings=timings,
    )
    second = score_recording(
        bundle, timings, transcript, "paired", [], config,
        baseline_bundle=bundle, baseline_timings=timings,
    )

    first_bytes = json.dumps(first, sort_keys=True, separators=(",", ":")).encode()
    second_bytes = json.dumps(second, sort_keys=True, separators=(",", ":")).encode()
    assert first_bytes == second_bytes
    assert first["total"] == 100.0
    assert set(first["scores"]) == set(SCORING_WEIGHTS)
    assert all(0.0 <= value <= 100.0 for value in first["scores"].values())


def test_reference_free_scores_use_robust_dev_statistics() -> None:
    bundle, timings, transcript = _tone_bundle()
    reference = {
        "fit_source": "synthetic DEV IDEAL fixture",
        "position_statistics": {
            position: {
                "duration_ratio": {"median": 1.0, "mad": 0.1},
                "f0_std": {"median": 0.1, "mad": 0.1},
                "pause_s": {"median": 0.0, "mad": 0.05},
                "volume_drop_db": {"median": 0.0, "mad": 1.0},
            }
            for position in ("first", "middle", "last")
        },
    }
    score = score_recording(
        bundle,
        timings,
        transcript,
        "reference_free",
        [],
        load_detection_config(),
        reference=reference,
    )

    assert score["mode"] == "reference_free"
    assert score["dimension_reliability"]["articulation"].startswith("limited")
    assert score["features"]["reference_fit_source"] == "synthetic DEV IDEAL fixture"
    assert 0.0 <= score["total"] <= 100.0
