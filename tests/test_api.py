"""Synthetic smoke coverage for analysis and local demo API routes."""

import asyncio
import io
import json
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf
from starlette.datastructures import UploadFile

from speechlens.api import app as api
from speechlens.api import service
from speechlens.schema import WordTiming


def _wav_bytes() -> bytes:
    sample_rate_hz = 16000
    times = np.arange(sample_rate_hz, dtype=np.float64) / sample_rate_hz
    audio = (0.15 * np.sin(2.0 * np.pi * 180.0 * times)).astype(np.float32)
    output = io.BytesIO()
    sf.write(output, audio, sample_rate_hz, format="WAV")
    return output.getvalue()


def test_analyze_endpoint_smoke_is_deterministic_with_synthetic_audio(monkeypatch) -> None:
    audio_bytes = _wav_bytes()
    test_metrics = {
        mode: {
            "pace_fast": {"precision": 0.8, "recall": 0.5},
            "pace_slow": {"precision": 0.3, "recall": 0.4},
            "long_pause": {"precision": 1.0, "recall": 0.7},
            "monotone": {"precision": 0.3, "recall": 0.4},
            "volume_dropoff": {"precision": 0.2, "recall": 0.2},
            "filler": {"precision": 0.9, "recall": 0.6},
            "stumble_repeat": {"precision": 0.0, "recall": 0.0},
        }
        for mode in ("paired", "reference_free")
    }
    monkeypatch.setattr(
        service,
        "align",
        lambda _audio, _text: [WordTiming(word="tone", start_s=0.0, end_s=1.0)],
    )
    monkeypatch.setattr(
        service,
        "load_test_detection_metrics",
        lambda _root: test_metrics,
    )
    api._ANALYSIS_CACHE.clear()
    upload = lambda: UploadFile(file=io.BytesIO(audio_bytes), filename="tone.wav")

    first = asyncio.run(api.analyze(upload(), "tone", upload()))
    second = asyncio.run(api.analyze(upload(), "tone", upload()))

    first_bytes = json.dumps(first, sort_keys=True, separators=(",", ":")).encode()
    second_bytes = json.dumps(second, sort_keys=True, separators=(",", ":")).encode()
    assert first_bytes == second_bytes
    assert first["mode"] == "paired"
    assert set(first["scores"]) == {
        "pacing", "pausing_fluency", "intonation", "energy", "articulation", "disfluency"
    }
    assert first["series"]["participant"]["f0_semitones"]
    assert first["series"]["baseline"]["energy_db"]
    assert set(route.path for route in api.app.routes) >= {
        "/health", "/analyze", "/demo", "/demo/{recording_id}"
    }


def test_health_and_precomputed_demo_endpoints(monkeypatch) -> None:
    with tempfile.TemporaryDirectory(prefix="speechlens-api-demo-", dir=".") as directory:
        demo_dir = Path(directory)
        monkeypatch.setattr(api, "DEMO_DIR", demo_dir)
        health = api.health()
        assert health["status"] == "ok"
        assert api.demo_list() == {"recording_ids": []}

        (demo_dir / "fixture.json").write_text(
            '{"mode":"paired"}\n', encoding="utf-8"
        )

        assert api.demo_list() == {"recording_ids": ["fixture"]}
        assert api.demo_recording("fixture") == {"mode": "paired"}
