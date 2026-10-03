"""Synthetic smoke coverage for analysis and local demo API routes."""

import asyncio
import io
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest
import soundfile as sf
from starlette.datastructures import UploadFile

from speechlens.api import app as api
from speechlens.api import service
from speechlens.schema import WordTiming


async def _get(path: str) -> tuple[int, dict[bytes, bytes], bytes]:
    """Issue a dependency-free in-process ASGI GET request."""
    messages = []
    request_sent = False

    async def receive():
        nonlocal request_sent
        if request_sent:
            return {"type": "http.disconnect"}
        request_sent = True
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    raw_path = path.encode("ascii")
    await api.app(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": path,
            "raw_path": raw_path,
            "query_string": b"",
            "root_path": "",
            "headers": [(b"host", b"testserver")],
            "client": ("testclient", 12345),
            "server": ("testserver", 80),
        },
        receive,
        send,
    )
    response_start = next(message for message in messages if message["type"] == "http.response.start")
    body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    return response_start["status"], dict(response_start["headers"]), body


@pytest.fixture
def demo_store(monkeypatch):
    with tempfile.TemporaryDirectory(prefix="speechlens-route-demo-", dir=".") as directory:
        root = Path(directory)
        demo_dir = root / "demo"
        audio_dir = root / "audio"
        demo_dir.mkdir()
        audio_dir.mkdir()
        (demo_dir / "fixture.json").write_text('{"mode":"paired"}\n', encoding="utf-8")
        (audio_dir / "fixture.flac").write_bytes(b"synthetic-flac-fixture")
        monkeypatch.setattr(api, "DEMO_DIR", demo_dir)
        monkeypatch.setattr(api, "AUDIO_DIR", audio_dir)
        yield demo_dir, audio_dir


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


def test_demo_audio_route_requires_listed_recording_and_streams_file(demo_store) -> None:
    _demo_dir, audio_dir = demo_store

    status, headers, body = asyncio.run(_get("/audio/fixture"))
    assert status == 200
    assert headers[b"content-type"] == b"audio/flac"
    assert body == (audio_dir / "fixture.flac").read_bytes()

    status, _headers, body = asyncio.run(_get("/audio/not-listed"))
    assert status == 404
    assert b"not available" in body

    (audio_dir / "fixture.flac").unlink()
    status, _headers, body = asyncio.run(_get("/audio/fixture"))
    assert status == 404
    assert b"audio is missing" in body


def test_dashboard_metrics_reports_absent_artifacts(monkeypatch) -> None:
    with tempfile.TemporaryDirectory(prefix="speechlens-route-metrics-", dir=".") as directory:
        monkeypatch.setattr(api, "RESULTS_DIR", Path(directory))

        response = api.dashboard_metrics()

        assert response["available"] is False
        assert "scoring_metrics.csv" in response["missing_artifacts"]
        assert response["scoring"] == {}


def test_dashboard_root_and_assets_are_served() -> None:
    status, headers, body = asyncio.run(_get("/"))
    assert status == 200
    assert headers[b"content-type"].startswith(b"text/html")
    assert b"SpeechLens" in body

    asset_status, _asset_headers, app_body = asyncio.run(_get("/app.js"))
    assert asset_status == 200
    assert b"loadDemoIndex" in app_body
