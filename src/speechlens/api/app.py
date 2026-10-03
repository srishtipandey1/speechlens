"""FastAPI application for deterministic offline speech analysis and demo results."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from speechlens.api import service


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = PROJECT_ROOT / "data" / "processed" / "demo"
_CACHE_LIMIT = 32
_ANALYSIS_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_CACHE_LOCK = threading.Lock()
_RECORDING_ID = re.compile(r"^[A-Za-z0-9_.-]+$")

app = FastAPI(title="SpeechLens", version="0.1.0")


def _cache_key(audio: bytes, transcript: str, ideal: bytes | None) -> str:
    digest = hashlib.sha256()
    for content in (
        audio,
        transcript.encode("utf-8"),
        ideal or b"",
        (PROJECT_ROOT / "config" / "detection.yaml").read_bytes(),
        service.REFERENCE_PATH.read_bytes() if service.REFERENCE_PATH.is_file() else b"missing-reference",
        service.TEST_METRICS_PATH.read_bytes() if service.TEST_METRICS_PATH.is_file() else b"missing-test-metrics",
    ):
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


@app.get("/health")
def health() -> dict[str, Any]:
    """Report service health and local artifact availability without loading models."""
    return {
        "status": "ok",
        "alignment_model_cached": service.alignment_model_available(),
        "reference_artifact_available": service.REFERENCE_PATH.is_file(),
        "held_out_metrics_available": service.TEST_METRICS_PATH.is_file(),
        "demo_recordings": len(list(DEMO_DIR.glob("*.json"))) if DEMO_DIR.is_dir() else 0,
    }


@app.post("/analyze")
async def analyze(
    audio_file: UploadFile = File(...),
    transcript: str = Form(...),
    ideal_audio: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    """Align, detect, score, and explain one uploaded recording."""
    audio_bytes = await audio_file.read()
    ideal_bytes = await ideal_audio.read() if ideal_audio is not None else None
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="audio file is empty")
    if not transcript.strip():
        raise HTTPException(status_code=422, detail="transcript must not be empty")
    key = _cache_key(audio_bytes, transcript, ideal_bytes)
    with _CACHE_LOCK:
        cached = _ANALYSIS_CACHE.get(key)
        if cached is not None:
            _ANALYSIS_CACHE.move_to_end(key)
            return json.loads(json.dumps(cached, sort_keys=True, separators=(",", ":")))
    try:
        result = service.analyze_audio_bytes(
            audio_bytes,
            transcript,
            ideal_bytes,
            project_root=PROJECT_ROOT,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    with _CACHE_LOCK:
        _ANALYSIS_CACHE[key] = result
        _ANALYSIS_CACHE.move_to_end(key)
        while len(_ANALYSIS_CACHE) > _CACHE_LIMIT:
            _ANALYSIS_CACHE.popitem(last=False)
    return result


@app.get("/demo")
def demo_list() -> dict[str, list[str]]:
    """List locally precomputed demo recording IDs."""
    if not DEMO_DIR.is_dir():
        return {"recording_ids": []}
    return {
        "recording_ids": sorted(path.stem for path in DEMO_DIR.glob("*.json"))
    }


@app.get("/demo/{recording_id}")
def demo_recording(recording_id: str) -> dict[str, Any]:
    """Return a locally precomputed analysis result without loading a model."""
    if not _RECORDING_ID.fullmatch(recording_id):
        raise HTTPException(status_code=400, detail="invalid recording ID")
    path = DEMO_DIR / f"{recording_id}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="precomputed demo recording not found")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=500, detail="precomputed demo JSON is invalid") from error
