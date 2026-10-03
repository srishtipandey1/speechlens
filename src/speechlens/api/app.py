"""FastAPI application for deterministic offline speech analysis and demo results."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import csv
from collections import OrderedDict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from speechlens.api import service
from speechlens.detection.core import load_detection_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = PROJECT_ROOT / "data" / "processed" / "demo"
AUDIO_DIR = PROJECT_ROOT / "data" / "processed" / "audio"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
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


def _read_csv(name: str, missing: list[str]) -> list[dict[str, str]]:
    """Read one optional results artifact and record missing provenance."""
    path = RESULTS_DIR / name
    if not path.is_file():
        missing.append(name)
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _number(row: dict[str, str], key: str) -> float | None:
    """Return a finite numeric CSV value, or None when it is unavailable."""
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if value == value and abs(value) != float("inf") else None


@app.get("/dashboard-metrics")
def dashboard_metrics() -> dict[str, Any]:
    """Expose only metrics already recorded in evaluation artifacts."""
    missing: list[str] = []
    scoring_rows = _read_csv("scoring_metrics.csv", missing)
    detector_rows = _read_csv("test_detection_type_metrics.csv", missing)
    summary_rows = _read_csv("test_detection_summary.csv", missing)
    control_rows = _read_csv("test_detection_control_false_positives.csv", missing)

    scoring: dict[str, dict[str, dict[str, float | None]]] = {}
    for row in scoring_rows:
        split = row.get("split", "")
        mode = row.get("mode", "")
        metric = row.get("metric", "")
        if split and mode and metric:
            scoring.setdefault(split, {}).setdefault(mode, {})[metric] = _number(row, "value")

    reliability: dict[str, dict[str, dict[str, float | None]]] = {}
    for row in detector_rows:
        if row.get("timing_source") != "realistic" or row.get("iou_threshold") != "0.3":
            continue
        mode = row.get("mode", "")
        flaw_type = row.get("flaw_type", "")
        if mode and flaw_type:
            reliability.setdefault(mode, {})[flaw_type] = {
                "precision": _number(row, "precision"),
                "recall": _number(row, "recall"),
            }

    performance: dict[str, dict[str, float | None]] = {}
    for row in summary_rows:
        if row.get("timing_source") == "realistic" and row.get("mode"):
            performance[row["mode"]] = {
                "false_regions_per_minute": _number(
                    row, "false_regions_per_minute_all_controls_and_ideals"
                ),
                "boundary_error_ms": _number(row, "mean_boundary_error_ms"),
            }

    control_false_regions: dict[str, dict[str, float | None]] = {}
    for row in control_rows:
        if row.get("timing_source") == "realistic" and row.get("mode") and row.get("control_type"):
            control_false_regions.setdefault(row["mode"], {})[row["control_type"]] = {
                "false_regions_per_minute": _number(row, "false_regions_per_minute"),
                "false_regions": _number(row, "false_regions"),
                "minutes": _number(row, "minutes"),
            }

    config = load_detection_config(PROJECT_ROOT / "config" / "detection.yaml")
    detector_modes: dict[str, dict[str, Any]] = {}
    for mode in ("paired", "reference_free"):
        enabled: list[str] = []
        disabled: list[dict[str, str]] = []
        for flaw_type, detector in config["detectors"].items():
            is_enabled = detector.get("enabled_modes", {}).get(
                mode, detector.get("enabled", True)
            )
            if is_enabled:
                enabled.append(flaw_type)
            else:
                disabled.append({
                    "type": flaw_type,
                    "reason": detector.get("disabled_reasons_by_mode", {}).get(
                        mode, detector.get("disabled_reason", "disabled by frozen config")
                    ),
                })
        detector_modes[mode] = {"enabled": enabled, "disabled": disabled}

    weak_paired_detectors = [
        flaw_type
        for flaw_type, values in reliability.get("paired", {}).items()
        if values.get("precision") is not None
        and values["precision"] < 0.4
        and flaw_type in detector_modes["paired"]["enabled"]
    ]
    return {
        "available": not missing,
        "missing_artifacts": missing,
        "scoring": scoring,
        "detector_reliability": reliability,
        "test_performance": performance,
        "control_false_regions": control_false_regions,
        "detectors_by_mode": detector_modes,
        "weak_paired_detectors": weak_paired_detectors,
        "false_region_budget_per_minute": _number(
            config.get("evaluation", {}), "false_region_budget_per_minute"
        ),
        "sources": [
            "scoring_metrics.csv",
            "test_detection_type_metrics.csv",
            "test_detection_summary.csv",
            "test_detection_control_false_positives.csv",
            "config/detection.yaml",
        ],
    }


@app.get("/audio/{recording_id}")
def demo_audio(recording_id: str) -> FileResponse:
    """Stream a FLAC only when its ID is present in the precomputed demo list."""
    if (
        not _RECORDING_ID.fullmatch(recording_id)
        or recording_id not in demo_list()["recording_ids"]
    ):
        raise HTTPException(
            status_code=404,
            detail={"message": "demo recording is not available"},
        )
    audio_root = AUDIO_DIR.resolve()
    audio_path = (AUDIO_DIR / f"{recording_id}.flac").resolve()
    if audio_path.parent != audio_root or not audio_path.is_file():
        raise HTTPException(
            status_code=404,
            detail={"message": "demo audio is missing for this recording"},
        )
    return FileResponse(
        audio_path,
        media_type="audio/flac",
        filename=f"{recording_id}.flac",
    )


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
