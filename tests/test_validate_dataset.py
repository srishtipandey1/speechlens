"""Manifest and validator tests using a tiny synthetic dataset."""

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError

from scripts.make_dataset import MANIFEST_FIELDS
from scripts.validate_dataset import ManifestRow, validate_dataset
from speechlens.schema import FlawLabel, Pair, Recording, WordTiming


def manifest_row(
    recording_id: str,
    kind: str,
    severity: int,
    path: str,
    flaw_types: str,
) -> dict[str, str | int | float]:
    """Create a valid manifest test row."""
    return {
        "recording_id": recording_id,
        "passage_id": "p1",
        "speaker": "speaker-1",
        "gender": "M",
        "split": "dev",
        "kind": kind,
        "severity_level": severity,
        "number_of_flaws": 1 if flaw_types else 0,
        "flaw_types": flaw_types,
        "path": path,
        "duration_s": "1.000000",
        "source": "synthetic test fixture",
        "license": "test-only",
    }


def write_sidecar(
    root: Path,
    recording: Recording,
    pair: Pair | None,
    timings: list[WordTiming],
) -> None:
    """Write the sidecar structure consumed by the validator."""
    sidecar = {
        "recording": recording.model_dump(mode="json"),
        "passage_id": "p1",
        "split": "dev",
        "gender": "M",
        "pair": None if pair is None else pair.model_dump(mode="json"),
        "word_timings": [timing.model_dump(mode="json") for timing in timings],
        "original_word_timings": [timing.model_dump(mode="json") for timing in timings],
        "alignment_quality": None,
        "control_transform": None,
    }
    (root / "data" / "labels" / f"{recording.id}.json").write_text(
        json.dumps(sidecar),
        encoding="utf-8",
    )


def test_manifest_row_schema_accepts_valid_and_rejects_invalid_gender() -> None:
    """Manifest values are validated against the declared column schema."""
    row = manifest_row("p1__ideal", "ideal", 0, "data/processed/audio/p1.flac", "")
    assert ManifestRow.model_validate(row).recording_id == "p1__ideal"
    with pytest.raises(ValidationError):
        ManifestRow.model_validate({**row, "gender": "unknown"})


def test_validator_accepts_a_tiny_synthetic_dataset() -> None:
    """Validate ideal/injected files, labels, and source word mapping."""
    with tempfile.TemporaryDirectory(prefix="speechlens-validator-", dir=Path.cwd()) as directory:
        root = Path(directory)
        audio_root = root / "data" / "processed" / "audio"
        labels_root = root / "data" / "labels"
        audio_root.mkdir(parents=True)
        labels_root.mkdir(parents=True)
        samples = (0.2 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)).astype(np.float32)
        timings = [
            WordTiming(word="hello", start_s=0.1, end_s=0.3, confidence=0.9),
            WordTiming(word="world", start_s=0.4, end_s=0.6, confidence=0.9),
        ]

        records: list[tuple[Recording, Pair | None, str]] = []
        ideal_path = "data/processed/audio/p1__ideal.flac"
        ideal = Recording(
            id="p1__ideal",
            transcript_id="p1",
            speaker_id="speaker-1",
            speaker_gender="M",
            kind="ideal",
            severity_level=0,
            audio_path=ideal_path,
            sample_rate=16000,
            duration_s=1.0,
            source="synthetic test fixture",
            license="test-only",
        )
        ideal_audio_path = root / ideal_path
        sf.write(ideal_audio_path, samples, 16000, format="FLAC", subtype="PCM_16")
        records.append((ideal, None, ""))

        injected_path = "data/processed/audio/p1__L1.flac"
        injected = Recording(
            id="p1__L1",
            transcript_id="p1",
            speaker_id="speaker-1",
            speaker_gender="M",
            kind="injected",
            severity_level=1,
            audio_path=injected_path,
            sample_rate=16000,
            duration_s=1.0,
            source="synthetic test fixture",
            license="test-only",
            parent_recording_id="p1__ideal",
        )
        sf.write(root / injected_path, samples, 16000, format="FLAC", subtype="PCM_16")
        label = FlawLabel(
            flaw_type="volume_dropoff",
            severity=0.2,
            original_start_s=0.1,
            original_end_s=0.3,
            rendered_start_s=0.1,
            rendered_end_s=0.3,
            word_indices=[0],
        )
        pair = Pair(
            ideal_recording_id="p1__ideal",
            flawed_recording_id="p1__L1",
            flaws=[label],
        )
        records.append((injected, pair, "volume_dropoff"))

        rows = []
        for recording, pair_value, types in records:
            write_sidecar(root, recording, pair_value, timings)
            rows.append(
                manifest_row(
                    recording.id,
                    recording.kind,
                    recording.severity_level,
                    recording.audio_path,
                    types,
                )
            )
        with (labels_root / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

        result = validate_dataset(root, create_spot_checks=False)

    assert result["valid"]
    assert result["checked_recordings"] == 2
    assert result["errors"] == []