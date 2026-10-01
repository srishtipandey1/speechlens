"""Validate dataset manifests, audio/labels, and deterministic spot-check plots."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import yaml
from pydantic import BaseModel, ConfigDict, FiniteFloat
from typing import Literal

from speechlens.schema import Recording, RecordingCollection
from speechlens.injection.engine import _quietest_room_tone


MANIFEST_FIELDS = (
    "recording_id",
    "passage_id",
    "speaker",
    "gender",
    "split",
    "kind",
    "severity_level",
    "number_of_flaws",
    "flaw_types",
    "path",
    "duration_s",
    "source",
    "license",
)


class ManifestRow(BaseModel):
    """Validated schema for one manifest CSV row."""

    model_config = ConfigDict(extra="forbid")

    recording_id: str
    passage_id: str
    speaker: str
    gender: Literal["M", "F"]
    split: Literal["dev", "test"]
    kind: Literal["ideal", "injected", "control"]
    severity_level: int
    number_of_flaws: int
    flaw_types: str
    path: str
    duration_s: FiniteFloat
    source: str
    license: str


def _load_config(project_root: Path) -> dict:
    """Load the seeded spot-check selection count."""
    with (project_root / "config" / "dataset.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _check_word_mapping(sidecar: dict, flaws: list[dict], errors: list[str]) -> None:
    """Check original intervals map to their declared source word timings."""
    source_timings = sidecar.get("original_word_timings", [])
    sample_rate = int(sidecar["recording"]["sample_rate"])
    tolerance_s = 1 / sample_rate
    recording_id = sidecar["recording"]["id"]
    for flaw_index, flaw in enumerate(flaws):
        word_indices = flaw.get("word_indices", [])
        if not word_indices or max(word_indices) >= len(source_timings):
            errors.append(f"{recording_id}: flaw {flaw_index} has invalid source word indices")
            continue
        first = source_timings[min(word_indices)]
        last = source_timings[max(word_indices)]
        start_s = float(flaw["original_start_s"])
        end_s = float(flaw["original_end_s"])
        if flaw["flaw_type"] in {"long_pause", "filler"}:
            boundary_s = float(first["end_s"])
            if (
                abs(start_s - boundary_s) > tolerance_s
                or end_s - start_s > tolerance_s + 1e-12
            ):
                errors.append(f"{recording_id}: inserted flaw {flaw_index} misses its source word boundary")
        elif (
            abs(start_s - float(first["start_s"])) > tolerance_s
            or abs(end_s - float(last["end_s"])) > tolerance_s
        ):
            errors.append(f"{recording_id}: flaw {flaw_index} does not cover its declared source words")


def _check_injected_recording(
    sidecar: dict,
    audio_samples: int,
    audio: np.ndarray,
    ideal_audio: np.ndarray | None,
    errors: list[str],
) -> None:
    """Validate flaw spans, non-overlap, and original word references."""
    recording = sidecar["recording"]
    recording_id = recording["id"]
    pair = sidecar.get("pair")
    if not pair:
        errors.append(f"{recording_id}: injected recording has no Pair")
        return
    flaws = pair.get("flaws", [])
    if not flaws:
        errors.append(f"{recording_id}: injected recording has no FlawLabels")
        return
    sample_rate = int(recording["sample_rate"])
    audio_duration_s = audio_samples / sample_rate
    previous_end_s = -1.0
    for flaw_index, flaw in enumerate(flaws):
        start_s = float(flaw["rendered_start_s"])
        end_s = float(flaw["rendered_end_s"])
        if not 0 <= start_s < end_s <= audio_duration_s:
            errors.append(f"{recording_id}: flaw {flaw_index} rendered interval is outside audio")
        if start_s < previous_end_s:
            errors.append(f"{recording_id}: rendered flaw intervals overlap")
        previous_end_s = end_s
    _check_word_mapping(sidecar, flaws, errors)
    if ideal_audio is None:
        errors.append(f"{recording_id}: ideal parent audio is unavailable for pause validation")
        return
    config_path = Path(__file__).resolve().parents[1] / "config" / "injection.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        pause_config = yaml.safe_load(config_file)["long_pause"]
    room_tone_samples = max(1, round(float(pause_config["room_tone_window_s"]) * sample_rate))
    reference = _quietest_room_tone(ideal_audio, room_tone_samples).astype(np.float64)
    reference_rms = float(np.sqrt(np.mean(np.square(reference))))
    for flaw_index, flaw in enumerate(flaws):
        if flaw["flaw_type"] != "long_pause":
            continue
        start_sample = round(float(flaw["rendered_start_s"]) * sample_rate)
        end_sample = round(float(flaw["rendered_end_s"]) * sample_rate)
        pause = audio[start_sample:end_sample].astype(np.float64)
        pause_rms = float(np.sqrt(np.mean(np.square(pause))))
        if pause_rms <= 0.0 or reference_rms <= 0.0:
            errors.append(f"{recording_id}: pause {flaw_index} has zero room-tone RMS")
        elif abs(20.0 * np.log10(pause_rms / reference_rms)) > 3.0:
            errors.append(f"{recording_id}: pause {flaw_index} room-tone RMS differs from parent ideal by more than 3 dB")


def _save_spot_plot(
    project_root: Path,
    row: dict[str, str],
    sidecar: dict,
    audio: np.ndarray,
    sample_rate: int,
) -> Path:
    """Save a waveform plot with the rendered flaw intervals shaded."""
    times = np.arange(audio.size, dtype=np.float32) / sample_rate
    figure, axis = plt.subplots(figsize=(15, 4))
    axis.plot(times, audio, color="black", linewidth=0.4)
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Amplitude")
    axis.set_title(row["recording_id"])
    for flaw in (sidecar.get("pair") or {}).get("flaws", []):
        axis.axvspan(
            float(flaw["rendered_start_s"]),
            float(flaw["rendered_end_s"]),
            color="tab:red",
            alpha=0.25,
            label=flaw["flaw_type"],
        )
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels)
    output_path = project_root / "eval" / "results" / f"spot_{row['recording_id']}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
    return output_path


def validate_dataset(project_root: Path, create_spot_checks: bool = False) -> dict:
    """Validate all manifest entries, sidecars, audio, and cross-recording parents."""
    manifest_path = project_root / "data" / "labels" / "manifest.csv"
    errors: list[str] = []
    if not manifest_path.is_file():
        return {"valid": False, "checked_recordings": 0, "errors": ["manifest.csv is missing"]}

    with manifest_path.open(encoding="utf-8", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            errors.append("manifest columns do not match required fields")
        rows = list(reader)

    recordings: list[Recording] = []
    loaded_sidecars: dict[str, dict] = {}
    loaded_audio: dict[str, np.ndarray] = {}
    audio_by_recording_id: dict[str, np.ndarray] = {}
    listed_audio_paths: set[str] = set()
    seen_ids: set[str] = set()
    for row in rows:
        try:
            ManifestRow.model_validate(row)
        except ValueError as error:
            errors.append(f"manifest row is invalid: {error}")
        recording_id = row.get("recording_id", "")
        if recording_id in seen_ids:
            errors.append(f"{recording_id}: duplicate manifest recording ID")
        seen_ids.add(recording_id)
        relative_audio_path = row.get("path", "")
        listed_audio_paths.add(relative_audio_path)
        audio_path = project_root / Path(relative_audio_path)
        sidecar_path = project_root / "data" / "labels" / f"{recording_id}.json"
        if not audio_path.is_file():
            errors.append(f"{recording_id}: manifest audio file is missing")
            continue
        if not sidecar_path.is_file():
            errors.append(f"{recording_id}: label sidecar is missing")
            continue
        try:
            audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            recording = Recording.model_validate(sidecar["recording"])
        except (OSError, ValueError, KeyError) as error:
            errors.append(f"{recording_id}: could not parse audio/metadata: {error}")
            continue
        if recording.id != recording_id or recording.audio_path != relative_audio_path:
            errors.append(f"{recording_id}: manifest path/ID disagrees with sidecar")
        if recording.kind != row.get("kind") or recording.severity_level != int(row["severity_level"]):
            errors.append(f"{recording_id}: manifest kind/severity disagrees with sidecar")
        if recording.speaker_id != row.get("speaker") or recording.speaker_gender != row.get("gender"):
            errors.append(f"{recording_id}: manifest speaker/gender disagrees with sidecar")
        if sidecar.get("passage_id") != row.get("passage_id") or sidecar.get("split") != row.get("split"):
            errors.append(f"{recording_id}: manifest passage/split disagrees with sidecar")
        if sample_rate != recording.sample_rate or audio.ndim != 1:
            errors.append(f"{recording_id}: audio is not mono at its declared sample rate")
        if abs(audio.size / sample_rate - recording.duration_s) > 1 / sample_rate:
            errors.append(f"{recording_id}: declared duration differs from file duration")
        if abs(audio.size / sample_rate - float(row["duration_s"])) > 1 / sample_rate:
            errors.append(f"{recording_id}: manifest duration differs from file duration")
        pair_flaws = (sidecar.get("pair") or {}).get("flaws", [])
        actual_flaw_types = ";".join(flaw["flaw_type"] for flaw in pair_flaws)
        if len(pair_flaws) != int(row["number_of_flaws"]):
            errors.append(f"{recording_id}: manifest flaw count disagrees with sidecar")
        if actual_flaw_types != row["flaw_types"]:
            errors.append(f"{recording_id}: manifest flaw types disagree with sidecar")
        if not np.isfinite(audio).all():
            errors.append(f"{recording_id}: audio contains NaN or infinite values")
        if np.any(np.abs(audio) >= 1.0):
            errors.append(f"{recording_id}: audio contains clipping")
        if recording.kind == "injected":
            parent_id = recording.parent_recording_id or ""
            parent_audio = audio_by_recording_id.get(parent_id)
            if parent_audio is None:
                parent_path = (
                    project_root
                    / "data"
                    / "processed"
                    / "audio"
                    / f"{parent_id}.flac"
                )
                if parent_path.is_file():
                    parent_audio, _ = sf.read(parent_path, dtype="float32", always_2d=False)
            _check_injected_recording(sidecar, audio.size, audio, parent_audio, errors)
        loaded_sidecars[recording_id] = sidecar
        loaded_audio[recording_id] = audio
        audio_by_recording_id[recording_id] = audio
        recordings.append(recording)

    actual_audio_paths = {
        path.relative_to(project_root).as_posix()
        for path in (project_root / "data" / "processed" / "audio").glob("*.flac")
    }
    if actual_audio_paths != listed_audio_paths:
        errors.append("manifest and processed/audio file sets differ")
    try:
        RecordingCollection(recordings=recordings)
    except ValueError as error:
        errors.append(f"recording parent references are invalid: {error}")

    spot_plots: list[str] = []
    if create_spot_checks and rows:
        config = _load_config(project_root)
        injected_rows = sorted(
            (
                row
                for row in rows
                if row["kind"] == "injected" and row["recording_id"] in loaded_audio
            ),
            key=lambda row: row["recording_id"],
        )
        if injected_rows:
            rng = np.random.default_rng(int(config["seed"]))
            count = min(int(config["validation"]["spot_check_count"]), len(injected_rows))
            indices = sorted(rng.choice(len(injected_rows), size=count, replace=False).tolist())
            for index in indices:
                row = injected_rows[index]
                rate = int(loaded_sidecars[row["recording_id"]]["recording"]["sample_rate"])
                plot = _save_spot_plot(
                    project_root,
                    row,
                    loaded_sidecars[row["recording_id"]],
                    loaded_audio[row["recording_id"]],
                    rate,
                )
                spot_plots.append(plot.relative_to(project_root).as_posix())

    return {
        "valid": not errors,
        "checked_recordings": len(recordings),
        "errors": errors,
        "spot_check_plots": spot_plots,
    }


def main() -> None:
    """Run validation and report the result."""
    project_root = Path(__file__).resolve().parents[1]
    result = validate_dataset(project_root, create_spot_checks=True)
    for error in result["errors"]:
        print(f"ERROR: {error}", file=sys.stderr)
    print(f"Validated {result['checked_recordings']} recordings: {'PASS' if result['valid'] else 'FAIL'}")
    for path in result["spot_check_plots"]:
        print(f"Spot check: {path}")
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()