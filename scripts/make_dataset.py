"""Build deterministic ideal, flaw-ladder, and control recording datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
import yaml

from scripts.build_passages import Passage, load_dataset_config, select_passages
from scripts.validate_dataset import validate_dataset
from speechlens.alignment.align import SAMPLE_RATE_HZ, align
from speechlens.injection import FlawSpec, apply_flaws, identity_resynthesis
from speechlens.schema import Pair, Recording, WordTiming


FLAW_TYPES = (
    "pace_fast",
    "pace_slow",
    "long_pause",
    "monotone",
    "volume_dropoff",
    "filler",
    "stumble_repeat",
)
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
SOURCE_NAME = "LibriSpeech ASR corpus, dev-clean"
SOURCE_LICENSE = "CC BY 4.0"


def load_config(project_root: Path) -> dict:
    """Read global dataset generation settings."""
    with (project_root / "config" / "dataset.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    with (project_root / "config" / "injection.yaml").open(encoding="utf-8") as stream:
        config["injection"] = yaml.safe_load(stream)
    return config


def _candidate_regions(
    flaw_type: str,
    timings: Sequence[WordTiming],
    join_after_word_indices: set[int],
    config: dict,
) -> list[tuple[int, int]]:
    """Enumerate valid word regions for a flaw type in transcript order."""
    word_count = len(timings)
    width = 1 if flaw_type in {"long_pause", "filler", "stumble_repeat"} else int(
        config["ladder"]["region_word_count"]
    )
    maximum_gap = float(config["injection"]["long_pause"]["maximum_natural_gap_s"])
    candidates: list[tuple[int, int]] = []
    for start_index in range(0, word_count - width + 1):
        end_index = start_index + width
        if flaw_type == "stumble_repeat" and start_index < int(
            config["injection"]["stumble_repeat"]["high_previous_words"]
        ):
            continue
        if flaw_type in {"long_pause", "filler"}:
            if start_index in join_after_word_indices or start_index + 1 >= word_count:
                continue
            natural_gap = timings[start_index + 1].start_s - timings[start_index].end_s
            if not 0 <= natural_gap < maximum_gap:
                continue
        candidates.append((start_index, end_index))
    return candidates


def generate_ladder(
    word_timings: Sequence[WordTiming],
    join_after_word_indices: Sequence[int],
    config: dict,
    seed: int,
) -> list[dict]:
    """Build balanced, seeded flaw types and separated regions for every level."""
    levels = config["ladder"]["levels"]
    total_flaws = sum(int(level["flaw_count"]) for level in levels)
    rng = np.random.default_rng(seed)
    type_cycle = list(FLAW_TYPES)
    rng.shuffle(type_cycle)
    type_sequence = [type_cycle[index % len(type_cycle)] for index in range(total_flaws)]
    rng.shuffle(type_sequence)
    join_indices = set(join_after_word_indices)
    separation_words = int(config["ladder"]["minimum_region_separation_words"])

    generated: list[dict] = []
    type_cursor = 0
    for level in levels:
        used_regions: list[tuple[int, int]] = []
        specifications: list[FlawSpec] = []
        for _ in range(int(level["flaw_count"])):
            flaw_type = type_sequence[type_cursor]
            type_cursor += 1
            candidates = _candidate_regions(flaw_type, word_timings, join_indices, config)
            previous_word_count = (
                int(config["injection"]["stumble_repeat"]["high_previous_words"])
                if flaw_type == "stumble_repeat"
                else 0
            )
            candidate_footprints = {
                region: (max(0, region[0] - previous_word_count), region[1])
                for region in candidates
            }
            available = [
                region
                for region in candidates
                if all(
                    candidate_footprints[region][0] >= previous[1] + separation_words
                    or previous[0] >= candidate_footprints[region][1] + separation_words
                    for previous in used_regions
                )
            ]
            if not available:
                raise ValueError(
                    f"not enough separated candidate regions for {flaw_type} "
                    f"at severity level {level['severity_level']}"
                )
            chosen = available[int(rng.integers(0, len(available)))]
            used_regions.append(candidate_footprints[chosen])
            specifications.append(
                FlawSpec(flaw_type, chosen, float(level["severity"]))
            )
        generated.append(
            {
                "severity_level": int(level["severity_level"]),
                "flaws": specifications,
            }
        )
    return generated


def _json_write(path: Path, value: object) -> None:
    """Write stable UTF-8 JSON with sorted keys and a final newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _recording_id(passage_id: str, suffix: str) -> str:
    """Combine passage and transform identifiers into a stable recording ID."""
    return f"{passage_id}__{suffix}"


def _relative_path(project_root: Path, path: Path) -> str:
    """Serialize workspace paths with POSIX separators for the manifest."""
    return path.relative_to(project_root).as_posix()


def _safe_audio(audio: np.ndarray, peak_ceiling: float) -> np.ndarray:
    """Return finite float32 audio strictly below the configured full-scale ceiling."""
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim != 1 or not np.isfinite(waveform).all():
        raise ValueError("dataset audio must be finite mono float32")
    peak = float(np.max(np.abs(waveform)))
    if peak >= peak_ceiling:
        waveform = waveform * np.float32(peak_ceiling / peak)
    return np.ascontiguousarray(waveform, dtype=np.float32)


def _write_flac(path: Path, audio: np.ndarray) -> None:
    """Write one mono 16 kHz FLAC at 24-bit PCM to retain quiet room tone."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, SAMPLE_RATE_HZ, format="FLAC", subtype="PCM_24")


def _recording(
    passage: Passage,
    recording_id: str,
    kind: str,
    severity_level: int,
    audio_path: Path,
    duration_s: float,
) -> Recording:
    """Create common recording metadata for one passage variant."""
    return Recording(
        id=recording_id,
        transcript_id=passage.passage_id,
        speaker_id=passage.speaker_id,
        speaker_gender=passage.gender,
        kind=kind,
        severity_level=severity_level,
        audio_path=audio_path.as_posix(),
        sample_rate=SAMPLE_RATE_HZ,
        duration_s=duration_s,
        source=SOURCE_NAME,
        license=SOURCE_LICENSE,
        parent_recording_id=(
            None if kind == "ideal" else _recording_id(passage.passage_id, "ideal")
        ),
    )


def _manifest_row(
    recording: Recording,
    passage: Passage,
    flaw_types: Sequence[str],
) -> dict[str, str | int | float]:
    """Build a CSV record with stable representations of list-valued fields."""
    return {
        "recording_id": recording.id,
        "passage_id": passage.passage_id,
        "speaker": passage.speaker_id,
        "gender": passage.gender,
        "split": passage.split,
        "kind": recording.kind,
        "severity_level": recording.severity_level,
        "number_of_flaws": len(flaw_types),
        "flaw_types": ";".join(flaw_types),
        "path": recording.audio_path,
        "duration_s": f"{recording.duration_s:.6f}",
        "source": recording.source,
        "license": recording.license,
    }


def _write_recording(
    project_root: Path,
    passage: Passage,
    recording_id: str,
    kind: str,
    severity_level: int,
    audio: np.ndarray,
    timings: Sequence[WordTiming],
    pair: Pair | None,
    quality_report: dict | None,
    flaw_types: Sequence[str],
    peak_ceiling: float,
    original_timings: Sequence[WordTiming] | None = None,
) -> dict:
    """Persist audio and per-recording JSON sidecar, returning manifest data."""
    audio_dir = project_root / "data" / "processed" / "audio"
    label_dir = project_root / "data" / "labels"
    audio = _safe_audio(audio, peak_ceiling)
    audio_path = audio_dir / f"{recording_id}.flac"
    _write_flac(audio_path, audio)
    recording = _recording(
        passage,
        recording_id,
        kind,
        severity_level,
        Path(_relative_path(project_root, audio_path)),
        audio.size / SAMPLE_RATE_HZ,
    )
    sidecar = {
        "recording": recording.model_dump(mode="json"),
        "passage_id": passage.passage_id,
        "split": passage.split,
        "gender": passage.gender,
        "pair": None if pair is None else pair.model_dump(mode="json"),
        "word_timings": [timing.model_dump(mode="json") for timing in timings],
        "original_word_timings": [
            timing.model_dump(mode="json")
            for timing in (original_timings if original_timings is not None else timings)
        ],
        "alignment_quality": quality_report,
        "control_transform": None,
    }
    _json_write(label_dir / f"{recording_id}.json", sidecar)
    return _manifest_row(recording, passage, flaw_types)


def _lossy_round_trip(audio: np.ndarray) -> tuple[np.ndarray, str]:
    """Use MP3 when the installed encoder works; otherwise use tested OGG/Vorbis."""
    formats = sf.available_formats()
    with tempfile.TemporaryDirectory(
        prefix="speechlens-codec-", dir=Path.cwd()
    ) as directory:
        if (
            "MP3" in formats
            and "MPEG_LAYER_III" in sf.available_subtypes("MP3")
        ):
            codec = "mp3"
            path = Path(directory) / "control.mp3"
            try:
                sf.write(path, audio, SAMPLE_RATE_HZ, format="MP3", subtype="MPEG_LAYER_III")
            except sf.LibsndfileError:
                codec = "ogg_vorbis"
            if codec == "mp3":
                round_trip, _ = sf.read(path, dtype="float32", always_2d=False)
                return np.asarray(round_trip, dtype=np.float32), codec

        if "OGG" not in formats or "VORBIS" not in sf.available_subtypes("OGG"):
            raise RuntimeError("neither MP3 nor OGG/Vorbis encode/decode is available")
        path = Path(directory) / "control.ogg"
        sf.write(path, audio, SAMPLE_RATE_HZ, format="OGG", subtype="VORBIS")
        round_trip, _ = sf.read(path, dtype="float32", always_2d=False)
        return np.asarray(round_trip, dtype=np.float32), "ogg_vorbis"


def _control_variants(
    audio: np.ndarray,
    passage_index: int,
    seed: int,
    config: dict,
) -> list[tuple[str, np.ndarray, str]]:
    """Create identity, lossy, gain, and deterministic SNR noise controls."""
    settings = config["controls"]
    variants: list[tuple[str, np.ndarray, str]] = [
        ("control_identity", identity_resynthesis(audio), "PSOLA identity resynthesis")
    ]
    lossy_audio, codec = _lossy_round_trip(audio)
    variants.append((f"control_lossy_{codec}", lossy_audio, f"{codec} lossy round trip"))

    gain_magnitude_db = float(settings["gain_db"])
    gain_db = gain_magnitude_db if passage_index % 2 == 0 else -gain_magnitude_db
    gain_factor = 10.0 ** (gain_db / 20.0)
    gained = audio * np.float32(gain_factor)
    variants.append((
        f"control_gain_{'plus' if gain_db > 0 else 'minus'}{gain_magnitude_db:g}db",
        gained,
        f"requested gain {gain_db:+.1f} dB, peak-limited below full scale",
    ))

    rng = np.random.default_rng(np.random.SeedSequence([seed, passage_index, 9127]))
    signal_power = float(np.mean(np.square(audio, dtype=np.float64)))
    noise_power = signal_power / (10.0 ** (float(settings["noise_snr_db"]) / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=audio.size).astype(np.float32)
    variants.append((
        "control_mild_noise",
        audio + noise,
        f"additive Gaussian noise at {float(settings['noise_snr_db']):.1f} dB SNR",
    ))
    return variants


def _quality_report(timings: Sequence[WordTiming], config: dict) -> dict:
    """Summarize and flag low-confidence alignment without dropping passages."""
    confidence_values = [
        float(timing.confidence)
        for timing in timings
        if timing.confidence is not None
    ]
    if not confidence_values:
        median_confidence = 0.0
        low_count = len(timings)
    else:
        median_confidence = float(statistics.median(confidence_values))
        low_count = sum(
            value < float(config["alignment"]["low_confidence_threshold"])
            for value in confidence_values
        )
    low_fraction = low_count / max(1, len(timings))
    flagged = (
        median_confidence < float(config["alignment"]["minimum_median_confidence"])
        or low_fraction > float(config["alignment"]["maximum_low_confidence_fraction"])
    )
    return {
        "word_count": len(timings),
        "median_confidence": median_confidence,
        "words_below_0_3": low_count,
        "low_confidence_fraction": low_fraction,
        "flagged": flagged,
    }


def _passage_rng_seed(seed: int, passage: Passage) -> int:
    """Derive a stable independent seed from the global seed and speaker ID."""
    return int(np.random.SeedSequence([seed, int(passage.speaker_id)]).generate_state(1)[0])


def _write_manifest(project_root: Path, rows: list[dict]) -> Path:
    """Write the stable manifest CSV with the required columns."""
    manifest_path = project_root / "data" / "labels" / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=MANIFEST_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def _write_checksums(project_root: Path) -> Path:
    """Hash generated labels and processed data in stable relative-path order."""
    roots = (project_root / "data" / "processed", project_root / "data" / "labels")
    target = project_root / "data" / "labels" / "checksums.sha256"
    files = sorted(
        (
            path
            for root in roots
            for path in root.rglob("*")
            if path.is_file() and path != target
        ),
        key=lambda path: path.relative_to(project_root).as_posix(),
    )
    entries = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(f"{digest}  {path.relative_to(project_root).as_posix()}")
    target.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return target


def _disk_size(path: Path) -> int:
    """Sum bytes for all regular files below a directory."""
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _write_datasheet(project_root: Path, manifest_path: Path, config: dict, flags: list[str]) -> None:
    """Generate dataset facts and file counts from the finalized manifest."""
    with manifest_path.open(encoding="utf-8", newline="") as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    counts = Counter(row["kind"] for row in rows)
    split_counts = Counter((row["split"], row["gender"]) for row in rows if row["kind"] == "ideal")
    flaw_settings = config["ladder"]["levels"]
    levels_summary = ", ".join(
        f"L{level['severity_level']}: {level['flaw_count']} flaws at severity {level['severity']}"
        for level in flaw_settings
    )
    flagged_text = ", ".join(flags) if flags else "None"
    passage_count = len({row["passage_id"] for row in rows})
    text = f"""# SpeechLens Dataset Datasheet

## Sources and License

LibriSpeech ASR corpus, dev-clean, downloaded from
https://openslr.trmal.net/resources/12/dev-clean.tar.gz. OpenSLR lists the
license as CC BY 4.0. Citation: Vassil Panayotov, Guoguo Chen, Daniel Povey,
and Sanjeev Khudanpur, “LibriSpeech: an ASR corpus based on public domain audio
books,” ICASSP 2015. The extracted passages are attributed in
`data/raw/SOURCES.md`.

## Construction

Passages are selected by ascending numeric speaker ID and chapter ID, using
SPEAKERS.TXT gender metadata. Utterance edge trimming uses a frame-RMS
threshold of {config['passages']['trim_threshold_dbfs']} dBFS, with a
{config['passages']['trim_frame_ms']} ms frame, {config['passages']['trim_hop_ms']} ms
hop, and {config['passages']['trim_padding_ms']} ms retained padding. Consecutive
trimmed utterances are joined with {config['passages']['silence_between_utterances_s']} s
of silence. The exact source transcript lines are joined in utterance order.
Processed audio is stored as mono {SAMPLE_RATE_HZ} Hz FLAC with 24-bit PCM so
quiet source-matched room tone is not quantized to digital zero.

Ideal passage word timings are generated with CPU MMS_FA forced alignment.
Alignment confidence is retained and low-confidence passages are flagged, not
dropped. Flaws use seeded, non-overlapping word regions with a configured
minimum separation. The severity ladder is: {levels_summary}.

## Flaw and Control Definitions

Flaw types are pace-fast and pace-slow PSOLA duration edits, long-pause silence
insertion using looped, faded room tone from the clip's quietest sustained
non-silent segment, monotone PSOLA F0 compression, smooth volume dropoff,
synthetic filler, and repeated preceding words with severity-scaled stutter
gaps. The severity-duration knots and stumble bands are defined in
`config/injection.yaml`. The filler is a synthetic approximation made from
same-speaker voiced audio. PSOLA may introduce timbre or boundary artifacts.
Praat duration overlap-add output is cached by source audio and transform
settings so repeated builds in this workspace reuse identical samples; a fresh
cache's first native Praat realization is not guaranteed byte-identical across
machines. LibriSpeech audiobook readings are treated as clean fluent reference
speech by volunteers, not as professional orator performances.

Each ideal passage has identity PSOLA resynthesis, lossy codec round trip, gain
change, and additive noise controls. Gain sign alternates deterministically
between passages and is peak-limited. Noise SNR is {config['controls']['noise_snr_db']} dB.
The available installed MP3 encoder is used for lossy round trips.

## Splits and Counts

Counts below are derived from `data/labels/manifest.csv`.

- Unique passages: {passage_count}
- Ideal recordings: {counts['ideal']}
- Injected recordings: {counts['injected']}
- Control recordings: {counts['control']}
- Total recordings: {len(rows)}
- Dev ideal recordings by gender: M={split_counts[('dev', 'M')]}, F={split_counts[('dev', 'F')]}
- Test ideal recordings by gender: M={split_counts[('test', 'M')]}, F={split_counts[('test', 'F')]}
- Alignment quality flags: {flagged_text}

Speakers are disjoint across dev and test. Dev is intended for threshold tuning;
test is held for final reporting. Passage speaker, chapter, utterance IDs,
transcript, and split are recorded under `data/labels/passages/`.
"""
    (project_root / "docs" / "DATASHEET.md").write_text(text, encoding="utf-8")


def build_dataset(project_root: Path) -> dict:
    """Create all recordings, validate them, hash outputs, and report metrics."""
    started_at = time.perf_counter()
    config = load_config(project_root)
    passage_config = load_dataset_config(project_root)
    seed = int(config["seed"])
    peak_ceiling = float(config["injection"]["peak_ceiling"])
    passages = select_passages(project_root)
    rows: list[dict] = []
    quality_flags: list[str] = []
    audio_output_root = project_root / "data" / "processed" / "audio"
    audio_output_root.mkdir(parents=True, exist_ok=True)

    for passage_index, passage in enumerate(passages):
        source_path = project_root / "data" / "processed" / "passages" / f"{passage.passage_id}.wav"
        ideal_audio, sample_rate = sf.read(source_path, dtype="float32", always_2d=False)
        if sample_rate != SAMPLE_RATE_HZ or ideal_audio.ndim != 1:
            raise ValueError(f"passage audio is not mono {SAMPLE_RATE_HZ} Hz: {source_path}")
        transcript = passage.transcript
        timings = align(ideal_audio, transcript)
        quality = _quality_report(timings, config)
        if quality["flagged"]:
            quality_flags.append(passage.passage_id)
            print(
                f"ALIGNMENT FLAG {passage.passage_id}: median="
                f"{quality['median_confidence']:.4f}, low words="
                f"{quality['words_below_0_3']}/{quality['word_count']}"
            )

        ideal_id = _recording_id(passage.passage_id, "ideal")
        ideal_row = _write_recording(
            project_root,
            passage,
            ideal_id,
            "ideal",
            0,
            ideal_audio,
            timings,
            None,
            quality,
            (),
            peak_ceiling,
        )
        rows.append(ideal_row)

        passage_seed = _passage_rng_seed(seed, passage)
        ladder = generate_ladder(
            timings,
            passage.utterance_join_word_indices,
            config,
            passage_seed,
        )
        for level in ladder:
            severity_level = int(level["severity_level"])
            specs: list[FlawSpec] = level["flaws"]
            flawed_audio, updated_timings, labels = apply_flaws(
                ideal_audio,
                timings,
                specs,
                seed=passage_seed + severity_level,
            )
            recording_id = _recording_id(passage.passage_id, f"L{severity_level}")
            recording_path = Path("data/processed/audio") / f"{recording_id}.flac"
            recording = _recording(
                passage,
                recording_id,
                "injected",
                severity_level,
                recording_path,
                flawed_audio.size / SAMPLE_RATE_HZ,
            )
            pair = Pair(
                ideal_recording_id=ideal_id,
                flawed_recording_id=recording_id,
                flaws=labels,
            )
            row = _write_recording(
                project_root,
                passage,
                recording_id,
                "injected",
                severity_level,
                flawed_audio,
                updated_timings,
                pair,
                None,
                [label.flaw_type for label in labels],
                peak_ceiling,
                original_timings=timings,
            )
            rows.append(row)

        for control_id_suffix, control_audio, transform in _control_variants(
            ideal_audio, passage_index, seed, config
        ):
            recording_id = _recording_id(passage.passage_id, control_id_suffix)
            control_audio = _safe_audio(control_audio, peak_ceiling)
            row = _write_recording(
                project_root,
                passage,
                recording_id,
                "control",
                0,
                control_audio,
                timings,
                None,
                None,
                (),
                peak_ceiling,
            )
            sidecar_path = project_root / "data" / "labels" / f"{recording_id}.json"
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["control_transform"] = transform
            _json_write(sidecar_path, sidecar)
            rows.append(row)

        print(
            f"Built {passage.passage_id}: {len(timings)} aligned words, "
            f"median confidence {quality['median_confidence']:.4f}"
        )

    rows.sort(key=lambda row: str(row["recording_id"]))
    manifest_path = _write_manifest(project_root, rows)
    _write_datasheet(project_root, manifest_path, config, quality_flags)
    validation = validate_dataset(project_root, create_spot_checks=True)
    if not validation["valid"]:
        raise RuntimeError("dataset validation failed: " + "; ".join(validation["errors"][:10]))
    checksums_path = _write_checksums(project_root)

    elapsed_s = time.perf_counter() - started_at
    disk_size_bytes = _disk_size(project_root / "data" / "processed")
    total_duration_s = sum(float(row["duration_s"]) for row in rows)
    result = {
        "passage_count": len(passages),
        "recording_count": len(rows),
        "control_count": sum(row["kind"] == "control" for row in rows),
        "total_duration_s": total_duration_s,
        "processed_disk_size_bytes": disk_size_bytes,
        "alignment_quality_flags": quality_flags,
        "validation": validation,
        "checksums_path": checksums_path,
        "elapsed_s": elapsed_s,
    }
    print("DATASET SUMMARY")
    print(f"Passages: {result['passage_count']}")
    print(f"Recordings: {result['recording_count']}")
    print(f"Controls: {result['control_count']}")
    print(f"Total audio duration: {total_duration_s / 60.0:.2f} minutes")
    print(f"data/processed disk size: {disk_size_bytes / (1024**2):.2f} MiB")
    print(f"Alignment quality flags: {quality_flags or 'none'}")
    print(f"Validation: {validation['checked_recordings']} recordings passed")
    print(f"Checksums: {checksums_path.relative_to(project_root).as_posix()}")
    print(f"Runtime: {elapsed_s:.2f} seconds")
    return result


def main() -> None:
    """Build the dataset from the repository root."""
    project_root = Path(__file__).resolve().parents[1]
    try:
        build_dataset(project_root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()