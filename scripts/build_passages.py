"""Deterministically select and assemble LibriSpeech dev-clean passages."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from speechlens.alignment.align import SAMPLE_RATE_HZ, normalize_transcript


TRANSCRIPT_PATTERN = re.compile(r"^(\d+)-(\d+)\.trans\.txt$")
SPEAKER_GENDERS = {"M", "F"}


@dataclass(frozen=True)
class UtteranceChunk:
    """One trimmed utterance and the exact transcript line it came from."""

    utterance_id: str
    text: str
    audio: np.ndarray


@dataclass(frozen=True)
class Passage:
    """A selected single-speaker, single-chapter joined speech passage."""

    passage_id: str
    speaker_id: str
    gender: str
    chapter_id: str
    split: str
    utterances: tuple[UtteranceChunk, ...]
    audio: np.ndarray
    transcript: str
    utterance_join_word_indices: tuple[int, ...]


def load_dataset_config(project_root: Path) -> dict:
    """Read the deterministic passage selection configuration."""
    with (project_root / "config" / "dataset.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def read_speakers(speaker_file: Path) -> dict[str, dict[str, str]]:
    """Parse speaker ID and gender from the official SPEAKERS.TXT metadata."""
    speakers: dict[str, dict[str, str]] = {}
    for line in speaker_file.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith(";"):
            continue
        columns = [column.strip() for column in value.split("|")]
        if len(columns) < 5:
            continue
        speaker_id, gender, subset, minutes, name = columns[:5]
        if gender in SPEAKER_GENDERS:
            speakers[speaker_id] = {
                "gender": gender,
                "subset": subset,
                "minutes": minutes,
                "name": name,
            }
    return speakers


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold_dbfs: float,
    frame_ms: float,
    hop_ms: float,
    padding_ms: float,
) -> np.ndarray:
    """Trim edges below a fixed frame-RMS dBFS threshold, retaining padding."""
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1, dtype=np.float32)
    if waveform.ndim != 1 or waveform.size == 0:
        return np.empty(0, dtype=np.float32)
    frame_samples = max(1, round(frame_ms * sample_rate / 1000.0))
    hop_samples = max(1, round(hop_ms * sample_rate / 1000.0))
    padding_samples = round(padding_ms * sample_rate / 1000.0)
    threshold = 10.0 ** (threshold_dbfs / 20.0)
    active_frames: list[int] = []
    for start in range(0, waveform.size, hop_samples):
        frame = waveform[start : start + frame_samples]
        if frame.size and float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))) >= threshold:
            active_frames.append(start)
    if not active_frames:
        return np.empty(0, dtype=np.float32)
    start_sample = max(0, active_frames[0] - padding_samples)
    end_sample = min(waveform.size, active_frames[-1] + frame_samples + padding_samples)
    return np.ascontiguousarray(waveform[start_sample:end_sample], dtype=np.float32)


def read_chapter_utterances(chapter_dir: Path, trim_settings: dict) -> list[UtteranceChunk]:
    """Read and silence-trim utterances in transcript-file order."""
    transcript_files = sorted(chapter_dir.glob("*.trans.txt"))
    if not transcript_files:
        return []
    transcript_path = transcript_files[0]
    if TRANSCRIPT_PATTERN.match(transcript_path.name) is None:
        return []
    chunks: list[UtteranceChunk] = []
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        utterance_id, separator, text = line.partition(" ")
        if not separator or not text.strip():
            continue
        audio_path = chapter_dir / f"{utterance_id}.flac"
        if not audio_path.is_file():
            continue
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1, dtype=np.float32)
        if sample_rate != SAMPLE_RATE_HZ:
            raise ValueError(f"unexpected sample rate in {audio_path}: {sample_rate}")
        trimmed = trim_silence(
            mono,
            sample_rate,
            float(trim_settings["trim_threshold_dbfs"]),
            float(trim_settings["trim_frame_ms"]),
            float(trim_settings["trim_hop_ms"]),
            float(trim_settings["trim_padding_ms"]),
        )
        if trimmed.size:
            chunks.append(UtteranceChunk(utterance_id, text.strip(), trimmed))
    return chunks


def select_window(
    utterances: list[UtteranceChunk],
    minimum_duration_s: float,
    maximum_duration_s: float,
    silence_s: float,
) -> tuple[UtteranceChunk, ...] | None:
    """Return the earliest-start shortest consecutive window within duration bounds."""
    for start_index in range(len(utterances)):
        selected: list[UtteranceChunk] = []
        audio_samples = 0
        for utterance in utterances[start_index:]:
            selected.append(utterance)
            audio_samples += utterance.audio.size
            silence_samples = round(silence_s * SAMPLE_RATE_HZ) * (len(selected) - 1)
            duration_s = (audio_samples + silence_samples) / SAMPLE_RATE_HZ
            if duration_s > maximum_duration_s:
                break
            if duration_s >= minimum_duration_s:
                return tuple(selected)
    return None


def _assemble(
    utterances: tuple[UtteranceChunk, ...], silence_s: float
) -> tuple[np.ndarray, str, tuple[int, ...]]:
    """Concatenate trimmed utterances with exact silence and source text."""
    silence = np.zeros(round(silence_s * SAMPLE_RATE_HZ), dtype=np.float32)
    audio_parts: list[np.ndarray] = []
    transcript_parts: list[str] = []
    join_word_indices: list[int] = []
    preceding_word_count = 0
    for index, utterance in enumerate(utterances):
        if index:
            audio_parts.append(silence)
            join_word_indices.append(preceding_word_count - 1)
        audio_parts.append(utterance.audio)
        transcript_parts.append(utterance.text)
        source_indices = {item.original_index for item in normalize_transcript(utterance.text)}
        preceding_word_count += len(source_indices)
    return (
        np.ascontiguousarray(np.concatenate(audio_parts), dtype=np.float32),
        " ".join(transcript_parts),
        tuple(join_word_indices),
    )


def _find_passage_for_speaker(
    speaker_dir: Path,
    speaker_id: str,
    gender: str,
    split: str,
    config: dict,
) -> Passage | None:
    """Choose the first chapter/window for a speaker in numeric chapter order."""
    passage_settings = config["passages"]
    trim_settings = passage_settings
    for chapter_dir in sorted(
        (path for path in speaker_dir.iterdir() if path.is_dir()),
        key=lambda path: int(path.name),
    ):
        utterances = read_chapter_utterances(chapter_dir, trim_settings)
        selected = select_window(
            utterances,
            float(passage_settings["minimum_duration_s"]),
            float(passage_settings["maximum_duration_s"]),
            float(passage_settings["silence_between_utterances_s"]),
        )
        if not selected:
            continue
        audio, transcript, joins = _assemble(
            selected,
            float(passage_settings["silence_between_utterances_s"]),
        )
        passage_id = f"{split}_{gender.lower()}_{speaker_id}"
        return Passage(
            passage_id,
            speaker_id,
            gender,
            chapter_dir.name,
            split,
            selected,
            audio,
            transcript,
            joins,
        )
    return None


def select_passages(project_root: Path) -> list[Passage]:
    """Select distinct speakers with balanced gender and split membership."""
    corpus_root = project_root / "data" / "raw" / "librispeech_dev_clean" / "LibriSpeech"
    data_root = corpus_root / "dev-clean"
    if not (corpus_root / "SPEAKERS.TXT").is_file() or not data_root.is_dir():
        raise FileNotFoundError("LibriSpeech dev-clean extraction is missing; run its fetcher first")
    config = load_dataset_config(project_root)
    settings = config["passages"]
    speakers = read_speakers(corpus_root / "SPEAKERS.TXT")
    selected_passages: list[Passage] = []
    required_per_gender = int(settings["count_per_gender"])
    dev_per_gender = int(settings["dev_per_gender"])

    for gender in ("M", "F"):
        eligible: list[Passage] = []
        speaker_dirs = sorted(
            (
                path
                for path in data_root.iterdir()
                if path.is_dir() and speakers.get(path.name, {}).get("gender") == gender
            ),
            key=lambda path: int(path.name),
        )
        for speaker_dir in speaker_dirs:
            if len(eligible) >= required_per_gender:
                break
            speaker_index = len(eligible)
            split = "dev" if speaker_index < dev_per_gender else "test"
            passage = _find_passage_for_speaker(
                speaker_dir,
                speaker_dir.name,
                gender,
                split,
                config,
            )
            if passage is not None:
                eligible.append(passage)
        if len(eligible) != required_per_gender:
            raise RuntimeError(
                f"found {len(eligible)} eligible {gender} speakers; "
                f"required {required_per_gender}"
            )
        selected_passages.extend(eligible)

    write_passages(project_root, selected_passages, config)
    return selected_passages


def write_passages(project_root: Path, passages: list[Passage], config: dict) -> None:
    """Write reproducible passage WAVs, transcripts, and provenance JSON."""
    audio_root = project_root / "data" / "processed" / "passages"
    label_root = project_root / "data" / "labels" / "passages"
    audio_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)
    settings = config["passages"]
    records: list[dict] = []
    for passage in passages:
        audio_path = audio_root / f"{passage.passage_id}.wav"
        sf.write(audio_path, passage.audio, SAMPLE_RATE_HZ, subtype="PCM_16")
        utterance_ids = [item.utterance_id for item in passage.utterances]
        metadata = {
            "passage_id": passage.passage_id,
            "speaker_id": passage.speaker_id,
            "gender": passage.gender,
            "chapter_id": passage.chapter_id,
            "split": passage.split,
            "utterance_ids": utterance_ids,
            "transcript": passage.transcript,
            "utterance_join_word_indices": list(passage.utterance_join_word_indices),
            "duration_s": passage.audio.size / SAMPLE_RATE_HZ,
            "source": "LibriSpeech ASR corpus, dev-clean",
            "source_url": "https://openslr.trmal.net/resources/12/dev-clean.tar.gz",
            "license": "CC BY 4.0",
            "trim_threshold_dbfs": float(settings["trim_threshold_dbfs"]),
            "silence_between_utterances_s": float(settings["silence_between_utterances_s"]),
            "audio_path": audio_path.relative_to(project_root).as_posix(),
        }
        label_path = label_root / f"{passage.passage_id}.json"
        label_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        records.append(metadata)

    index_path = project_root / "data" / "labels" / "passages.json"
    index_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Selected {len(passages)} passages ({sum(p.split == 'dev' for p in passages)} dev, {sum(p.split == 'test' for p in passages)} test)")


def main() -> None:
    """Build and save the deterministic passage selection."""
    project_root = Path(__file__).resolve().parents[1]
    try:
        passages = select_passages(project_root)
        for passage in passages:
            print(
                f"{passage.passage_id}: speaker={passage.speaker_id} "
                f"gender={passage.gender} chapter={passage.chapter_id} "
                f"duration={passage.audio.size / SAMPLE_RATE_HZ:.2f}s "
                f"utterances={len(passage.utterances)}"
            )
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()