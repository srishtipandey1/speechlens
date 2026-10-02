"""Offline greedy CTC transcript comparison for inserted-word grounding."""

from __future__ import annotations

import hashlib
import json
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torchaudio

from speechlens.detection.measurements import normalize_token


def _frame_stride_samples(model: torch.nn.Module) -> int:
    """Return the product of the ASR feature extractor's convolution strides."""
    base_model = getattr(model, "model", model)
    feature_extractor = base_model.feature_extractor
    return int(np.prod([
        int(layer.conv.stride[0]) for layer in feature_extractor.conv_layers
    ]))


@lru_cache(maxsize=2)
def _load_asr_model(model_cache_dir: str) -> tuple[torch.nn.Module, tuple[str, ...], int, int]:
    """Load cached Wav2Vec2 ASR weights without allowing implicit downloads."""
    cache_root = Path(model_cache_dir).resolve()
    checkpoint = cache_root / "checkpoints" / torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H._path
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Wav2Vec2 ASR checkpoint is not cached: {checkpoint}")
    previous_cache_dir = torch.hub.get_dir()
    torch.hub.set_dir(str(cache_root))
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    try:
        model = bundle.get_model(dl_kwargs={"progress": False}).cpu().eval()
    finally:
        torch.hub.set_dir(previous_cache_dir)
    return model, bundle.get_labels(), bundle.sample_rate, _frame_stride_samples(model)


def greedy_ctc_word_spans(
    emissions: torch.Tensor,
    labels: tuple[str, ...],
    frame_step_s: float,
) -> list[tuple[str, float, float, float]]:
    """Greedily collapse CTC emissions into word spans and token confidences."""
    if emissions.ndim == 3:
        if emissions.shape[0] != 1:
            raise ValueError("greedy decoding expects one waveform at a time")
        emissions = emissions[0]
    if emissions.ndim != 2 or emissions.shape[1] != len(labels):
        raise ValueError("emissions must have shape (frames, labels)")
    if frame_step_s <= 0:
        raise ValueError("frame_step_s must be positive")

    probabilities = torch.softmax(emissions, dim=-1)
    best_probabilities, best_indices = probabilities.max(dim=-1)
    blank_index = labels.index("-")
    separator_index = labels.index("|")
    character_spans: list[tuple[str, int, int, float]] = []
    previous_index = blank_index
    run_start = 0
    for frame_index, token_index in enumerate(best_indices.tolist() + [blank_index]):
        if token_index == previous_index:
            continue
        if previous_index not in (blank_index, separator_index):
            mean_probability = float(best_probabilities[run_start:frame_index].mean())
            character_spans.append((
                labels[previous_index], run_start, frame_index, mean_probability
            ))
        if token_index == separator_index:
            character_spans.append(("|", frame_index, frame_index + 1, 1.0))
        if token_index not in (blank_index, separator_index):
            run_start = frame_index
        previous_index = token_index

    word_spans: list[tuple[str, float, float, float]] = []
    current_characters: list[tuple[str, int, int, float]] = []
    for character in character_spans + [("|", 0, 0, 0.0)]:
        if character[0] == "|":
            if current_characters:
                word_spans.append((
                    "".join(item[0] for item in current_characters).casefold(),
                    current_characters[0][1] * frame_step_s,
                    current_characters[-1][2] * frame_step_s,
                    float(np.mean([item[3] for item in current_characters])),
                ))
                current_characters = []
        else:
            current_characters.append(character)
    return word_spans


def _inserted_word_indices(
    reference_tokens: list[str], decoded_tokens: list[str]
) -> set[int]:
    """Select decoded insertions and filler tokens without treating substitutions as insertions."""
    matcher = SequenceMatcher(a=reference_tokens, b=decoded_tokens, autojunk=False)
    inserted: set[int] = set()
    filler_tokens = {"uh", "um", "erm", "hmm"}
    for operation, _, _, decoded_start, decoded_end in matcher.get_opcodes():
        if operation == "insert":
            inserted.update(range(decoded_start, decoded_end))
        elif operation == "replace":
            inserted.update(
                index for index in range(decoded_start, decoded_end)
                if decoded_tokens[index] in filler_tokens
            )
    return inserted


def _decode_inserted_words(
    audio: np.ndarray,
    reference_transcript: str,
    model_cache_dir: Path,
    chunk_duration_s: float,
    overlap_s: float,
) -> list[tuple[float, float, float]]:
    model, labels, sample_rate_hz, stride_samples = _load_asr_model(str(model_cache_dir.resolve()))
    waveform = np.ascontiguousarray(audio, dtype=np.float32)
    if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("ASR audio must be a finite, non-empty mono waveform")
    if chunk_duration_s <= 0 or overlap_s < 0 or overlap_s >= chunk_duration_s:
        raise ValueError("ASR chunk duration must be positive and overlap smaller than the chunk")
    chunk_samples = round(chunk_duration_s * sample_rate_hz)
    overlap_samples = round(overlap_s * sample_rate_hz)
    step_samples = chunk_samples - overlap_samples
    decoded: list[tuple[str, float, float, float]] = []
    chunk_start = 0
    while chunk_start < waveform.size:
        chunk_end = min(waveform.size, chunk_start + chunk_samples)
        core_start = 0 if chunk_start == 0 else chunk_start + overlap_samples // 2
        core_end = waveform.size if chunk_end == waveform.size else chunk_end - overlap_samples // 2
        chunk = torch.from_numpy(waveform[chunk_start:chunk_end]).unsqueeze(0)
        with torch.inference_mode():
            emissions, _ = model(chunk)
        chunk_words = greedy_ctc_word_spans(
            emissions,
            labels,
            stride_samples / sample_rate_hz,
        )
        offset_s = chunk_start / sample_rate_hz
        core_start_s = core_start / sample_rate_hz
        core_end_s = core_end / sample_rate_hz
        decoded.extend(
            (word, start_s + offset_s, end_s + offset_s, confidence)
            for word, start_s, end_s, confidence in chunk_words
            if core_start_s <= (start_s + end_s) * 0.5 + offset_s < core_end_s
        )
        if chunk_end == waveform.size:
            break
        chunk_start += step_samples
    expected_tokens = [normalize_token(word) for word in reference_transcript.split()]
    decoded_tokens = [normalize_token(word[0]) for word in decoded]
    inserted_decoded = _inserted_word_indices(expected_tokens, decoded_tokens)
    duration_s = waveform.size / sample_rate_hz
    return [
        (max(0.0, start_s), min(duration_s, end_s), confidence)
        for index, (_, start_s, end_s, confidence) in enumerate(decoded)
        if index in inserted_decoded and end_s > start_s
    ]


def cached_inserted_word_spans(
    audio: np.ndarray,
    reference_transcript: str,
    cache_dir: Path,
    model_cache_dir: Path,
    chunk_duration_s: float,
    overlap_s: float,
) -> list[tuple[float, float, float]]:
    """Return unmatched greedy-ASR word spans from a deterministic local cache."""
    waveform = np.ascontiguousarray(audio, dtype=np.float32)
    cache_material = (
        b"wav2vec2-asr-base-960h-v1\0"
        + hashlib.sha256(waveform.tobytes()).digest()
        + reference_transcript.encode("utf-8")
        + f"\0{chunk_duration_s:.6f}\0{overlap_s:.6f}".encode("ascii")
    )
    path = cache_dir / f"{hashlib.sha256(cache_material).hexdigest()}.json"
    if path.is_file():
        with path.open(encoding="utf-8") as cache_file:
            values = json.load(cache_file)
        return [tuple(map(float, value)) for value in values]
    spans = _decode_inserted_words(
        waveform,
        reference_transcript,
        model_cache_dir,
        chunk_duration_s,
        overlap_s,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(spans, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return spans