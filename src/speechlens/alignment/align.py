"""Transcript normalization and CPU-only MMS_FA forced alignment."""

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml

from speechlens.schema import WordTiming


_WORD_PATTERN = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)*|\d+(?:,\d{3})*(?:\.\d+)?|\*"
)
_SMALL_NUMBERS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = (
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)
_SCALES = ("", "thousand", "million", "billion", "trillion", "quadrillion")
_DIGITS = _SMALL_NUMBERS[:10]
_BUNDLE = torchaudio.pipelines.MMS_FA
SAMPLE_RATE_HZ = int(_BUNDLE.sample_rate)


@dataclass(frozen=True)
class NormalizedToken:
    """A normalized MMS_FA word and its source transcript mapping."""

    token: str
    original_word: str
    original_index: int


def _spell_under_thousand(value: int) -> list[str]:
    """Spell an integer from zero through nine hundred ninety-nine."""
    words: list[str] = []
    hundreds, remainder = divmod(value, 100)
    if hundreds:
        words.extend((_SMALL_NUMBERS[hundreds], "hundred"))
    if remainder < 20:
        if remainder:
            words.append(_SMALL_NUMBERS[remainder])
    else:
        tens, ones = divmod(remainder, 10)
        words.append(_TENS[tens])
        if ones:
            words.append(_SMALL_NUMBERS[ones])
    return words


def _spell_integer(value: int) -> list[str]:
    """Spell a non-negative integer using English cardinal words."""
    if value == 0:
        return ["zero"]

    groups: list[int] = []
    remaining = value
    while remaining:
        remaining, group = divmod(remaining, 1000)
        groups.append(group)
    if len(groups) > len(_SCALES):
        return [_DIGITS[int(digit)] for digit in str(value)]

    words: list[str] = []
    for scale_index in range(len(groups) - 1, -1, -1):
        group = groups[scale_index]
        if group:
            words.extend(_spell_under_thousand(group))
            if _SCALES[scale_index]:
                words.append(_SCALES[scale_index])
    return words


def _spell_number(value: str) -> list[str]:
    """Spell an integer or decimal numeric token in English."""
    integer_part, decimal_separator, fractional_part = value.replace(",", "").partition(".")
    words = _spell_integer(int(integer_part))
    if decimal_separator:
        words.append("point")
        words.extend(_DIGITS[int(digit)] for digit in fractional_part)
    return words


def normalize_transcript(text: str) -> list[NormalizedToken]:
    """Normalize text to MMS_FA tokens while retaining source-word mappings.

    Word indices refer to the zero-based whitespace-tokenized input. A
    standalone ``*`` or ``<star>`` is passed through as MMS_FA's wildcard.
    """
    normalized: list[NormalizedToken] = []
    for original_index, original_word in enumerate(text.split()):
        if original_word.lower() in {"*", "<star>"}:
            normalized.append(NormalizedToken("*", original_word, original_index))
            continue

        for match in _WORD_PATTERN.finditer(original_word):
            value = match.group(0).lower()
            tokens = _spell_number(value) if value[0].isdigit() else [value]
            normalized.extend(
                NormalizedToken(token, original_word, original_index)
                for token in tokens
            )
    return normalized


def frame_to_seconds(frame: int, sample_rate: int, frame_stride_samples: int) -> float:
    """Convert an emission frame index using the model stride and sample rate."""
    if frame < 0:
        raise ValueError("frame must be non-negative")
    if sample_rate <= 0 or frame_stride_samples <= 0:
        raise ValueError("sample_rate and frame_stride_samples must be positive")
    return frame * frame_stride_samples / sample_rate


def _frame_stride_samples(model: torch.nn.Module) -> int:
    """Derive the waveform stride from the model's convolution layers."""
    base_model = getattr(model, "model", model)
    feature_extractor = getattr(base_model, "feature_extractor", None)
    if feature_extractor is None:
        raise ValueError("model does not expose a Wav2Vec2 feature extractor")
    layers = feature_extractor.conv_layers
    stride = math.prod(int(layer.conv.stride[0]) for layer in layers)
    if stride <= 0:
        raise ValueError("model feature extractor has an invalid frame stride")
    return stride


def _cached_checkpoint_path() -> Path:
    """Return the exact checkpoint path used by the installed torch hub loader."""
    return Path(torch.hub.get_dir()) / "checkpoints" / "model.pt"


def has_cached_mms_fa_model() -> bool:
    """Return whether the MMS_FA checkpoint is already present locally."""
    return _cached_checkpoint_path().is_file()


def _configured_seed() -> int:
    """Read the reproducibility seed from the repository default config."""
    config_candidates = (
        Path.cwd() / "config" / "default.yaml",
        Path(__file__).resolve().parents[3] / "config" / "default.yaml",
    )
    config_path = next((path for path in config_candidates if path.is_file()), None)
    if config_path is None:
        raise FileNotFoundError("config/default.yaml is required for alignment")
    with config_path.open(encoding="utf-8") as config_file:
        config: dict[str, Any] = yaml.safe_load(config_file)
    return int(config["seed"])


@lru_cache(maxsize=1)
def _load_alignment_components() -> tuple[torch.nn.Module, Any, int]:
    """Load the cached MMS_FA model and derive its frame stride."""
    checkpoint_path = _cached_checkpoint_path()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"MMS_FA weights are not cached at {checkpoint_path}; "
            "alignment will not download model weights"
        )
    model = _BUNDLE.get_model(with_star=True, dl_kwargs={"progress": False}).cpu()
    tokenizer = _BUNDLE.get_tokenizer()
    return model, tokenizer, _frame_stride_samples(model)


def load_audio(path: str | Path) -> np.ndarray:
    """Load WAV or MP3 audio as mono float32 at the MMS_FA sample rate."""
    audio_path = Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    suffix = audio_path.suffix.lower()
    if suffix == ".wav":
        samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        waveform = samples.mean(axis=1, dtype=np.float32)
        if sample_rate != SAMPLE_RATE_HZ:
            waveform = librosa.resample(
                waveform,
                orig_sr=sample_rate,
                target_sr=SAMPLE_RATE_HZ,
            )
    elif suffix == ".mp3":
        waveform, _ = librosa.load(
            audio_path,
            sr=SAMPLE_RATE_HZ,
            mono=True,
            dtype=np.float32,
        )
    else:
        raise ValueError("audio must be a WAV or MP3 file")

    waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("audio must contain finite, non-empty samples")
    return waveform


def _align_normalized_tokens(
    waveform: np.ndarray,
    normalized_tokens: list[NormalizedToken],
    model: torch.nn.Module,
    tokenizer: Any,
    frame_stride_samples: int,
) -> list[WordTiming]:
    """Align normalized tokens and aggregate frame spans by source word."""
    token_texts = [item.token for item in normalized_tokens]
    token_ids_by_word = tokenizer(token_texts)
    target_ids = [token_id for word_ids in token_ids_by_word for token_id in word_ids]
    target_word_indices = [
        normalized_index
        for normalized_index, word_ids in enumerate(token_ids_by_word)
        for _ in word_ids
    ]
    if not target_ids:
        return []

    audio_tensor = torch.from_numpy(waveform).to(device="cpu").unsqueeze(0)
    targets = torch.tensor([target_ids], dtype=torch.int32, device="cpu")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_configured_seed())
        with torch.no_grad():
            emissions, _ = model(audio_tensor)
            aligned_tokens, frame_scores = torchaudio.functional.forced_align(
                emissions,
                targets,
                target_lengths=torch.tensor([len(target_ids)], dtype=torch.int32),
                blank=0,
            )

    per_source_word: dict[int, list[tuple[int, float]]] = {}
    target_index = -1
    previous_token: int | None = None
    for frame, (aligned_token, frame_score) in enumerate(
        zip(aligned_tokens[0].tolist(), frame_scores[0].tolist())
    ):
        if aligned_token == 0:
            previous_token = None
            continue
        if aligned_token != previous_token:
            target_index += 1
            if target_index >= len(target_ids) or aligned_token != target_ids[target_index]:
                raise RuntimeError("forced alignment path did not match the target tokens")
        normalized_index = target_word_indices[target_index]
        source_index = normalized_tokens[normalized_index].original_index
        per_source_word.setdefault(source_index, []).append((frame, frame_score))
        previous_token = aligned_token

    audio_duration = len(waveform) / SAMPLE_RATE_HZ
    timings: list[WordTiming] = []
    for source_index, original_word in enumerate(
        _source_words(normalized_tokens)
    ):
        frames = per_source_word.get(source_index)
        if not frames:
            continue
        start_frame = min(frame for frame, _ in frames)
        end_frame = max(frame for frame, _ in frames) + 1
        start_s = frame_to_seconds(start_frame, SAMPLE_RATE_HZ, frame_stride_samples)
        end_s = min(
            frame_to_seconds(end_frame, SAMPLE_RATE_HZ, frame_stride_samples),
            audio_duration,
        )
        mean_log_probability = sum(score for _, score in frames) / len(frames)
        confidence = min(1.0, max(0.0, math.exp(mean_log_probability)))
        if start_s < end_s:
            timings.append(
                WordTiming(
                    word=original_word,
                    start_s=start_s,
                    end_s=end_s,
                    confidence=confidence,
                )
            )
    return timings


def _source_words(normalized_tokens: list[NormalizedToken]) -> list[str]:
    """Build original source words indexed by their whitespace position."""
    if not normalized_tokens:
        return []
    source_words = [""] * (max(item.original_index for item in normalized_tokens) + 1)
    for item in normalized_tokens:
        source_words[item.original_index] = item.original_word
    return source_words


def align(audio: np.ndarray, transcript_text: str) -> list[WordTiming]:
    """Align mono 16 kHz audio to transcript words using cached MMS_FA weights."""
    normalized_tokens = normalize_transcript(transcript_text)
    if not normalized_tokens:
        return []
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim != 1:
        raise ValueError("audio must be a mono one-dimensional waveform")
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("audio must contain finite, non-empty samples")
    model, tokenizer, stride = _load_alignment_components()
    return _align_normalized_tokens(
        np.ascontiguousarray(waveform),
        normalized_tokens,
        model,
        tokenizer,
        stride,
    )


def main() -> None:
    """Align an audio file and write word timings as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("transcript", type=Path)
    parser.add_argument("--out", type=Path, default=Path("out.json"))
    arguments = parser.parse_args()

    try:
        timings = align(
            load_audio(arguments.audio),
            arguments.transcript.read_text(encoding="utf-8"),
        )
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(
            json.dumps(
                [timing.model_dump(mode="json") for timing in timings],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()