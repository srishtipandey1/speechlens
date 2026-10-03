"""Pure-logic and optional integration tests for forced alignment."""

from pathlib import Path
from types import SimpleNamespace
import tempfile

import numpy as np
import pytest
import soundfile as sf

from speechlens.alignment.align import (
    align,
    _frame_stride_samples,
    frame_to_seconds,
    load_audio,
    normalize_transcript,
)


def test_normalize_transcript_strips_punctuation_and_keeps_apostrophes() -> None:
    """Normalize punctuation while preserving contractions and source mapping."""
    tokens = normalize_transcript("Hello, don't stop!")

    assert [item.token for item in tokens] == ["hello", "don't", "stop"]
    assert [item.original_word for item in tokens] == ["Hello,", "don't", "stop!"]
    assert [item.original_index for item in tokens] == [0, 1, 2]


def test_normalize_transcript_spells_digits_and_maps_expanded_words() -> None:
    """Number expansion keeps each generated token tied to its source word."""
    tokens = normalize_transcript("I have 42 apples.")

    assert [item.token for item in tokens] == ["i", "have", "forty", "two", "apples"]
    assert [(item.original_word, item.original_index) for item in tokens[2:4]] == [
        ("42", 2),
        ("42", 2),
    ]


@pytest.mark.parametrize("text", ["", "   ", "... !!!"])
def test_normalize_transcript_returns_empty_for_no_words(text: str) -> None:
    """Empty and punctuation-only input produces no alignment tokens."""
    assert normalize_transcript(text) == []


def test_normalize_transcript_preserves_explicit_star_token() -> None:
    """A standalone star passes through as MMS_FA's wildcard token."""
    token = normalize_transcript("hello * there")[1]

    assert token.token == "*"
    assert token.original_word == "*"
    assert token.original_index == 1


def test_frame_to_seconds_uses_supplied_stride_and_sample_rate() -> None:
    """Frame conversion derives time from exact sample-domain parameters."""
    assert frame_to_seconds(50, sample_rate=16000, frame_stride_samples=320) == 1.0


def test_frame_to_seconds_rejects_invalid_parameters() -> None:
    """Frame conversion rejects negative frames and non-positive scales."""
    with pytest.raises(ValueError):
        frame_to_seconds(-1, sample_rate=16000, frame_stride_samples=320)
    with pytest.raises(ValueError):
        frame_to_seconds(1, sample_rate=0, frame_stride_samples=320)


def test_frame_stride_is_product_of_model_convolution_strides() -> None:
    """Derive emission stride from the convolution layers, not a constant."""
    model = SimpleNamespace(
        feature_extractor=SimpleNamespace(
            conv_layers=[
                SimpleNamespace(conv=SimpleNamespace(stride=(5,))),
                SimpleNamespace(conv=SimpleNamespace(stride=(2,))),
                SimpleNamespace(conv=SimpleNamespace(stride=(2,))),
            ]
        )
    )

    assert _frame_stride_samples(model) == 20
    assert _frame_stride_samples(SimpleNamespace(model=model)) == 20


def test_load_audio_returns_mono_float32_at_model_rate() -> None:
    """Load and resample a stereo WAV to the MMS_FA mono sample format."""
    source_rate = 8000
    samples = np.arange(source_rate // 10, dtype=np.float32)
    tone = np.sin(2 * np.pi * 440 * samples / source_rate).astype(np.float32)
    stereo_tone = np.column_stack((tone, tone))

    with tempfile.TemporaryDirectory(prefix="speechlens-audio-test-", dir=Path.cwd()) as directory:
        audio_path = Path(directory) / "tone.wav"
        sf.write(audio_path, stereo_tone, source_rate)
        waveform = load_audio(audio_path)

    assert waveform.dtype == np.float32
    assert waveform.ndim == 1
    assert waveform.size == source_rate // 10 * 2


def test_align_empty_transcript_does_not_load_model() -> None:
    """An empty transcript yields no timings without touching model weights."""
    assert align([], "... !!!") == []


@pytest.mark.slow
def test_align_real_speech_clip(real_alignment_fixture: tuple[Path, Path]) -> None:
    """Align the optional local speech fixture without fetching model weights."""
    audio_path, transcript_path = real_alignment_fixture

    timings = align(
        load_audio(audio_path),
        transcript_path.read_text(encoding="utf-8"),
    )

    assert timings
    assert all(item.start_s < item.end_s for item in timings)
    assert all(item.confidence is not None for item in timings)