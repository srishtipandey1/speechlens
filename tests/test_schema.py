"""Validation and serialization tests for the label schema."""

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from speechlens.schema import (
    FlawLabel,
    Pair,
    Recording,
    RecordingCollection,
    Transcript,
    WordTiming,
    load_json,
    save_json,
)


def make_recording(
    recording_id: str,
    kind: str = "ideal",
    parent_recording_id: str | None = None,
) -> Recording:
    """Build a valid recording for schema tests."""
    return Recording(
        id=recording_id,
        transcript_id="transcript-1",
        speaker_id="speaker-1",
        kind=kind,
        severity_level=0 if kind in {"ideal", "control"} else 2,
        audio_path="audio/sample.wav",
        sample_rate=16000,
        duration_s=30.0,
        source="test fixture",
        license="test-only",
        parent_recording_id=parent_recording_id,
    )


def make_flaw(
    rendered_start_s: float = 10.0,
    rendered_end_s: float = 12.0,
) -> FlawLabel:
    """Build a valid flaw label with a shifted rendered interval."""
    return FlawLabel(
        flaw_type="long_pause",
        severity=0.7,
        original_start_s=10.0,
        original_end_s=11.0,
        rendered_start_s=rendered_start_s,
        rendered_end_s=rendered_end_s,
        word_indices=[4, 5],
        notes="Pause extended in rendered audio.",
    )


def test_word_timing_rejects_invalid_and_negative_intervals() -> None:
    """Word intervals must be positive-length and non-negative."""
    with pytest.raises(ValidationError):
        WordTiming(word="hello", start_s=2.0, end_s=2.0)
    with pytest.raises(ValidationError):
        WordTiming(word="hello", start_s=-0.1, end_s=1.0)
    with pytest.raises(ValidationError):
        FlawLabel(
            flaw_type="filler",
            severity=0.5,
            original_start_s=-0.1,
            original_end_s=1.0,
            rendered_start_s=0.0,
            rendered_end_s=1.0,
            word_indices=[],
        )


def test_flaw_rejects_out_of_range_severity_and_invalid_intervals() -> None:
    """Flaw severity and both timeline intervals are constrained."""
    with pytest.raises(ValidationError):
        FlawLabel.model_validate({**make_flaw().model_dump(), "severity": 1.1})
    with pytest.raises(ValidationError):
        FlawLabel(
            flaw_type="filler",
            severity=0.5,
            original_start_s=10.0,
            original_end_s=9.0,
            rendered_start_s=10.0,
            rendered_end_s=12.0,
            word_indices=[],
        )
    with pytest.raises(ValidationError):
        FlawLabel(
            flaw_type="filler",
            severity=0.5,
            original_start_s=10.0,
            original_end_s=11.0,
            rendered_start_s=10.0,
            rendered_end_s=10.0,
            word_indices=[],
        )


def test_pair_requires_flaws_sorted_by_rendered_start() -> None:
    """Pair flaw order follows the rendered recording timeline."""
    later = make_flaw(rendered_start_s=12.0, rendered_end_s=13.0)
    earlier = make_flaw(rendered_start_s=10.0, rendered_end_s=11.0)
    with pytest.raises(ValidationError, match="sorted by rendered_start_s"):
        Pair(
            ideal_recording_id="ideal-1",
            flawed_recording_id="flawed-1",
            flaws=[later, earlier],
        )


def test_recording_collection_validates_parent_existence_and_kind() -> None:
    """Derived recordings must reference an ideal in the collection."""
    ideal = make_recording("ideal-1")
    with pytest.raises(ValidationError, match="does not exist"):
        RecordingCollection(
            recordings=[make_recording("flawed-1", "injected", "missing")]
        )
    with pytest.raises(ValidationError, match="must refer to an ideal"):
        RecordingCollection(
            recordings=[
                ideal,
                make_recording("injected-1", "injected", "ideal-1"),
                make_recording("human-1", "human_flawed", "injected-1"),
            ]
        )


def test_non_ideal_recording_requires_parent() -> None:
    """A non-ideal recording cannot omit its source ideal ID."""
    with pytest.raises(ValidationError, match="require parent_recording_id"):
        make_recording("flawed-1", "injected")


@pytest.mark.parametrize(
    ("kind", "severity_level", "valid"),
    [
        ("ideal", 0, True),
        ("ideal", 1, False),
        ("control", 0, True),
        ("control", 2, False),
        ("injected", 1, True),
        ("injected", 0, False),
        ("human_flawed", 5, True),
        ("human_flawed", 0, False),
    ],
)
def test_recording_severity_matches_kind(
    kind: str,
    severity_level: int,
    valid: bool,
) -> None:
    """Recording severity is constrained by the recording kind."""
    parent_recording_id = None if kind == "ideal" else "ideal-1"
    data = make_recording(kind, kind, parent_recording_id).model_dump()
    data["severity_level"] = severity_level

    if valid:
        Recording.model_validate(data)
    else:
        with pytest.raises(ValidationError, match="severity_level"):
            Recording.model_validate(data)


def test_pair_json_round_trip_preserves_shifted_flaw() -> None:
    """Round-trip a pair and retain distinct source and rendered intervals."""
    transcript = Transcript(
        id="transcript-1",
        title="Sample",
        text="An example transcript.",
        source="test fixture",
        license="test-only",
        language="en",
    )
    pair = Pair(
        ideal_recording_id="ideal-1",
        flawed_recording_id="flawed-1",
        flaws=[make_flaw()],
    )
    with tempfile.TemporaryDirectory(prefix="speechlens-test-", dir=Path.cwd()) as directory:
        path = Path(directory) / "pair.json"
        save_json(pair, path)
        loaded_pair = load_json(Pair, path)
        loaded_transcript_path = Path(directory) / "transcript.json"
        save_json(transcript, loaded_transcript_path)
        loaded_transcript = load_json(Transcript, loaded_transcript_path)

        assert loaded_pair == pair
        assert loaded_pair.flaws[0].original_start_s == 10.0
        assert loaded_pair.flaws[0].original_end_s == 11.0
        assert loaded_pair.flaws[0].rendered_start_s == 10.0
        assert loaded_pair.flaws[0].rendered_end_s == 12.0
        assert loaded_transcript == transcript
        assert path.read_text(encoding="utf-8").endswith("\n")
        assert list(json.loads(path.read_text(encoding="utf-8")))[0] == "flawed_recording_id"