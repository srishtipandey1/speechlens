"""Pydantic data models for SpeechLens labels and metadata."""

import json
from pathlib import Path
from typing import Annotated, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictInt, model_validator


NonNegativeTime = Annotated[FiniteFloat, Field(ge=0)]
PositiveInt = Annotated[StrictInt, Field(gt=0)]
SeverityLevel = Annotated[StrictInt, Field(ge=0, le=5)]
FlawSeverity = Annotated[FiniteFloat, Field(ge=0, le=1)]
WordIndex = Annotated[StrictInt, Field(ge=0)]


class SchemaModel(BaseModel):
    """Base model that rejects undeclared fields."""

    model_config = ConfigDict(extra="forbid")


class Transcript(SchemaModel):
    """Transcript text and its provenance metadata."""

    id: str
    title: str
    text: str
    source: str
    license: str
    language: str


class Recording(SchemaModel):
    """Audio recording metadata; no audio is loaded by this model."""

    id: str
    transcript_id: str
    speaker_id: str
    speaker_gender: str | None = None
    kind: Literal["ideal", "injected", "human_flawed", "control"]
    severity_level: SeverityLevel
    audio_path: str
    sample_rate: PositiveInt
    duration_s: Annotated[FiniteFloat, Field(gt=0)]
    source: str
    license: str
    parent_recording_id: str | None = None

    @model_validator(mode="after")
    def validate_recording_rules(self) -> "Recording":
        """Validate recording severity and parent requirements by kind."""
        if self.kind in {"ideal", "control"} and self.severity_level != 0:
            raise ValueError(f"{self.kind} recordings must have severity_level 0")
        if self.kind in {"injected", "human_flawed"} and self.severity_level == 0:
            raise ValueError(f"{self.kind} recordings must have severity_level 1 to 5")
        if self.kind == "ideal" and self.parent_recording_id is not None:
            raise ValueError("ideal recordings cannot have a parent recording")
        if self.kind != "ideal" and not self.parent_recording_id:
            raise ValueError("non-ideal recordings require parent_recording_id")
        return self


class RecordingCollection(SchemaModel):
    """Recording set that validates parent references across its members."""

    recordings: list[Recording]

    @model_validator(mode="after")
    def validate_recording_parents(self) -> "RecordingCollection":
        """Ensure every derived recording points to an ideal in this collection."""
        recordings_by_id = {recording.id: recording for recording in self.recordings}
        if len(recordings_by_id) != len(self.recordings):
            raise ValueError("recording IDs must be unique")

        for recording in self.recordings:
            if recording.kind == "ideal":
                continue
            parent = recordings_by_id.get(recording.parent_recording_id)
            if parent is None:
                raise ValueError(
                    f"parent recording {recording.parent_recording_id!r} does not exist"
                )
            if parent.kind != "ideal":
                raise ValueError("parent_recording_id must refer to an ideal recording")
        return self


class WordTiming(SchemaModel):
    """Time interval for a word in a recording."""

    word: str
    start_s: NonNegativeTime
    end_s: NonNegativeTime
    confidence: Annotated[FiniteFloat, Field(ge=0, le=1)] | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> "WordTiming":
        """Require a positive-length interval."""
        if self.start_s >= self.end_s:
            raise ValueError("start_s must be less than end_s")
        return self


class FlawLabel(SchemaModel):
    """A labeled flaw with intervals in both source and rendered timelines."""

    flaw_type: Literal[
        "pace_fast",
        "pace_slow",
        "long_pause",
        "monotone",
        "volume_dropoff",
        "filler",
        "stumble_repeat",
    ]
    severity: FlawSeverity
    original_start_s: NonNegativeTime
    original_end_s: NonNegativeTime
    rendered_start_s: NonNegativeTime
    rendered_end_s: NonNegativeTime
    word_indices: list[WordIndex]
    notes: str = ""

    @model_validator(mode="after")
    def validate_intervals(self) -> "FlawLabel":
        """Require positive-length intervals on both timelines."""
        if self.original_start_s >= self.original_end_s:
            raise ValueError("original_start_s must be less than original_end_s")
        if self.rendered_start_s >= self.rendered_end_s:
            raise ValueError("rendered_start_s must be less than rendered_end_s")
        return self


class Pair(SchemaModel):
    """Link an ideal recording to a flawed recording and its flaw labels."""

    ideal_recording_id: str
    flawed_recording_id: str
    flaws: list[FlawLabel]

    @model_validator(mode="after")
    def validate_pair(self) -> "Pair":
        """Reject self-pairs and flaws not ordered by rendered start time."""
        if self.ideal_recording_id == self.flawed_recording_id:
            raise ValueError("ideal and flawed recording IDs must differ")
        if any(
            earlier.rendered_start_s > later.rendered_start_s
            for earlier, later in zip(self.flaws, self.flaws[1:])
        ):
            raise ValueError("flaws must be sorted by rendered_start_s")
        return self


ModelT = TypeVar("ModelT", bound=BaseModel)


def save_json(model: BaseModel, path: str | Path) -> None:
    """Serialize a Pydantic model with deterministic object-key ordering."""
    serialized = json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    Path(path).write_text(f"{serialized}\n", encoding="utf-8")


def load_json(model_type: type[ModelT], path: str | Path) -> ModelT:
    """Load JSON from disk and validate it as the requested Pydantic model."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return model_type.model_validate(data)