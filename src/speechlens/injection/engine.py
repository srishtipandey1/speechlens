"""Sample-accurate audio flaw transformations and ground-truth labels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import parselmouth
import yaml
from parselmouth import praat

from speechlens.schema import FlawLabel, WordTiming


Region = tuple[int, int]
InjectionResult = tuple[np.ndarray, list[FlawLabel], list[WordTiming]]
_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "injection.yaml"


@dataclass(frozen=True)
class FlawSpec:
    """One flaw request over a half-open range of original word indices."""

    flaw_type: str
    region: Region
    severity: float


@lru_cache(maxsize=1)
def _config() -> dict:
    """Load the shared injection parameters once per process."""
    with _CONFIG_PATH.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def _sample_rate() -> int:
    """Return the project injection sample rate."""
    return int(_config()["sample_rate_hz"])


def _sample_at(seconds: float) -> int:
    """Convert seconds to the nearest sample using the configured rate."""
    return round(seconds * _sample_rate())


def _seconds_at(sample: int) -> float:
    """Convert a sample index to seconds using the configured rate."""
    return sample / _sample_rate()


def _validate(
    audio: np.ndarray,
    word_timings: Sequence[WordTiming],
    region: Region,
    severity: float,
) -> tuple[np.ndarray, int, int]:
    """Validate mono input and return copied float32 audio and region samples."""
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim != 1:
        raise ValueError("audio must be a mono one-dimensional waveform")
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("audio must contain finite, non-empty samples")
    if not math.isfinite(severity) or not 0.0 <= severity <= 1.0:
        raise ValueError("severity must be finite and between zero and one")
    if len(region) != 2:
        raise ValueError("region must be a half-open (start_index, end_index) pair")
    start_index, end_index = region
    if not 0 <= start_index < end_index <= len(word_timings):
        raise ValueError("region indices must select at least one word timing")
    if any(
        previous.start_s > current.start_s
        or previous.end_s > current.start_s
        for previous, current in zip(word_timings, word_timings[1:])
    ):
        raise ValueError("word timings must be monotonic and non-overlapping")

    start_sample = _sample_at(word_timings[start_index].start_s)
    end_sample = _sample_at(word_timings[end_index - 1].end_s)
    if not 0 <= start_sample < end_sample <= waveform.size:
        raise ValueError("selected word region lies outside the audio waveform")
    return np.ascontiguousarray(waveform), start_sample, end_sample


def _no_op(audio: np.ndarray, word_timings: Sequence[WordTiming]) -> InjectionResult:
    """Return a byte-identical audio copy for zero-severity requests."""
    return np.ascontiguousarray(audio).copy(), [], list(word_timings)


def _limit_peak(audio: np.ndarray) -> np.ndarray:
    """Keep finite float32 output strictly below full scale."""
    output = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(output)))
    ceiling = float(_config()["peak_ceiling"])
    if peak >= ceiling:
        output = output * np.float32(ceiling / peak)
    return np.ascontiguousarray(output, dtype=np.float32)


def _splice(audio: np.ndarray, start_sample: int, end_sample: int, replacement: np.ndarray) -> np.ndarray:
    """Replace an exact half-open sample interval."""
    return np.concatenate(
        (audio[:start_sample], replacement, audio[end_sample:])
    ).astype(np.float32, copy=False)


def _make_timing(template: WordTiming, start_sample: int, end_sample: int) -> WordTiming:
    """Build a word timing from integer samples, retaining confidence."""
    if end_sample <= start_sample:
        end_sample = start_sample + 1
    return WordTiming(
        word=template.word,
        start_s=_seconds_at(start_sample),
        end_s=_seconds_at(end_sample),
        confidence=template.confidence,
    )


def _replace_region_timings(
    timings: Sequence[WordTiming],
    region: Region,
    start_sample: int,
    old_end_sample: int,
    new_region_samples: int,
) -> list[WordTiming]:
    """Rescale selected word boundaries and shift every later word by sample delta."""
    start_index, end_index = region
    old_region_samples = old_end_sample - start_sample
    delta_samples = new_region_samples - old_region_samples
    scale = new_region_samples / old_region_samples
    updated: list[WordTiming] = []
    for index, timing in enumerate(timings):
        old_start = _sample_at(timing.start_s)
        old_end = _sample_at(timing.end_s)
        if start_index <= index < end_index:
            new_start = start_sample + round((old_start - start_sample) * scale)
            new_end = start_sample + round((old_end - start_sample) * scale)
            new_start = max(start_sample, min(new_start, start_sample + new_region_samples - 1))
            new_end = min(start_sample + new_region_samples, max(new_end, new_start + 1))
        elif index >= end_index:
            new_start = old_start + delta_samples
            new_end = old_end + delta_samples
        else:
            new_start, new_end = old_start, old_end
        updated.append(_make_timing(timing, new_start, new_end))
    return updated


def _insert_audio(
    audio: np.ndarray,
    boundary_sample: int,
    inserted_audio: np.ndarray,
    timings: Sequence[WordTiming],
    inserted_timings: Sequence[WordTiming] = (),
) -> tuple[np.ndarray, list[WordTiming]]:
    """Insert audio and shift all words at or after its exact sample boundary."""
    amount = inserted_audio.size
    output = np.concatenate(
        (audio[:boundary_sample], inserted_audio, audio[boundary_sample:])
    ).astype(np.float32, copy=False)
    updated: list[WordTiming] = []
    for timing in timings:
        start = _sample_at(timing.start_s)
        end = _sample_at(timing.end_s)
        if start >= boundary_sample:
            start += amount
            end += amount
        elif end > boundary_sample:
            end += amount
        updated.append(_make_timing(timing, start, end))
    for timing in inserted_timings:
        updated.append(timing)
    updated.sort(key=lambda timing: (timing.start_s, timing.end_s))
    return output, updated


def _region_label(
    flaw_type: str,
    severity: float,
    original_start_s: float,
    original_end_s: float,
    rendered_start_sample: int,
    rendered_end_sample: int,
    word_indices: Sequence[int],
    notes: str = "",
) -> FlawLabel:
    """Build a label from source seconds and rendered sample boundaries."""
    return FlawLabel(
        flaw_type=flaw_type,
        severity=severity,
        original_start_s=original_start_s,
        original_end_s=original_end_s,
        rendered_start_s=_seconds_at(rendered_start_sample),
        rendered_end_s=_seconds_at(rendered_end_sample),
        word_indices=list(word_indices),
        notes=notes,
    )


def _praat_manipulation(segment: np.ndarray) -> parselmouth.Data:
    """Create a Praat Manipulation using the verified command interface."""
    settings = _config()["praat"]
    sound = parselmouth.Sound(segment, sampling_frequency=_sample_rate())
    return praat.call(
        sound,
        "To Manipulation",
        float(settings["time_step_s"]),
        float(settings["pitch_floor_hz"]),
        float(settings["pitch_ceiling_hz"]),
    )


def _psola_duration(segment: np.ndarray, rate_factor: float) -> np.ndarray:
    """Apply a constant Praat duration-tier factor with PSOLA resynthesis."""
    manipulation = _praat_manipulation(segment)
    duration_tier = praat.call(manipulation, "Extract duration tier")
    duration_s = segment.size / _sample_rate()
    duration_factor = 1.0 / rate_factor
    praat.call(duration_tier, "Add point", 0.0, duration_factor)
    praat.call(duration_tier, "Add point", duration_s, duration_factor)
    praat.call([manipulation, duration_tier], "Replace duration tier")
    resynthesized = praat.call(manipulation, "Get resynthesis (overlap-add)")
    samples = np.asarray(resynthesized.values[0], dtype=np.float32)
    target_samples = max(1, round(segment.size / rate_factor))
    if samples.size < target_samples:
        samples = np.pad(samples, (0, target_samples - samples.size))
    return np.ascontiguousarray(samples[:target_samples], dtype=np.float32)


def _pitch_points(tier: parselmouth.Data) -> list[tuple[float, float]]:
    """Read time/value pairs from a Praat PitchTier via its real commands."""
    point_count = int(praat.call(tier, "Get number of points"))
    return [
        (
            float(praat.call(tier, "Get time from index", index)),
            float(praat.call(tier, "Get value at index", index)),
        )
        for index in range(1, point_count + 1)
    ]


def _psola_pitch_compress(segment: np.ndarray, contour_scale: float) -> np.ndarray:
    """Compress voiced F0 points toward their median and resynthesize with PSOLA."""
    manipulation = _praat_manipulation(segment)
    pitch_tier = praat.call(manipulation, "Extract pitch tier")
    points = _pitch_points(pitch_tier)
    if not points:
        raise ValueError("selected region has no voiced pitch points for PSOLA")
    median_f0 = float(np.median([value for _, value in points]))
    for index in range(len(points), 0, -1):
        praat.call(pitch_tier, "Remove point", index)
    for time_s, frequency in points:
        flattened = median_f0 + (frequency - median_f0) * contour_scale
        praat.call(pitch_tier, "Add point", time_s, flattened)
    praat.call([manipulation, pitch_tier], "Replace pitch tier")
    resynthesized = praat.call(manipulation, "Get resynthesis (overlap-add)")
    samples = np.asarray(resynthesized.values[0], dtype=np.float32)
    if samples.size < segment.size:
        samples = np.pad(samples, (0, segment.size - samples.size))
    return np.ascontiguousarray(samples[: segment.size], dtype=np.float32)


def _region_times(timings: Sequence[WordTiming], region: Region) -> tuple[float, float]:
    """Get original interval endpoints for a half-open word range."""
    return timings[region[0]].start_s, timings[region[1] - 1].end_s


def pace_fast(
    audio: np.ndarray,
    word_timings: Sequence[WordTiming],
    region: Region,
    severity: float,
    rng: np.random.Generator,
) -> InjectionResult:
    """Increase regional speaking rate using Praat PSOLA duration manipulation."""
    waveform, start_sample, end_sample = _validate(audio, word_timings, region, severity)
    if severity == 0:
        return _no_op(waveform, word_timings)
    original_start, original_end = _region_times(word_timings, region)
    maximum_rate = float(_config()["pace"]["fast_rate_max"])
    rate_factor = 1.0 + (maximum_rate - 1.0) * severity
    segment = waveform[start_sample:end_sample]
    changed = _psola_duration(segment, rate_factor)
    output = _limit_peak(_splice(waveform, start_sample, end_sample, changed))
    timings = _replace_region_timings(
        word_timings, region, start_sample, end_sample, changed.size
    )
    label = _region_label(
        "pace_fast", severity, original_start, original_end,
        start_sample, start_sample + changed.size, range(*region),
        notes=f"PSOLA rate factor {rate_factor:.6f}.",
    )
    return output, [label], timings


def pace_slow(
    audio: np.ndarray,
    word_timings: Sequence[WordTiming],
    region: Region,
    severity: float,
    rng: np.random.Generator,
) -> InjectionResult:
    """Decrease regional speaking rate using Praat PSOLA duration manipulation."""
    waveform, start_sample, end_sample = _validate(audio, word_timings, region, severity)
    if severity == 0:
        return _no_op(waveform, word_timings)
    original_start, original_end = _region_times(word_timings, region)
    minimum_rate = float(_config()["pace"]["slow_rate_min"])
    rate_factor = 1.0 - (1.0 - minimum_rate) * severity
    segment = waveform[start_sample:end_sample]
    changed = _psola_duration(segment, rate_factor)
    output = _limit_peak(_splice(waveform, start_sample, end_sample, changed))
    timings = _replace_region_timings(
        word_timings, region, start_sample, end_sample, changed.size
    )
    label = _region_label(
        "pace_slow", severity, original_start, original_end,
        start_sample, start_sample + changed.size, range(*region),
        notes=f"PSOLA rate factor {rate_factor:.6f}.",
    )
    return output, [label], timings


def _short_gap_boundary(
    word_timings: Sequence[WordTiming], region: Region
) -> tuple[int, int, float]:
    """Choose the shortest eligible natural gap within the selected word range."""
    maximum_gap = float(_config()["long_pause"]["maximum_natural_gap_s"])
    candidates: list[tuple[float, int, int]] = []
    start_index, end_index = region
    for previous_index in range(start_index, min(end_index, len(word_timings) - 1)):
        previous = word_timings[previous_index]
        following = word_timings[previous_index + 1]
        natural_gap = max(0.0, following.start_s - previous.end_s)
        if natural_gap < maximum_gap:
            candidates.append((natural_gap, previous_index, _sample_at(previous.end_s)))
    if not candidates:
        raise ValueError("selected region has no boundary below the natural long-gap threshold")
    natural_gap, previous_index, boundary_sample = min(candidates)
    return previous_index, boundary_sample, natural_gap


def long_pause(
    audio: np.ndarray,
    word_timings: Sequence[WordTiming],
    region: Region,
    severity: float,
    rng: np.random.Generator,
) -> InjectionResult:
    """Insert a silence pause at a word boundary that is not already long."""
    waveform, _, _ = _validate(audio, word_timings, region, severity)
    if severity == 0:
        return _no_op(waveform, word_timings)
    previous_index, boundary_sample, natural_gap = _short_gap_boundary(word_timings, region)
    settings = _config()["long_pause"]
    pause_s = float(settings["minimum_duration_s"]) + severity * (
        float(settings["maximum_duration_s"]) - float(settings["minimum_duration_s"])
    )
    inserted = np.zeros(round(pause_s * _sample_rate()), dtype=np.float32)
    output, timings = _insert_audio(waveform, boundary_sample, inserted, word_timings)
    anchor_s = _seconds_at(boundary_sample)
    label = _region_label(
        "long_pause", severity, anchor_s, anchor_s + 1 / _sample_rate(),
        boundary_sample, boundary_sample + inserted.size,
        (previous_index, previous_index + 1),
        notes=f"Inserted silence at a {natural_gap:.3f} s natural gap.",
    )
    return _limit_peak(output), [label], timings


def monotone(
    audio: np.ndarray,
    word_timings: Sequence[WordTiming],
    region: Region,
    severity: float,
    rng: np.random.Generator,
) -> InjectionResult:
    """Compress the region's F0 contour toward its median using Praat PSOLA."""
    waveform, start_sample, end_sample = _validate(audio, word_timings, region, severity)
    if severity == 0:
        return _no_op(waveform, word_timings)
    original_start, original_end = _region_times(word_timings, region)
    minimum_scale = float(_config()["monotone"]["minimum_contour_scale"])
    contour_scale = 1.0 - severity * (1.0 - minimum_scale)
    changed = _psola_pitch_compress(waveform[start_sample:end_sample], contour_scale)
    output = _limit_peak(_splice(waveform, start_sample, end_sample, changed))
    label = _region_label(
        "monotone", severity, original_start, original_end,
        start_sample, end_sample, range(*region),
        notes=f"F0 deviations scaled by {contour_scale:.6f} toward median F0.",
    )
    return output, [label], list(word_timings)


def volume_dropoff(
    audio: np.ndarray,
    word_timings: Sequence[WordTiming],
    region: Region,
    severity: float,
    rng: np.random.Generator,
) -> InjectionResult:
    """Apply a smooth, severity-scaled downward gain ramp within the region."""
    waveform, start_sample, end_sample = _validate(audio, word_timings, region, severity)
    if severity == 0:
        return _no_op(waveform, word_timings)
    original_start, original_end = _region_times(word_timings, region)
    max_drop_db = float(_config()["volume_dropoff"]["maximum_drop_db"])
    ramp = np.linspace(0.0, 1.0, end_sample - start_sample, dtype=np.float32)
    smooth_ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    gain = np.power(10.0, -max_drop_db * severity * smooth_ramp / 20.0)
    output = waveform.copy()
    output[start_sample:end_sample] *= gain.astype(np.float32)
    label = _region_label(
        "volume_dropoff", severity, original_start, original_end,
        start_sample, end_sample, range(*region),
        notes=f"Smooth gain ramp to {-max_drop_db * severity:.2f} dB.",
    )
    return _limit_peak(output), [label], list(word_timings)


def _steady_voiced_excerpt(
    audio: np.ndarray,
    duration_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose a low-variation voiced excerpt from this speaker's own audio."""
    settings = _config()["praat"]
    sound = parselmouth.Sound(audio, sampling_frequency=_sample_rate())
    pitch = sound.to_pitch_ac(
        time_step=float(settings["time_step_s"]),
        pitch_floor=float(settings["pitch_floor_hz"]),
        pitch_ceiling=float(settings["pitch_ceiling_hz"]),
    )
    frequencies = pitch.selected_array["frequency"]
    frame_count = max(1, math.ceil(duration_samples / (pitch.dx * _sample_rate())))
    candidates: list[tuple[float, int]] = []
    for frame_start in range(0, len(frequencies) - frame_count + 1):
        window = frequencies[frame_start : frame_start + frame_count]
        if np.any(window <= 0.0):
            continue
        start_sample = round((pitch.x1 + frame_start * pitch.dx) * _sample_rate())
        end_sample = start_sample + duration_samples
        if start_sample < 0 or end_sample > audio.size:
            continue
        excerpt = audio[start_sample:end_sample]
        if float(np.sqrt(np.mean(excerpt**2))) < 1e-4:
            continue
        log_f0 = np.log2(window)
        candidates.append((float(np.std(log_f0)), start_sample))
    if not candidates:
        raise ValueError("could not find a steady voiced excerpt for the synthetic filler")
    candidates.sort()
    choice_index = int(rng.integers(0, min(len(candidates), 5)))
    start_sample = candidates[choice_index][1]
    return np.ascontiguousarray(audio[start_sample : start_sample + duration_samples])


def _fade(audio: np.ndarray, fade_samples: int) -> np.ndarray:
    """Apply a short linear fade at both ends of an excerpt."""
    output = audio.copy()
    fade_length = min(fade_samples, output.size // 2)
    if fade_length:
        ramp = np.linspace(0.0, 1.0, fade_length, endpoint=False, dtype=np.float32)
        output[:fade_length] *= ramp
        output[-fade_length:] *= ramp[::-1]
    return output


def filler(
    audio: np.ndarray,
    word_timings: Sequence[WordTiming],
    region: Region,
    severity: float,
    rng: np.random.Generator,
) -> InjectionResult:
    """Insert a lightly pitch-flattened synthetic vowel hesitation from this clip."""
    waveform, _, _ = _validate(audio, word_timings, region, severity)
    if severity == 0:
        return _no_op(waveform, word_timings)
    previous_index, boundary_sample, _ = _short_gap_boundary(word_timings, region)
    settings = _config()["filler"]
    duration_s = float(settings["minimum_duration_s"]) + severity * (
        float(settings["maximum_duration_s"]) - float(settings["minimum_duration_s"])
    )
    duration_samples = round(duration_s * _sample_rate())
    excerpt = _steady_voiced_excerpt(waveform, duration_samples, rng)
    flattened = _psola_pitch_compress(
        excerpt, float(settings["pitch_contour_scale"])
    )
    fade_samples = round(float(settings["fade_duration_s"]) * _sample_rate())
    inserted = _fade(flattened, fade_samples)
    word = str(settings["label_word"])
    filler_timing = WordTiming(
        word=word,
        start_s=_seconds_at(boundary_sample),
        end_s=_seconds_at(boundary_sample + inserted.size),
        confidence=None,
    )
    output, timings = _insert_audio(
        waveform, boundary_sample, inserted, word_timings, (filler_timing,)
    )
    anchor_s = _seconds_at(boundary_sample)
    label = _region_label(
        "filler", severity, anchor_s, anchor_s + 1 / _sample_rate(),
        boundary_sample, boundary_sample + inserted.size, (previous_index,),
        notes="Synthetic hesitation approximated with a pitch-flattened voiced excerpt from this speaker.",
    )
    return _limit_peak(output), [label], timings


def stumble_repeat(
    audio: np.ndarray,
    word_timings: Sequence[WordTiming],
    region: Region,
    severity: float,
    rng: np.random.Generator,
) -> InjectionResult:
    """Repeat one or two preceding word waveforms immediately before the region."""
    waveform, region_start_sample, _ = _validate(audio, word_timings, region, severity)
    if severity == 0:
        return _no_op(waveform, word_timings)
    settings = _config()["stumble_repeat"]
    available = region[0]
    if available < int(settings["minimum_words"]):
        raise ValueError("stumble_repeat needs at least one preceding word")
    max_words = min(int(settings["maximum_words"]), available)
    minimum_words = int(settings["minimum_words"])
    requested_words = (
        max_words
        if rng.random() < severity
        else minimum_words
    )
    repeated_count = min(requested_words, max_words)
    repeated_start_index = region[0] - repeated_count
    repeated_end_index = region[0]
    source_start = _sample_at(word_timings[repeated_start_index].start_s)
    source_end = _sample_at(word_timings[repeated_end_index - 1].end_s)
    repeated_audio = waveform[source_start:source_end].copy()
    inserted_timings = [
        _make_timing(
            timing,
            region_start_sample + _sample_at(timing.start_s) - source_start,
            region_start_sample + _sample_at(timing.end_s) - source_start,
        )
        for timing in word_timings[repeated_start_index:repeated_end_index]
    ]
    output, timings = _insert_audio(
        waveform, region_start_sample, repeated_audio, word_timings, inserted_timings
    )
    original_start = word_timings[repeated_start_index].start_s
    original_end = word_timings[repeated_end_index - 1].end_s
    label = _region_label(
        "stumble_repeat", severity, original_start, original_end,
        region_start_sample, region_start_sample + repeated_audio.size,
        range(repeated_start_index, repeated_end_index),
        notes=f"Repeated {repeated_count} preceding word(s).",
    )
    return _limit_peak(output), [label], timings


_INJECTORS: dict[str, Callable[..., InjectionResult]] = {
    "pace_fast": pace_fast,
    "pace_slow": pace_slow,
    "long_pause": long_pause,
    "monotone": monotone,
    "volume_dropoff": volume_dropoff,
    "filler": filler,
    "stumble_repeat": stumble_repeat,
}


def _coerce_spec(spec: FlawSpec | Mapping[str, object]) -> FlawSpec:
    """Normalize a mapping or dataclass to a validated flaw request."""
    if isinstance(spec, FlawSpec):
        return spec
    return FlawSpec(
        flaw_type=str(spec["flaw_type"]),
        region=tuple(spec["region"]),  # type: ignore[arg-type]
        severity=float(spec["severity"]),
    )


def _shift_label_rendered(label: FlawLabel, delta_samples: int) -> FlawLabel:
    """Shift an already-rendered label after an earlier source-region edit."""
    return label.model_copy(
        update={
            "rendered_start_s": _seconds_at(_sample_at(label.rendered_start_s) + delta_samples),
            "rendered_end_s": _seconds_at(_sample_at(label.rendered_end_s) + delta_samples),
        }
    )


def apply_flaws(
    audio: np.ndarray,
    word_timings: Sequence[WordTiming],
    flaw_specs: Sequence[FlawSpec | Mapping[str, object]],
    seed: int,
) -> tuple[np.ndarray, list[WordTiming], list[FlawLabel]]:
    """Apply non-overlapping original-word regions from end to start.

    Regions use half-open indices into the supplied ideal word timings. They
    must not overlap; this keeps original word indices stable when earlier
    edits shift later samples or insert synthetic timing entries.
    """
    current_audio = np.ascontiguousarray(audio, dtype=np.float32)
    current_timings = list(word_timings)
    specs = [_coerce_spec(item) for item in flaw_specs]
    ordered_by_source = sorted(specs, key=lambda item: (item.region[0], item.region[1]))
    for previous, following in zip(ordered_by_source, ordered_by_source[1:]):
        if previous.region[1] > following.region[0]:
            raise ValueError("composed flaw regions must not overlap")

    rng = np.random.default_rng(seed)
    all_labels: list[FlawLabel] = []
    for spec in sorted(specs, key=lambda item: (item.region[0], item.region[1]), reverse=True):
        injector = _INJECTORS.get(spec.flaw_type)
        if injector is None:
            raise ValueError(f"unknown flaw_type: {spec.flaw_type}")
        start_index = spec.region[0]
        if not 0 <= start_index < len(current_timings):
            raise ValueError("composed region no longer addresses a word timing")
        shift_boundary = _sample_at(current_timings[start_index].start_s)
        old_length = current_audio.size
        current_audio, labels, current_timings = injector(
            current_audio,
            current_timings,
            spec.region,
            spec.severity,
            rng,
        )
        delta_samples = current_audio.size - old_length
        if delta_samples:
            all_labels = [
                _shift_label_rendered(label, delta_samples)
                if _sample_at(label.rendered_start_s) >= shift_boundary
                else label
                for label in all_labels
            ]
        all_labels.extend(labels)
    all_labels.sort(key=lambda label: label.rendered_start_s)
    return _limit_peak(current_audio), current_timings, all_labels