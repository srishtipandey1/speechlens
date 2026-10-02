"""Speaker-normalized frame, word, phrase, pause, and modulation features."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import librosa
import numpy as np
import pandas as pd
import parselmouth
from scipy.signal import find_peaks

from speechlens.schema import WordTiming


SAMPLE_RATE_HZ = 16000
_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "features.yaml"
_VOWEL_GROUPS = re.compile(r"[aeiouy]+", re.IGNORECASE)
_TRAILING_PUNCTUATION = re.compile(r"[.!?;:][\"')\]]*$")
MFCC_COLUMNS = tuple(f"mfcc_{index}" for index in range(1, 14))
DELTA_MFCC_COLUMNS = tuple(f"delta_mfcc_{index}" for index in range(1, 14))


@dataclass(frozen=True)
class FeatureBundle:
    """Aligned frame/word/phrase tables plus pause and modulation summaries."""

    frames: pd.DataFrame
    words: pd.DataFrame
    phrases: pd.DataFrame
    pauses: pd.DataFrame
    summary: dict[str, float | int]
    modulation: dict[str, float]


@lru_cache(maxsize=1)
def _config() -> dict:
    """Load reproducible acoustic feature parameters."""
    with _CONFIG_PATH.open(encoding="utf-8") as config_file:
        import yaml

        return yaml.safe_load(config_file)


def hz_to_semitones(frequency_hz: np.ndarray | float, reference_hz: float) -> np.ndarray:
    """Convert positive F0 values to semitones relative to a reference F0."""
    if reference_hz <= 0:
        raise ValueError("reference_hz must be positive")
    frequencies = np.asarray(frequency_hz, dtype=np.float64)
    result = np.full(frequencies.shape, np.nan, dtype=np.float64)
    voiced = frequencies > 0
    result[voiced] = 12.0 * np.log2(frequencies[voiced] / reference_hz)
    return result


def normalize_to_median(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Subtract the median over selected finite frames; return zeros if none."""
    series = np.asarray(values, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool) & np.isfinite(series)
    normalized = np.zeros(series.shape, dtype=np.float64)
    if np.any(selected):
        normalized = series - float(np.median(series[selected]))
    normalized[~np.isfinite(series)] = np.nan
    return normalized


def _rms_frames(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Compute centered frame RMS on the configured hop grid."""
    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )
    return np.asarray(rms[0], dtype=np.float64)


def _praat_frame_track(
    values: np.ndarray,
    track_x1: float,
    track_dx: float,
    frame_times: np.ndarray,
    default: float = np.nan,
) -> np.ndarray:
    """Map a Praat uniformly sampled track to frames by nearest track frame."""
    track_values = np.asarray(values, dtype=np.float64).reshape(-1)
    result = np.full(frame_times.shape, default, dtype=np.float64)
    if not track_values.size or track_dx <= 0:
        return result
    indices = np.rint((frame_times - track_x1) / track_dx).astype(np.int64)
    valid = (indices >= 0) & (indices < track_values.size)
    result[valid] = track_values[indices[valid]]
    return result


def _correct_isolated_octave_jumps(
    frequencies_hz: np.ndarray,
    target_semitones: float,
    tolerance_semitones: float,
    neighbor_tolerance_semitones: float,
) -> np.ndarray:
    """Correct isolated one-frame octave slips bracketed by stable voiced frames.

    A sustained octave change is retained; only a single-frame jump close to a
    configured octave interval, with mutually stable neighbors, is corrected.
    """
    corrected = np.asarray(frequencies_hz, dtype=np.float64).copy()
    original = corrected.copy()
    for index in range(1, len(original) - 1):
        previous, current, following = original[index - 1 : index + 2]
        if min(previous, current, following) <= 0:
            continue
        neighbor_delta = abs(12.0 * np.log2(previous / following))
        if neighbor_delta > neighbor_tolerance_semitones:
            continue
        jump = 12.0 * np.log2(current / previous)
        if abs(abs(jump) - target_semitones) <= tolerance_semitones:
            corrected[index] = np.sqrt(previous * following)
    return corrected


def modulation_spectrum(
    signal: np.ndarray,
    sample_rate_hz: float,
    frequency_band_hz: tuple[float, float],
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Return integrated band power, peak frequency, FFT frequencies and power.

    For mean-centered samples x[n] and Hann window w[n], power is
    ``|rFFT((x - mean(x)) * w)|**2 / sum(w)**2``. Band power is the sum of bins
    inside the inclusive configured frequency range.
    """
    values = np.asarray(signal, dtype=np.float64)
    if values.size < 2 or sample_rate_hz <= 0:
        return 0.0, 0.0, np.empty(0), np.empty(0)
    if not np.isfinite(values).all():
        valid = np.isfinite(values)
        if valid.sum() < 2:
            return 0.0, 0.0, np.empty(0), np.empty(0)
        values = np.interp(np.arange(values.size), np.flatnonzero(valid), values[valid])
    centered = values - float(np.mean(values))
    window = np.hanning(centered.size)
    spectrum = np.fft.rfft(centered * window)
    power = (np.abs(spectrum) ** 2) / max(float(np.sum(window) ** 2), 1e-30)
    frequencies = np.fft.rfftfreq(centered.size, d=1.0 / sample_rate_hz)
    low_hz, high_hz = frequency_band_hz
    band_mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    if not np.any(band_mask):
        return 0.0, 0.0, frequencies, power
    band_indices = np.flatnonzero(band_mask)
    band_power = float(np.sum(power[band_mask]))
    peak_frequency = float(frequencies[band_indices[np.argmax(power[band_mask])]])
    return band_power, peak_frequency, frequencies, power


def _assign_word_indices(
    frame_times: np.ndarray,
    word_timings: Sequence[WordTiming],
) -> np.ndarray:
    """Assign frame centers to half-open word timing intervals, else -1."""
    indices = np.full(frame_times.shape, -1, dtype=np.int64)
    for word_index, timing in enumerate(word_timings):
        selected = (frame_times >= timing.start_s) & (frame_times < timing.end_s)
        indices[selected & (indices == -1)] = word_index
    return indices


def _speech_phrases(
    word_timings: Sequence[WordTiming],
    transcript_text: str,
) -> list[tuple[int, int, str]]:
    """Split indexed transcript words at terminal punctuation."""
    transcript_words = transcript_text.split()
    word_count = min(len(word_timings), len(transcript_words))
    phrases: list[tuple[int, int, str]] = []
    phrase_start = 0
    for word_index in range(word_count):
        if _TRAILING_PUNCTUATION.search(transcript_words[word_index]):
            phrases.append((phrase_start, word_index + 1, " ".join(transcript_words[phrase_start : word_index + 1])))
            phrase_start = word_index + 1
    if phrase_start < word_count:
        phrases.append((phrase_start, word_count, " ".join(transcript_words[phrase_start:word_count])))
    return phrases


def _energy_syllable_peaks(
    intensity_raw_dbfs: np.ndarray,
    mask: np.ndarray,
    sample_rate_hz: float,
) -> int:
    """Count prominent energy-envelope nuclei in a word/phrase interval."""
    envelope = np.asarray(intensity_raw_dbfs, dtype=np.float64)
    active_mask = np.asarray(mask, dtype=bool) & np.isfinite(envelope)
    selected = envelope[active_mask]
    if selected.size < 3:
        return int(selected.size > 0)
    config = _config()
    distance = max(
        1,
        round(float(config["syllable_peak_minimum_distance_s"]) * sample_rate_hz),
    )
    peak_envelope = envelope.copy()
    peak_envelope[~active_mask] = float(np.min(selected)) - 2.0 * float(
        config["syllable_peak_prominence_db"]
    )
    peaks, _ = find_peaks(
        peak_envelope,
        prominence=float(config["syllable_peak_prominence_db"]),
        distance=distance,
    )
    return max(1, int(np.count_nonzero(active_mask[peaks])))


def _word_group_count(word: str) -> int:
    """Return a simple vowel-group syllable-count cross-check for a word."""
    return max(1, len(_VOWEL_GROUPS.findall(word))) if re.search(r"[A-Za-z]", word) else 0


def _build_word_table(
    frames: pd.DataFrame,
    word_timings: Sequence[WordTiming],
    transcript_text: str,
    frame_length_s: float,
    hop_s: float,
) -> pd.DataFrame:
    """Aggregate frame features per aligned word and cross-check syllable counts."""
    transcript_words = transcript_text.split()
    rows: list[dict[str, float | int | str]] = []
    for word_index, timing in enumerate(word_timings):
        mask = frames["word_index"].to_numpy() == word_index
        word = transcript_words[word_index] if word_index < len(transcript_words) else timing.word
        active = mask & frames["speech_activity"].to_numpy(dtype=bool)
        voiced = mask & frames["voiced"].to_numpy(dtype=bool)
        nuclei = _energy_syllable_peaks(
            frames["intensity_raw_dbfs"].to_numpy(),
            active,
            1.0 / hop_s,
        )
        vowel_groups = _word_group_count(word)
        speech_duration_s = max(float(np.count_nonzero(active)) * hop_s, hop_s)
        local_f0 = frames.loc[voiced, "f0_semitones"].to_numpy(dtype=np.float64)
        local_intensity = frames.loc[active, "intensity_db"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "word_index": word_index,
                "word": word,
                "start_s": timing.start_s,
                "end_s": timing.end_s,
                "duration_s": timing.end_s - timing.start_s,
                "speech_duration_s": speech_duration_s,
                "syllable_nuclei": nuclei,
                "vowel_group_count": vowel_groups,
                "syllable_count_abs_difference": abs(nuclei - vowel_groups),
                "articulation_rate_sps": nuclei / speech_duration_s,
                "words_per_second": 1.0 / max(timing.end_s - timing.start_s, hop_s),
                "voiced_fraction": float(np.count_nonzero(voiced) / max(1, np.count_nonzero(active))),
                "f0_range_semitones": float(np.ptp(local_f0)) if local_f0.size else 0.0,
                "f0_std_semitones": float(np.std(local_f0)) if local_f0.size else 0.0,
                "intensity_range_db": float(np.ptp(local_intensity)) if local_intensity.size else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _build_phrase_table(
    frames: pd.DataFrame,
    word_table: pd.DataFrame,
    word_timings: Sequence[WordTiming],
    transcript_text: str,
    pauses: pd.DataFrame,
    hop_s: float,
) -> pd.DataFrame:
    """Aggregate acoustic and delivery features per punctuation-delimited phrase."""
    rows: list[dict[str, float | int | str]] = []
    for phrase_index, (start_index, end_index, text) in enumerate(
        _speech_phrases(word_timings, transcript_text)
    ):
        start_s = word_timings[start_index].start_s
        end_s = word_timings[end_index - 1].end_s
        word_mask = frames["word_index"].between(start_index, end_index - 1).to_numpy()
        active = word_mask & frames["speech_activity"].to_numpy(dtype=bool)
        voiced = word_mask & frames["voiced"].to_numpy(dtype=bool)
        f0_values = frames.loc[voiced, "f0_semitones"].to_numpy(dtype=np.float64)
        intensity_values = frames.loc[active, "intensity_db"].to_numpy(dtype=np.float64)
        phrase_words = word_table.iloc[start_index:end_index]
        active_duration = max(float(np.count_nonzero(active)) * hop_s, hop_s)
        phrase_duration = max(end_s - start_s, hop_s)
        within_phrase = (pauses["start_s"] >= start_s) & (pauses["end_s"] <= end_s)
        trailing_punctuation_pause = (
            (pauses["boundary_after_word_index"] == end_index - 1)
            & (pauses["boundary_type"] == "punctuation_boundary")
        )
        phrase_pauses = pauses[within_phrase | trailing_punctuation_pause]
        phrase_pause_duration = (
            float(phrase_pauses["duration_s"].sum()) if not phrase_pauses.empty else 0.0
        )
        trailing_pause_duration = float(
            phrase_pauses.loc[trailing_punctuation_pause.loc[phrase_pauses.index], "duration_s"].sum()
        ) if not phrase_pauses.empty else 0.0
        phrase_total_duration = phrase_duration + trailing_pause_duration
        rows.append(
            {
                "phrase_index": phrase_index,
                "first_word_index": start_index,
                "end_word_index_exclusive": end_index,
                "text": text,
                "start_s": start_s,
                "end_s": end_s,
                "word_count": end_index - start_index,
                "syllable_nuclei": int(phrase_words["syllable_nuclei"].sum()),
                "vowel_group_count": int(phrase_words["vowel_group_count"].sum()),
                "articulation_rate_sps": float(phrase_words["syllable_nuclei"].sum()) / active_duration,
                "words_per_second": (end_index - start_index) / phrase_duration,
                "pause_count": len(phrase_pauses),
                "pause_duration_s": phrase_pause_duration,
                "pause_ratio": phrase_pause_duration / max(phrase_total_duration, hop_s),
                "f0_range_semitones": float(np.ptp(f0_values)) if f0_values.size else 0.0,
                "f0_std_semitones": float(np.std(f0_values)) if f0_values.size else 0.0,
                "intensity_range_db": float(np.ptp(intensity_values)) if intensity_values.size else 0.0,
                "voiced_fraction": float(np.count_nonzero(voiced) / max(1, np.count_nonzero(active))),
            }
        )
    return pd.DataFrame(rows)


def _build_pauses(
    word_timings: Sequence[WordTiming],
    transcript_text: str,
) -> pd.DataFrame:
    """List inter-word pauses and classify punctuation vs mid-phrase boundaries."""
    transcript_words = transcript_text.split()
    minimum_duration = float(_config()["pause_minimum_duration_s"])
    rows: list[dict[str, float | int | str]] = []
    for boundary_index, (previous, following) in enumerate(zip(word_timings, word_timings[1:])):
        duration_s = following.start_s - previous.end_s
        if duration_s < minimum_duration:
            continue
        previous_text = transcript_words[boundary_index] if boundary_index < len(transcript_words) else previous.word
        boundary_type = "punctuation_boundary" if _TRAILING_PUNCTUATION.search(previous_text) else "mid_phrase_boundary"
        rows.append(
            {
                "start_s": previous.end_s,
                "end_s": following.start_s,
                "duration_s": duration_s,
                "boundary_after_word_index": boundary_index,
                "boundary_type": boundary_type,
            }
        )
    return pd.DataFrame(
        rows,
        columns=("start_s", "end_s", "duration_s", "boundary_after_word_index", "boundary_type"),
    )


def extract_frame_features(
    audio: np.ndarray,
    word_timings: Sequence[WordTiming],
    sample_rate_hz: int = SAMPLE_RATE_HZ,
) -> pd.DataFrame:
    """Extract deterministic speaker-normalized acoustic features every 10 ms."""
    config = _config()
    if sample_rate_hz != int(config["sample_rate_hz"]):
        raise ValueError(f"audio must be {config['sample_rate_hz']} Hz")
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("audio must be finite, non-empty, mono float32")

    hop_length = round(float(config["hop_s"]) * sample_rate_hz)
    frame_length = round(float(config["frame_length_s"]) * sample_rate_hz)
    n_fft = int(config["n_fft"])
    hop_s = hop_length / sample_rate_hz
    frame_times = np.arange(0, waveform.size // hop_length + 1, dtype=np.float64) * hop_s
    rms = _rms_frames(waveform, frame_length, hop_length)
    frame_count = min(frame_times.size, rms.size)
    frame_times = frame_times[:frame_count]
    rms = rms[:frame_count]
    raw_intensity_dbfs = 20.0 * np.log10(rms + float(config["energy_epsilon"]))
    speech_activity = raw_intensity_dbfs >= float(config["speech_activity_threshold_dbfs"])

    praat_config = config["praat"]
    sound = parselmouth.Sound(waveform, sampling_frequency=sample_rate_hz)
    pitch = sound.to_pitch_ac(
        time_step=hop_s,
        pitch_floor=float(praat_config["pitch_floor_hz"]),
        pitch_ceiling=float(praat_config["pitch_ceiling_hz"]),
    )
    f0_hz = _praat_frame_track(
        pitch.selected_array["frequency"], pitch.x1, pitch.dx, frame_times, default=0.0
    )
    octave_config = config["octave_correction"]
    f0_hz = _correct_isolated_octave_jumps(
        f0_hz,
        float(octave_config["target_semitones"]),
        float(octave_config["tolerance_semitones"]),
        float(octave_config["neighbor_tolerance_semitones"]),
    )
    voiced = f0_hz > 0.0
    median_f0_hz = float(np.median(f0_hz[voiced])) if np.any(voiced) else 0.0
    f0_semitones = np.zeros(frame_count, dtype=np.float64)
    if median_f0_hz > 0.0:
        f0_semitones[voiced] = 12.0 * np.log2(f0_hz[voiced] / median_f0_hz)
    intensity_db = normalize_to_median(raw_intensity_dbfs, speech_activity)

    intensity = sound.to_intensity(
        time_step=hop_s,
        minimum_pitch=float(praat_config["intensity_minimum_pitch_hz"]),
    )
    intensity_praat = _praat_frame_track(
        intensity.values[0], intensity.x1, intensity.dx, frame_times
    )
    harmonicity = sound.to_harmonicity_ac(
        time_step=hop_s,
        minimum_pitch=float(praat_config["harmonicity_minimum_pitch_hz"]),
        silence_threshold=float(praat_config["harmonicity_silence_threshold"]),
        periods_per_window=float(praat_config["harmonicity_periods_per_window"]),
    )
    hnr_db = _praat_frame_track(
        harmonicity.values[0], harmonicity.x1, harmonicity.dx, frame_times
    )
    hnr_db[(~speech_activity) | (hnr_db <= -199.0)] = np.nan

    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sample_rate_hz,
        n_mfcc=int(config["mfcc_count"]),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=frame_length,
        n_mels=int(config["mel_bands"]),
        center=True,
    )
    mfcc = mfcc[:, :frame_count]
    delta_width = min(int(config["mfcc_delta_width"]), frame_count if frame_count % 2 else frame_count - 1)
    if delta_width >= 3:
        delta_mfcc = librosa.feature.delta(mfcc, width=delta_width, mode="interp")
    else:
        delta_mfcc = np.zeros_like(mfcc)

    spectrum = np.abs(
        librosa.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=frame_length,
            center=True,
        )
    )[:, :frame_count]
    positive_difference = np.maximum(np.diff(spectrum, axis=1, prepend=spectrum[:, :1]), 0.0)
    spectral_flux = np.sqrt(np.sum(positive_difference**2, axis=0))
    word_index = _assign_word_indices(frame_times, word_timings)

    data: dict[str, np.ndarray] = {
        "time_s": frame_times,
        "f0_hz": f0_hz,
        "voiced": voiced,
        "f0_semitones": f0_semitones,
        "intensity_raw_dbfs": raw_intensity_dbfs,
        "intensity_db": intensity_db,
        "spectral_flux": spectral_flux,
        "hnr_db": hnr_db,
        "speech_activity": speech_activity,
        "word_index": word_index,
        "intensity_praat_db": intensity_praat,
        "frame_rms": rms,
    }
    data.update({name: mfcc[index] for index, name in enumerate(MFCC_COLUMNS)})
    data.update({name: delta_mfcc[index] for index, name in enumerate(DELTA_MFCC_COLUMNS)})
    return pd.DataFrame(data)


def extract_recording_features(
    audio: np.ndarray,
    word_timings: Sequence[WordTiming],
    transcript_text: str,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
) -> FeatureBundle:
    """Build aligned frame/word/phrase/pause tables and modulation summaries."""
    frames = extract_frame_features(audio, word_timings, sample_rate_hz)
    config = _config()
    hop_s = float(config["hop_s"])
    word_table = _build_word_table(
        frames,
        word_timings,
        transcript_text,
        float(config["frame_length_s"]),
        hop_s,
    )
    pauses = _build_pauses(word_timings, transcript_text)
    phrase_table = _build_phrase_table(
        frames,
        word_table,
        word_timings,
        transcript_text,
        pauses,
        hop_s,
    )
    total_duration_s = len(audio) / sample_rate_hz
    pause_duration_s = float(pauses["duration_s"].sum()) if not pauses.empty else 0.0
    speech_mask = frames["speech_activity"].to_numpy(dtype=bool)
    voiced_mask = frames["voiced"].to_numpy(dtype=bool)
    energy_band = tuple(config["modulation_bands_hz"]["energy_syllable_rate"])
    f0_band = tuple(config["modulation_bands_hz"]["f0_monotone"])
    energy_power, energy_peak, _, _ = modulation_spectrum(
        frames["frame_rms"].to_numpy(), 1.0 / hop_s, energy_band
    )
    f0_series = frames["f0_semitones"].to_numpy(dtype=np.float64)
    f0_series[~voiced_mask] = np.nan
    f0_power, f0_peak, _, _ = modulation_spectrum(f0_series, 1.0 / hop_s, f0_band)
    modulation = {
        "energy_power_3_6_hz": energy_power,
        "energy_peak_modulation_hz": energy_peak,
        "f0_power_0_5_3_hz": f0_power,
        "f0_peak_modulation_hz": f0_peak,
    }
    speech_frame_count = int(np.count_nonzero(speech_mask))
    summary: dict[str, float | int] = {
        "duration_s": total_duration_s,
        "word_count": len(word_timings),
        "pause_count": len(pauses),
        "pause_duration_s": pause_duration_s,
        "pause_ratio": pause_duration_s / max(total_duration_s, hop_s),
        "voiced_fraction": float(np.count_nonzero(voiced_mask) / max(1, speech_frame_count)),
        "articulation_rate_sps": (
            float(word_table["syllable_nuclei"].sum())
            / max(speech_frame_count * hop_s, hop_s)
        ),
        "words_per_second": len(word_timings) / max(total_duration_s, hop_s),
    }
    return FeatureBundle(frames, word_table, phrase_table, pauses, summary, modulation)