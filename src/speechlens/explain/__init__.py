"""Deterministic structured and human-readable causal explanations."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence


_SUGGESTIONS = {
	"pace_fast": "Slow down and leave a clear pause at phrase boundaries.",
	"pace_slow": "Keep the pace moving while preserving clear articulation.",
	"long_pause": "Reduce the pause and continue smoothly into the next phrase.",
	"monotone": "Vary pitch emphasis across the key words in this phrase.",
	"volume_dropoff": "Maintain a steadier volume through the end of the phrase.",
	"filler": "Replace the filler sound with a brief, intentional pause.",
	"stumble_repeat": "Pause briefly, then continue without repeating the previous word.",
	"other_deviation": "Rehearse this phrase while keeping its timing and delivery consistent.",
}


def _time_text(value: float) -> str:
	"""Render a timestamp with one decimal place and stable zero padding."""
	minutes, seconds = divmod(max(0.0, value), 60.0)
	return f"{int(minutes):02d}:{seconds:04.1f}"


def _phrase_text(words: Sequence[str]) -> str:
	"""Return a quoted, whitespace-normalized phrase."""
	return " ".join(str(word) for word in words).strip()


def explanation_record(region: Mapping, *, mode: str | None = None) -> dict:
	"""Build a structured record with deterministic causal explanation text."""
	flaw_type = str(region["type"])
	start_s = float(region["start_s"])
	end_s = float(region["end_s"])
	words = [str(word) for word in region.get("words", [])]
	features = region.get("features", {})
	if flaw_type in {"pace_fast", "pace_slow"}:
		peak_ratio = features.get("peak_rate_ratio")
		if peak_ratio is None and mode == "paired" and "score" in features:
			peak_score = float(features["score"])
			peak_ratio = 1.0 - peak_score if flaw_type == "pace_fast" else 1.0 + peak_score
		if peak_ratio is not None or "rate_ratio" in features:
			observed = float(peak_ratio if peak_ratio is not None else features["rate_ratio"])
			expected = 1.0
			z_or_ratio = observed
			formula = "peak-scoring local participant/ideal word-duration ratio"
			observed_value = f"{observed:.2f} duration ratio at peak score"
			expected_value = "1.00 relative to the ideal"
			if observed < expected:
				causal = f"the peak-scoring duration ratio was {observed:.2f}, below 1.00 expected"
				consequence = "this section was rushed"
			elif observed > expected:
				causal = f"the peak-scoring duration ratio was {observed:.2f}, above 1.00 expected"
				consequence = "this section was drawn out"
			else:
				causal = f"the peak-scoring duration ratio was {observed:.2f}, matching 1.00 expected"
				consequence = "timing differs from the reference here"
		else:
			z_or_ratio = float(features.get("peak_rate_z", features.get("rate_z", 0.0)))
			formula = "log(observed_rate_sps / expected_rate_sps) / paired_rate_scale"
			if region.get("observed_rate_sps") is None or region.get("expected_rate_sps") is None:
				observed = None
				expected = None
				observed_value = "Peak rate unavailable"
				expected_value = "Expected rate unavailable"
				causal = "the peak-scoring rate is unavailable"
				consequence = "timing differs from the reference here"
			else:
				observed = float(region["observed_rate_sps"])
				expected = float(region["expected_rate_sps"])
				observed_value = f"{observed:.1f} syllables/s"
				expected_value = f"{expected:.1f} syllables/s"
				if observed < expected:
					causal = f"the peak-scoring rate was {observed:.1f} syllables/s, below {expected:.1f} expected (z={z_or_ratio:+.1f})"
					consequence = "this section was rushed"
				elif observed > expected:
					causal = f"the peak-scoring rate was {observed:.1f} syllables/s, above {expected:.1f} expected (z={z_or_ratio:+.1f})"
					consequence = "this section was drawn out"
				else:
					causal = f"the peak-scoring rate was {observed:.1f} syllables/s, matching {expected:.1f} expected"
					consequence = "timing differs from the reference here"
	elif flaw_type == "long_pause":
		observed = float(features.get("pause_excess_s", 0.0))
		expected = 0.0
		z_or_ratio = observed
		formula = "participant_pause_s - ideal_pause_s"
		observed_value = f"{observed:.2f} s excess pause"
		expected_value = "no excess pause"
		causal = f"{observed:.2f} s beyond the expected pause"
		consequence = "the delivery stalled here"
	elif flaw_type == "monotone":
		observed = float(features.get("f0_std_ratio", 1.0))
		expected = 1.0
		z_or_ratio = observed
		formula = "participant_f0_std_semitones / ideal_f0_std_semitones"
		observed_value = f"{observed:.2f} pitch-variation ratio"
		expected_value = "1.00 relative to the ideal"
		causal = f"pitch variation was {observed:.2f}x the ideal"
		consequence = "the phrase sounded comparatively monotone"
	elif flaw_type == "volume_dropoff":
		observed = float(features.get(
			"intensity_drop_db",
			float(region.get("observed_intensity_db", 0.0))
			- float(region.get("expected_intensity_db", 0.0)),
		))
		expected = 0.0
		z_or_ratio = observed
		formula = "first-word speech level dB - last-word speech level dB"
		observed_value = f"{observed:.1f} dB downward level change"
		expected_value = "0.0 dB downward change"
		causal = f"speech level fell by {observed:.1f} dB across the window"
		consequence = "the phrase lost vocal presence"
	elif flaw_type in {"filler", "stumble_repeat"}:
		observed = float(features.get("score", features.get("repeat_score", 0.0)))
		if flaw_type == "filler":
			observed = max(end_s - start_s, 0.0)
			expected = 0.0
			formula = "duration of speech-active voiced frames outside aligned words"
			observed_value = f"{observed:.2f} s non-lexical voiced segment"
			expected_value = "no non-lexical voiced segment"
			causal = f"a voiced non-lexical segment lasted {observed:.2f} s"
			z_or_ratio = observed
			consequence = "a filler interrupted the phrase"
		else:
			expected = 0.0
			formula = "acoustic repeated-word similarity score"
			observed_value = f"{observed:.2f} repeat score"
			expected_value = "below the configured repeat threshold"
			causal = f"the repeated-word similarity score was {observed:.2f}"
			z_or_ratio = observed
			consequence = "a word sequence was repeated"
	else:
		observed = float(region.get("max_deviation", 0.0))
		expected = 0.0
		z_or_ratio = observed
		formula = "maximum absolute normalized feature deviation in region"
		observed_value = f"{observed:.1f} composite deviation"
		expected_value = "0.0 expected deviation"
		causal = f"the strongest feature deviation was {observed:.1f}"
		consequence = "delivery differed from the expected pattern"

	phrase = _phrase_text(words)
	wording = f' \'{phrase}\'' if phrase else ""
	sentence = (
		f"{_time_text(start_s)}-{_time_text(end_s)}{wording}: "
		f"{causal}, so {consequence}."
	)
	return {
		"start_s": start_s,
		"end_s": end_s,
		"words": words,
		"type": flaw_type,
		"observed_value": observed_value,
		"observed_numeric": float(observed) if observed is not None else None,
		"expected_value": expected_value,
		"expected_numeric": float(expected) if expected is not None else None,
		"z_score_or_ratio": float(z_or_ratio),
		"formula": formula,
		"severity": float(region.get("severity", 0.0)),
		"confidence": float(region.get("confidence", 0.0)),
		"sentence": sentence,
		"suggestion": _SUGGESTIONS.get(flaw_type, _SUGGESTIONS["other_deviation"]),
		"rendered_text": (
			f"{sentence} "
			f"{_SUGGESTIONS.get(flaw_type, _SUGGESTIONS['other_deviation'])}"
		),
	}


def render_explanation(record: Mapping) -> str:
	"""Render one explanation as stable prose and a practical suggestion."""
	return f"{record['sentence']} {record['suggestion']}"


def render_explanations_json(records: Sequence[Mapping]) -> str:
	"""Serialize structured explanation records with byte-stable JSON text."""
	return json.dumps(
		list(records),
		ensure_ascii=False,
		sort_keys=True,
		separators=(",", ":"),
	)


__all__ = [
	"explanation_record",
	"render_explanation",
	"render_explanations_json",
]