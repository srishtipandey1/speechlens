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


def explanation_record(region: Mapping) -> dict:
	"""Build a structured record with deterministic causal explanation text."""
	flaw_type = str(region["type"])
	start_s = float(region["start_s"])
	end_s = float(region["end_s"])
	words = [str(word) for word in region.get("words", [])]
	features = region.get("features", {})
	if flaw_type in {"pace_fast", "pace_slow"}:
		if "rate_ratio" in features:
			observed = float(features["rate_ratio"])
			expected = 1.0
			z_or_ratio = observed
			formula = "local participant/ideal word-duration ratio / passage-wide rate ratio"
			observed_value = f"{observed:.2f} normalized duration ratio"
			expected_value = "1.00 normalized ideal duration"
			causal = f"the normalized local duration ratio was {observed:.2f} vs 1.00 expected"
		else:
			observed = float(region.get("observed_rate_sps", 0.0))
			expected = float(region.get("expected_rate_sps", 0.0))
			z_or_ratio = float(features.get("rate_z", 0.0))
			formula = "log(observed_rate_sps / expected_rate_sps) / paired_rate_scale"
			observed_value = f"{observed:.1f} syllables/s"
			expected_value = f"{expected:.1f} syllables/s"
			causal = f"{observed:.1f} syllables/s vs {expected:.1f} expected (z={z_or_ratio:+.1f})"
		consequence = "this section was rushed" if flaw_type == "pace_fast" else "this section was drawn out"
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
		"observed_numeric": float(observed),
		"expected_value": expected_value,
		"expected_numeric": float(expected),
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