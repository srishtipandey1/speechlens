"""Flaw injection components."""

from speechlens.injection.engine import (
	FlawSpec,
	apply_flaws,
	filler,
	long_pause,
	monotone,
	pace_fast,
	pace_slow,
	stumble_repeat,
	volume_dropoff,
)

__all__ = [
	"FlawSpec",
	"apply_flaws",
	"filler",
	"long_pause",
	"monotone",
	"pace_fast",
	"pace_slow",
	"stumble_repeat",
	"volume_dropoff",
]