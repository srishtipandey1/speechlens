"""Deterministic speech delivery scoring."""

from speechlens.scoring.rubric import (
	DIMENSION_NAMES,
	SCORING_WEIGHTS,
	aggregate_reference_folds,
	score_recording,
)

__all__ = [
	"DIMENSION_NAMES",
	"SCORING_WEIGHTS",
	"aggregate_reference_folds",
	"score_recording",
]
