"""Flaw detection components."""

from speechlens.detection.core import (
	detect_deviation_regions,
	fit_reference_free,
	paired_deviation_table,
	reference_free_deviation_table,
)

__all__ = [
	"detect_deviation_regions",
	"fit_reference_free",
	"paired_deviation_table",
	"reference_free_deviation_table",
]