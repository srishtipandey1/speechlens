# Detection and Causal Explanations

The detection implementation uses the existing mono 16 kHz feature extractor.
F0 and intensity are speaker-relative; local rate is syllable nuclei per active
speech second. Detection emits per-frame deviation tables, smooths them, applies
enter/exit hysteresis, drops regions shorter than the configured duration,
merges nearby regions, and snaps eligible edges to nearby word boundaries.
Thresholds and scales live in `config/detection.yaml`.

## Baseline Modes

Paired mode first monotonically matches normalized word tokens between the
participant and same-transcript IDEAL timing lists. An injected extra repeated
token remains unmatched and does not shift later words. For every matched word,
librosa MFCC DTW maps participant frames to IDEAL frames; normalized pitch,
intensity, local speech rate, and inter-word pause are compared on those
word-anchored spans. A pause following punctuation or an IDEAL pause at or above
the natural-pause limit is excluded.

Reference-free mode stores position-class medians and median absolute
deviations for rate, pitch spread, intensity, and pause duration, plus phrase
cadence, phrase pitch spread, and pause-ratio statistics. The evaluator fits
these distributions from DEV IDEAL recordings only and writes
`eval/results/reference_free_ideal_stats.json`. Robust z-scores use
`(value - median) / max(1.4826 * MAD, configured_floor)`.

## Classification Rules

Rules are applied top-to-bottom. This priority ensures detectable lexical
events take precedence over broad acoustic patterns.

| Priority | Type | Transparent condition |
| --- | --- | --- |
| 1 | `stumble_repeat` | An adjacent repeated sequence of up to the configured maximum word count is unmatched by monotone transcript alignment, or the repeated-word score reaches its configured threshold. |
| 2 | `long_pause` | Mid-phrase pause excess reaches the configured duration; punctuation boundaries and pauses already present in the IDEAL are excluded. |
| 3 | `filler` | A configured filler token is unmatched by transcript alignment and has speech-active voiced frames, or a sustained voiced segment occurs outside aligned word timings. |
| 4 | `pace_fast` | Rate z-score is at or above the configured positive threshold and the region is not a long pause. |
| 5 | `pace_slow` | Rate z-score is at or below the negative configured threshold. |
| 6 | `monotone` | Participant-to-IDEAL F0 standard-deviation ratio is at or below the configured limit. |
| 7 | `volume_dropoff` | Speaker-normalized intensity z-score is at or below the negative configured limit. |
| 8 | `other_deviation` | A detected region does not satisfy an earlier explicit rule. |

Low-confidence aligned words do not contribute a separate flaw score. Their
confidence is averaged within each region, and the configured fraction below
the alignment confidence threshold reduces the region confidence. Confidence
is clipped to `[0, 1]`.

## Explanation Records

Each region is represented as deterministic JSON with rendered-timeline bounds,
word text, type, observed and expected values, z-score or ratio, formula,
severity, confidence, a sentence containing the measured values, and an
actionable suggestion. Template rendering is local and deterministic; there is
no LLM in scoring, classification, or explanation generation.

## Limits

Reference-free filler detection can only identify voiced speech outside the
provided aligned words. It cannot tell whether that segment is “uh”, a
misalignment, or another non-lexical sound without a decoder or human transcript.
Reference-free stumble detection depends on repeated tokens being represented
in the supplied word timings; it will miss repetitions absent from those
timings. The current detector does not run an ASR model to create either cue.
These limits are reported rather than hidden behind an acoustic label.

## DEV Evaluation

Run `python run.py eval` with the project venv activated. The evaluator selects
manifest rows with `split == "dev"` before resolving any sidecar, passage, or
audio path; it does not open test-split audio or labels. It reports region
precision/recall/F1 at temporal IoU 0.3 and 0.5, matched boundary MAE, class
confusion, severity sensitivity, and false regions per minute for IDEAL and
each control family. Threshold selection maximizes mean mode F1 subject to at
most one false region per minute in every IDEAL/control group. The selected
thresholds are frozen in `config/detection.yaml` and hashed in
`eval/results/frozen_config.sha256`.