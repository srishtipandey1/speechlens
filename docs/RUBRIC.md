# SpeechLens Scoring Rubric

Scores are quality-oriented: higher means closer to the ideal/reference. Each
of the six dimensions is mapped continuously to the range `[0, 100]` using
`score_d = 100 * exp(-penalty_d)`. Penalties are non-negative and are capped
only to keep extreme outliers numerically finite. The implementation and named
normalization scales live in `src/speechlens/scoring/rubric.py`.

## Fixed Weighted Total

The total is the fixed weighted mean of the six dimension scores:

`total = sum(weight_d * score_d)`

Weights are declared as constants in `SCORING_WEIGHTS` in the rubric module.
They are not loaded from `config/detection.yaml`, not selected by evaluation,
and not adjusted after examining held-out results. The score-evaluation CSVs
record the total and every component score per recording.

## Dimension Formulas

For paired mode, align participant and ideal words monotonically by normalized
plain-alignment tokens. All frame and interval measurements remain in rendered
seconds. Compute continuous feature deltas only over matched words and adjacent
matched boundaries. Detector evidence includes only region types enabled for the
selected mode in the frozen detection config.

| Dimension | Paired penalty | Reference-free penalty |
| --- | --- | --- |
| Pacing | Mean absolute log participant/ideal word-duration ratio, normalized by the fixed rate scale, plus enabled `pace_fast`/`pace_slow` region evidence. | Mean absolute robust z-score of local duration ratio against the DEV-IDEAL position statistics, plus enabled pace-region evidence. |
| Pausing / fluency | Mean positive participant-minus-ideal inter-word pause excess, normalized by the fixed pause scale, plus enabled `long_pause` region evidence. Punctuation is retained as a labeled boundary, not treated as a lexical timing error. | Positive robust z-score of pause duration against DEV-IDEAL position statistics, plus enabled `long_pause` region evidence. |
| Intonation | Mean absolute log ratio of participant/ideal voiced F0 standard deviation in semitones, plus enabled `monotone` region evidence. | Positive robust z-score for reduced F0 spread against DEV-IDEAL position statistics, plus enabled `monotone` region evidence. |
| Energy | Mean additional downward speech-level change in dB relative to the ideal, after speaker-median normalization, plus enabled `volume_dropoff` region evidence. | Positive robust z-score of downward level change against DEV-IDEAL position statistics, plus enabled volume-region evidence. |
| Articulation | Mean absolute log ratio of participant/ideal syllable-nuclei articulation rate, plus relative syllable-count disagreement where both alignments provide counts and enabled pace-region evidence. | Robust z-score of local duration/articulation-rate proxy against DEV-IDEAL position statistics. It cannot validate whether the supplied transcript is lexically correct. |
| Disfluency | Excess speech-active, voiced, unassigned time versus the ideal, plus enabled `filler` and `stumble_repeat` region evidence. | Unassigned voiced-time cue plus enabled filler-region evidence. This cue cannot establish lexical content. |

For a robust z-score, use the stored DEV-IDEAL median and MAD with the configured
robust scale floor. TEST scoring uses the frozen DEV reference artifact; it never
fits statistics or selects thresholds from TEST recordings. TEST results are
used only to report evaluation metrics and region reliability.

Detector evidence for a dimension is the sum of `region severity * region
duration / recording duration` over its mapped enabled types, multiplied by
the fixed detector-evidence scale. Disabled detector types contribute nothing.
The fixed weights and normalization scales are source constants so evaluation
cannot tune them.

## Reference-Free Reliability

Reference-free mode is not equally reliable for every dimension. Only
pausing/fluency and disfluency have enabled reference-free detector support in
the frozen config; their returned score remains a reference-relative estimate,
not a paired causal comparison. Pacing, intonation, energy, and articulation
are marked limited because their corresponding reference-free detectors are
disabled or the available reference statistics are only indirect proxies.
Paired mode uses the same-transcript ideal and is preferred when an ideal audio
recording is available.

## Evaluation

`python run.py score_eval` writes one row per recording and mode to
`eval/results/scoring_recordings.csv`, and split/mode Spearman severity
correlations with passage-bootstrap confidence intervals plus ideal-vs-flawed
AUC to `eval/results/scoring_metrics.csv`. The AUC treats lower quality totals
as stronger flaw evidence. Controls are scored but excluded from the ideal vs.
flawed AUC and severity correlation populations.
