# Detection and Causal Explanations

Detection uses the existing audio feature extractor and deterministic, typed
frame scores. F0, intensity, and rate are speaker-relative. Hysteresis merges
nearby active frames into regions, enforces configured duration limits, and
snaps eligible boundaries to nearby word timings. Runtime thresholds are stored
in `config/detection.yaml`.

## Alignment and Modes

Pace, pause, monotone, volume, and filler scoring use plain forced alignment of
the supplied transcript. Wildcard alignment and ASR are not part of this bounded
evaluation path. Paired mode compares a recording to its same-transcript IDEAL;
an IDEAL compared with itself must produce no regions. Reference-free mode
scores each DEV passage against statistics fitted from other DEV IDEAL passages
only. The TEST runner fits its reference-free statistics from DEV IDEALs and
does not tune on TEST data.

Filler candidates are restricted to sustained, stable, voiced and speech-active
frames that remain unassigned by plain alignment inside an inter-word gap.
Stumble-repeat is disabled with the reason `needs ASR, out of scope`; plain
alignment cannot supply inserted-token intervals.

## DEV-Tuned Detectors

Run `python run.py eval` using the project virtual environment. It reads DEV
manifest rows only, searches each detector's extended score-derived threshold
grid independently per mode, and greedily accepts the best-F1 candidate only
when every DEV control family and the aggregate remain within the combined
false-region budget. A mode is disabled when its best DEV F1 is below the
configured minimum. Thresholds, mode-specific F1, and disable reasons below
are refreshed from `eval/results/detection_threshold_tuning.csv` after a full
DEV evaluation.

<!-- DEV_TUNED_DETECTORS_START -->

| Detector | Mode | DEV status | F1@0.3 | Enter | Exit | Reason when disabled |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `pace_fast` | paired | enabled | 0.5454545454545455 | 0.2857142857142886 | 0.07 |  |
| `pace_fast` | reference_free | disabled | 0.0 | 2.4494664423095425 | 0.07 | Best DEV F1 0.000 is below 0.150 or no threshold met the combined false-region budget. |
| `pace_slow` | paired | enabled | 0.2 | 0.12 | 0.06 |  |
| `pace_slow` | reference_free | disabled | 0.05797101449275362 | 5.937293659076101 | 0.1 | Best DEV F1 0.058 is below 0.150 or no threshold met the combined false-region budget. |
| `long_pause` | paired | enabled | 0.7567567567567567 | 0.3 | 0.15 |  |
| `long_pause` | reference_free | enabled | 0.6666666666666667 | 1.2299999999999998 | 0.25 |  |
| `monotone` | paired | enabled | 0.17777777777777776 | 0.5899794388054722 | 0.15 |  |
| `monotone` | reference_free | disabled | 0.0851063829787234 | 1.316003966865027 | 0.15 | Best DEV F1 0.085 is below 0.150 or no threshold met the combined false-region budget. |
| `volume_dropoff` | paired | enabled | 0.6046511627906976 | 12.0 | 2.0 |  |
| `volume_dropoff` | reference_free | disabled | 0.10126582278481013 | 3.772754340592023 | 1.8863771702960115 | Best DEV F1 0.101 is below 0.150 or no threshold met the combined false-region budget. |
| `filler` | paired | enabled | 0.7096774193548387 | 0.25 | 0.1 |  |
| `filler` | reference_free | enabled | 0.75 | 0.25 | 0.1 |  |
| `stumble_repeat` | paired | disabled | 0.0 | 5e-324 | 0.0 | needs ASR, out of scope |
| `stumble_repeat` | reference_free | disabled | 0.0 | 5e-324 | 0.0 | needs ASR, out of scope |
<!-- DEV_TUNED_DETECTORS_END -->

The run writes DEV summaries, per-type metrics, severity recall, control false
regions, and a SHA-256 freeze at `eval/results/frozen_config.sha256`.
`python run.py eval --smoke` exercises the same pipeline on two DEV passages but
keeps its artifacts isolated and does not freeze or modify the production
config.

Frame-score separation is recorded before and after measurement changes in
`eval/results/score_diagnostics.csv`, including labeled-region medians, clean
and control medians, frame AUC, and low-threshold region recall. Reference-free
scores use other DEV IDEAL passages rather than the recording's paired ideal;
weak modes remain disabled when they cannot meet the DEV F1 and false-region
gates.

## TEST Evaluation

Run `python run.py eval_test` only after the DEV config has been frozen. The
command refuses to proceed if `config/detection.yaml` no longer matches the
recorded checksum. It evaluates the frozen enabled detectors in paired and
reference-free modes without changing thresholds, and writes separate
`test_*.csv` reports including per-type overlap metrics, severity recall,
boundary errors, control false-region rates, and passage-bootstrap confidence
intervals.

## What the Detector Cannot Do

Voiced speech left unassigned in an inter-word gap is not proof of a lexical
filler; it may be a transcription omission or an alignment error. Acoustic
features cannot identify the word content of that segment. Without ASR, this
pipeline cannot reliably find inserted repeated words, so `stumble_repeat` is
expected to be disabled if it cannot meet the DEV budget. Plain forced
alignment can also misplace boundaries when the supplied transcript does not
match the recording. These limits are reported rather than hidden behind an
acoustic label.

## Explanations

Each region is represented as deterministic JSON with rendered-timeline bounds,
word text, type, observed and expected values, formula, severity, confidence,
and a concise rationale. Template rendering is local and deterministic; no LLM
participates in scoring, flaw detection, or explanation generation.