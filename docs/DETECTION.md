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
Stumble-repeat remains disabled unless DEV tuning shows positive per-type F1 in
both modes while meeting its allocated false-region budget.

## DEV-Tuned Detectors

Run `python run.py eval` using the project virtual environment. It reads DEV
manifest rows only, searches each detector's configured threshold grid, and
accepts a detector only when its false-region share fits the combined budget
and it has positive per-type F1 in both modes. A final combined-budget check
disables additional types if necessary. Thresholds and disable reasons below
are refreshed from `eval/results/detection_threshold_tuning.csv` after a full
DEV evaluation.

<!-- DEV_TUNED_DETECTORS_START -->
Run the full DEV evaluation to populate the enabled/disabled detector table.
<!-- DEV_TUNED_DETECTORS_END -->

The run writes DEV summaries, per-type metrics, severity recall, control false
regions, and a SHA-256 freeze at `eval/results/frozen_config.sha256`.
`python run.py eval --smoke` exercises the same pipeline on two DEV passages but
keeps its artifacts isolated and does not freeze or modify the production
config.

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