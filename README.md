# SpeechLens

SpeechLens is a deterministic speech-evaluation prototype that compares a spoken recording against a same-text reference and highlights the specific delivery regions that differ. The project keeps the pipeline simple enough to reason about: it aligns the transcript to the audio, extracts timing, pitch, loudness, and disfluency signals, and surfaces evidence-backed explanations in a lightweight dashboard.

This is not a replacement for a human judge. The aim is to make objective delivery differences legible, reproducible, and explainable.

## Current evaluation status

The repository includes the frozen evaluation artifacts under `eval/results`, and the numbers below are copied directly from those CSVs.

- Paired dev F1@IoU 0.3: 0.464
- Reference-free dev F1@IoU 0.3: 0.282
- Paired test F1@IoU 0.3: 0.383
- Reference-free test F1@IoU 0.3: 0.278
- Paired test mean boundary error: 256.34 ms
- Paired test gain-control false-region rate: 3.09 per minute
- Paired test total-vs-severity Spearman: -0.886

This is a deliberately honest summary. The first pass of the detection work was contaminated by a bad timing source: the pipeline was reading dataset sidecar timings instead of forcing alignment from the audio itself. The current numbers above are from the audio-derived alignment path and are the values we trust.

## What the system does

- Aligns a transcript to the audio with plain forced alignment.
- Compares a reference reading against a participant read in paired mode.
- Uses DEV-only reference statistics for the reference-free path.
- Detects pace, pauses, monotone pitch, volume dropoff, and filler-like voiced gaps.
- Produces structured region explanations and a local dashboard for review.

The detection logic, scoring rubric, and dataset notes live in `docs/DETECTION.md`, `docs/RUBRIC.md`, and `docs/DATASHEET.md`.

## Local setup

From PowerShell, create a local environment and install the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python run.py setup
```

Run the unit and integration suite with:

```powershell
python run.py test
```

The CI variant skips slow tests:

```powershell
$env:CI = "true"
python run.py test
```

To start the local dashboard:

```powershell
python run.py serve
```

## Docker

The project keeps a CPU-only runtime and a small application bundle. Build the image from the repo root and run it locally:

```powershell
docker build -t speechlens .
docker run --rm -it speechlens
```

The container starts the same FastAPI dashboard the local command does, but it does not change any frozen detection thresholds or scoring weights.

## Dataset and licensing

The audio and transcript source material comes from LibriSpeech dev-clean, under the CC BY 4.0 license. The project-specific dataset notes, source citations, and label schema are in `docs/DATASHEET.md` and `docs/LABEL_SCHEMA.md`.

The project code is under the MIT license (`LICENSE`). The dataset-derived assets are covered separately in `DATA_LICENSE.md`.

## Project structure

- `src/speechlens/` — feature extraction, alignment, detection, scoring, and API code.
- `config/` — frozen runtime configuration for detection and dataset behavior.
- `data/` — labels, processed data, and raw source relations.
- `frontend/` — simple static dashboard and browser UI.
- `eval/results/` — the evidence-backed summary CSVs used for the final report.
- `docs/results/` — a compact copy of the key result files for documentation use.

## Notes on reliability

The project is designed to be conservative:

- A region must be backed by actual feature evidence, not a vague language-model summary.
- The TEST split is held out and evaluated only with the frozen threshold set.
- The reference-free mode is weaker than paired mode when an ideal reading exists.
- The detector coverage is narrower than a general-purpose ASR system; it is meant for a controlled evaluation workflow, not unrestricted talking assessment.

If you want to inspect the exact evidence behind the summary, start with `eval/results/test_detection_summary.csv`, `eval/results/test_detection_control_false_positives.csv`, and `eval/results/scoring_metrics.csv`.
