# Result artifacts

These CSVs are the compact, source-backed summaries used in the project docs and README:

- `detection_summary.csv` — dev paired and reference-free detection summary, including F1 and boundary-error metrics.
- `test_detection_summary.csv` — held-out test summary using the frozen config and threshold lock.
- `test_detection_control_false_positives.csv` — control false-region rate by mode and control type.
- `scoring_metrics.csv` — weighted total vs severity and ideal-vs-flawed AUC by split and mode.

These files are copied from `eval/results/` for quick review in the docs and packaging bundle.
