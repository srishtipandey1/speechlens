# Acoustic Feature Extraction

Analysis scripts in this document use only manifest rows whose `split` is
`dev`. The report helpers filter those rows before opening audio or sidecars;
test-passage audio and labels are not read.

## Frame Table

`extract_frame_features(audio, word_timings)` returns a pandas DataFrame with
one row per configured hop. The frame-center time is
`time_s = frame_index * hop_s`; sample rate, hop, frame length, FFT, and
threshold settings are in `config/features.yaml`.

| Column | Meaning |
|---|---|
| `time_s` | Frame-center time in seconds. |
| `f0_hz` | Praat autocorrelation pitch in Hz; zero where unvoiced. |
| `voiced` | True when Praat returns positive F0 for the frame. |
| `f0_semitones` | `12 * log2(f0_hz / recording_median_voiced_f0_hz)`, computed over the recording itself; use `voiced` as the validity mask. |
| `intensity_raw_dbfs` | Frame RMS in dBFS: `20 * log10(rms + epsilon)`. |
| `intensity_db` | Raw intensity minus the recording median over speech-active frames. |
| `mfcc_1` ... `mfcc_13` | DCT coefficients of log Mel power. |
| `delta_mfcc_1` ... `delta_mfcc_13` | librosa local-regression first derivatives of MFCCs. |
| `spectral_flux` | L2 norm of positive spectral-magnitude differences from the previous frame. |
| `hnr_db` | Praat autocorrelation harmonicity in dB; undefined/silent frames are NaN. |
| `speech_activity` | RMS dBFS is at or above `speech_activity_threshold_dbfs`. |
| `word_index` | Zero-based aligned word index for `[start_s, end_s)`; `-1` outside all words. |
| `intensity_praat_db` | Praat intensity mapped to the frame grid for cross-checking. |
| `frame_rms` | Linear RMS envelope used by modulation and syllable-nucleus calculations. |

Praat calls verified against Parselmouth are `Sound.to_pitch_ac`,
`Sound.to_intensity`, and `Sound.to_harmonicity_ac`. Praat tracks are sampled
using their returned `x1` and `dx`, then mapped to frame centers by nearest
track frame. Isolated one-frame octave jumps are corrected only when both
neighboring voiced frames agree within the configured tolerance; sustained
octave changes are kept.

MFCCs use the configured Mel-band count, FFT size, window length, and hop.
Delta MFCCs use `librosa.feature.delta` with the configured regression width.
Spectral flux is
`sqrt(sum(max(0, |X_m[k]| - |X_(m-1)[k]|)^2))` for STFT magnitudes `X_m`.

## Word and Phrase Features

The energy envelope is converted to dBFS and local peaks are counted as
syllable-nucleus estimates using the configured prominence and minimum-spacing
thresholds. For each word the table reports that count, the transcript
vowel-group count as a cross-check, their absolute difference, articulation
rate (nuclei divided by speech-active time), words per second, voiced fraction,
F0 range/std in semitones, and intensity range. Vowel groups are a simple
orthographic estimate, not a phonetic syllabifier.

Phrase boundaries are transcript words ending in terminal/clause punctuation.
Phrase features aggregate words per second, articulation rate, pause count and
duration, pause ratio, F0 range/std, intensity range, and voiced fraction.
Pause rows contain start, end, duration, the preceding word index, and a
`punctuation_boundary` or `mid_phrase_boundary` classification derived from
the transcript token after the preceding word. Recording/phrase pause ratio is
summed pause duration divided by interval duration.

Voiced fraction is voiced-frame count divided by speech-active frame count. It
does not calculate jitter and is therefore a jitter-free voicing measure.

## Modulation Spectra

For sampled envelope `x[n]`, subtract its mean and apply a Hann window `w[n]`:

`X[k] = rFFT((x[n] - mean(x)) * w[n])`

`P[k] = |X[k]|^2 / sum(w[n])^2`

Band power is the sum of `P[k]` for bins in the configured band; peak
modulation frequency is the frequency at the largest-power bin in that band.
The frame-RMS envelope is analyzed in `energy_syllable_rate` for syllable-rate
modulation. The voiced F0-semitone contour, with short unvoiced gaps
linearly interpolated between voiced frames for the FFT, is analyzed in
`f0_monotone`; reduced low-frequency F0 power can indicate flattened pitch but
is not a standalone classifier. Long gaps are interpolated too, so interpret
this spectrum alongside voiced fraction and pause features.

## Dev Diagnostics

`scripts/check_normalization.py` plots raw voiced F0 and raw speech intensity
beside speaker-relative semitones and intensity, using dev ideals only. It
prints male/female median gaps in their respective units and the percentage
reduction in the unitless Cohen's d between dev speaker medians. The normalized
gender gap is expected to approach zero because each recording is centered on
its own median. `scripts/check_feature_deltas.py` compares each dev
flaw's rendered interval with its parent ideal's original word indices, writes
`eval/results/feature_deltas_dev.csv`, and reports expected-direction checks.
It separately writes `eval/results/control_drift_dev.csv` for dev identity,
lossy, gain, and noise controls. No parameter choice or inspection in these
scripts reads the test split.# Acoustic Feature Extraction

Feature inspection and parameter checks for this task use only recordings whose
manifest row has `split=dev`. `scripts/dev_feature_reports.py` filters the
manifest before opening audio or sidecars; the held-out test recordings are not
loaded by either report script.

## Frame Table

`extract_frame_features` emits one row per frame on the hop grid. Frame center
time is `t_m = m H / f_s`, where `H` and `f_s` are read from
`config/features.yaml`. The table contains:

- `time_s`: frame-center time in seconds.
- `f0_hz`, `voiced`: Praat autocorrelation pitch and its positive-F0 voiced
  indicator. Isolated one-frame octave slips are corrected only when both
  neighbors are voiced and agree within the configured semitone tolerance;
  sustained pitch changes are retained.
- `f0_semitones`: F0 relative to the recording's median voiced F0:
  `12 log2(f0_hz / median_voiced_f0_hz)`. Unvoiced frames are zero in this
  table, and must be masked with `voiced` for pitch statistics.
- `intensity_raw_dbfs`: RMS level in dBFS, `20 log10(rms + epsilon)`.
- `intensity_db`: per-recording speech-level normalization,
  `intensity_raw_dbfs - median(intensity_raw_dbfs over speech_activity frames)`.
- `mfcc_1` through `mfcc_13`: DCT-II coefficients of log Mel power, using the
  configured Mel band count, FFT length, and 10 ms frame hop.
- `delta_mfcc_1` through `delta_mfcc_13`: librosa local-regression first
  temporal derivatives of the MFCC sequences.
- `spectral_flux`: L2 norm of positive spectral-magnitude change between
  adjacent STFT frames.
- `hnr_db`: Praat autocorrelation harmonic-to-noise ratio; undefined or silent
  frames are NaN.
- `speech_activity`: `True` when frame RMS in dBFS is at or above
  `speech_activity_threshold_dbfs` from `config/features.yaml`.
- `word_index`: zero-based aligned word index for frames in `[start_s, end_s)`;
  `-1` marks frames outside every word timing.

Praat's verified calls are `Sound.to_pitch_ac`, `Sound.to_intensity`, and
`Sound.to_harmonicity_ac`. Pitch/intensity tracks are mapped to the frame grid
using their own Praat `x1` and `dx`, not assumed timestamps.

## Word and Phrase Metrics

The frame intensity envelope is searched for local peaks with configured
minimum prominence and spacing. Each local peak is a syllable-nucleus estimate.
Per-word output also includes the transcript's vowel-group count as a simple
independent cross-check; vowel groups are not treated as a phonetic syllabifier.

- Articulation rate is estimated syllable nuclei divided by speech-active time.
- Words per second is the aligned word count divided by interval duration.
- Voiced fraction is voiced-frame count divided by speech-active frame count;
  this is a jitter-free voicing measure, not a jitter estimate.
- F0 range and population standard deviation are computed in speaker-relative
  semitones, over voiced frames only.
- Intensity range is the max-minus-min normalized speech-frame intensity.
- Phrase boundaries are the aligned word spans split after terminal transcript
  punctuation. Phrase output includes articulation rate, words per second, F0
  range/std, intensity range, and pause duration/ratio.

The pause table contains `start_s`, `end_s`, `duration_s`,
`boundary_after_word_index`, and `boundary_type`. A pause after a transcript
token ending in terminal or clause punctuation is a `punctuation_boundary`;
otherwise it is a `mid_phrase_boundary`. Pause ratio is summed pause duration
divided by the containing recording or phrase duration.

## Modulation Spectra

For a sampled envelope `x[n]`, subtract its mean, multiply by a Hann window
`w[n]`, and compute `X[k] = rFFT((x[n] - mean(x)) w[n])`. The reported power
bins are `P[k] = |X[k]|^2 / (sum(w[n]))^2`; band power is the sum of bins in the
configured range. The peak modulation frequency is the frequency of the
maximum-power bin in that range.

The energy-envelope spectrum reports power and peak frequency in the
`energy_syllable_rate` band. Its peak is a coarse syllable-rate indicator, not
an alignment-derived syllable count. The F0-contour spectrum interpolates only
short voiced gaps before transforming and reports power/peak in the
`f0_monotone` band; reduced low-frequency F0 power can indicate flatter pitch,
but is not by itself a monotonicity classifier.

## Normalization and Dev Reports

`scripts/check_normalization.py` plots raw F0/intensity distributions beside
per-recording normalized semitone/dB distributions using dev ideals only. It
prints the female-minus-male median gap before and after normalization and its
percentage reduction. `scripts/check_feature_deltas.py` pairs each dev flaw
interval with the same original word indices in its parent ideal, reports the
targeted and candidate feature deltas, and separately summarizes dev control
drift. No thresholds are fit on the test split.