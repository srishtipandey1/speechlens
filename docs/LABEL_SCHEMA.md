# Label Schema

This document defines the metadata and temporal flaw-label contract. Times are
seconds from the start of the corresponding audio timeline. The schema does not
perform audio processing or infer labels.

## Transcript

Each transcript has a stable `id`, human-readable `title`, full `text`,
`source`, `license`, and `language`. Source and license record provenance; do
not assume a recording is reusable without checking its license.

## Recording

Each recording has an `id`, its `transcript_id`, a `speaker_id`, optional
`speaker_gender`, `kind`, `severity_level`, `audio_path`, `sample_rate`,
`duration_s`, `source`, and `license`. `kind` is one of `ideal`, `injected`,
`human_flawed`, or `control`. Severity is an integer from zero through five.
Ideal and control recordings must have severity level zero. Injected and
human-flawed recordings must have a severity level from one through five.
Non-ideal recordings require `parent_recording_id`, which must resolve to an
ideal recording in the same recording collection. Ideal recordings cannot
have a parent.

`sample_rate` is a positive integer. `duration_s` is finite and positive. The
schema records the declared metadata only; it does not inspect the audio file.

## Word Timing

Each `WordTiming` stores `word`, `start_s`, and `end_s`. Both times are finite
and non-negative, and `start_s` must be less than `end_s`. `confidence` is an
optional value from zero through one; forced alignment populates it from the
geometric mean of the aligned frame probabilities.

## Flaw Label

`flaw_type` is one of `pace_fast`, `pace_slow`, `long_pause`, `monotone`,
`volume_dropoff`, `filler`, or `stumble_repeat`. `severity` is a finite value
from zero through one. `word_indices` contains zero-based, non-negative word
indices; an empty list is allowed when a flaw cannot be attributed to a word.
`notes` is an optional explanatory string and defaults to empty.

Each flaw stores two independent intervals:

- `original_start_s` and `original_end_s` refer to the ideal/source timeline.
- `rendered_start_s` and `rendered_end_s` refer to the flawed recording timeline.

All four times are finite and non-negative, and each start must be less than
its end. Time transformations can shift or stretch an interval, so the two
timelines must be labeled independently. Flaws in a `Pair` are ordered by
`rendered_start_s`; equal start times retain their input order.

## Pair

A `Pair` links `ideal_recording_id` and `flawed_recording_id` and contains a
list of `FlawLabel` items. Pair-level membership is represented by IDs; use a
recording collection to validate parent references and a repository/data
catalog to resolve pair IDs.

## JSON

`save_json` writes UTF-8 JSON with recursively sorted object keys and a trailing
newline. `load_json` parses JSON and validates it against the requested Pydantic
model. List order is preserved, including the sorted flaw order in a pair.

## MMS_FA Alignment

The aligner lowercases ASCII words, retains apostrophes, strips other
punctuation, and expands numeric tokens into English words. Each expanded token
retains its original whitespace-token index; output timings are aggregated
back to the original source word. Alignment is limited to the English
character inventory supported by the installed MMS_FA bundle.

A standalone `*` or `<star>` in the transcript becomes MMS_FA's wildcard token.
It allows the CTC alignment to assign frames to an unknown speech span between
known words. The wildcard does not identify or transcribe that speech, and its
span can be broad or absorb speech that would otherwise align to nearby words.
Use it sparingly and manually inspect alignments that contain it.

## Injection Ground Truth

Injection functions accept a half-open word-index region `[start, end)`. Their
labels preserve source intervals from the ideal timings and separately record
rendered intervals after sample-domain splices. Insertions use a one-sample
source interval to represent the original boundary; rendered intervals cover
the inserted audio exactly. Later word timings are shifted by integer sample
counts, and pace transformations rescale the selected word boundaries.

Inserted `filler` audio is a synthetic approximation, not a human-recorded
filler: it selects a steady voiced vowel excerpt from the same speaker,
compresses its F0 contour toward the excerpt median, and applies short fades.
The corresponding label notes this approximation. A duplicated `stumble_repeat`
uses the speaker's own preceding word audio. Its severity bands choose the
number of preceding words and repeat count, with a short inserted hesitation
between repeated copies at higher severity.

`long_pause` uses a sustained low-level non-silent room-tone excerpt selected
from the same clip, loops it with crossfades, and fades both pause edges. The
severity-duration knots and sentence-join exclusion settings are maintained in
`config/injection.yaml`; the inserted audio is labeled across its full rendered
interval while its source label points to the original word boundary.