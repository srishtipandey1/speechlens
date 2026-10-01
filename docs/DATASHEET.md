# SpeechLens Dataset Datasheet

## Sources and License

LibriSpeech ASR corpus, dev-clean, downloaded from
https://openslr.trmal.net/resources/12/dev-clean.tar.gz. OpenSLR lists the
license as CC BY 4.0. Citation: Vassil Panayotov, Guoguo Chen, Daniel Povey,
and Sanjeev Khudanpur, “LibriSpeech: an ASR corpus based on public domain audio
books,” ICASSP 2015. The extracted passages are attributed in
`data/raw/SOURCES.md`.

## Construction

Passages are selected by ascending numeric speaker ID and chapter ID, using
SPEAKERS.TXT gender metadata. Utterance edge trimming uses a frame-RMS
threshold of -40.0 dBFS, with a
20.0 ms frame, 10.0 ms
hop, and 10.0 ms retained padding. Consecutive
trimmed utterances are joined with 0.5 s
of silence. The exact source transcript lines are joined in utterance order.
Processed audio is stored as mono 16000 Hz FLAC with 24-bit PCM so
quiet source-matched room tone is not quantized to digital zero.

Ideal passage word timings are generated with CPU MMS_FA forced alignment.
Alignment confidence is retained and low-confidence passages are flagged, not
dropped. Flaws use seeded, non-overlapping word regions with a configured
minimum separation. The severity ladder is: L1: 1 flaws at severity 0.2, L2: 2 flaws at severity 0.35, L3: 3 flaws at severity 0.5, L4: 5 flaws at severity 0.7, L5: 7 flaws at severity 0.9.

## Flaw and Control Definitions

Flaw types are pace-fast and pace-slow PSOLA duration edits, long-pause silence
insertion using looped, faded room tone from the clip's quietest sustained
non-silent segment, monotone PSOLA F0 compression, smooth volume dropoff,
synthetic filler, and repeated preceding words with severity-scaled stutter
gaps. The severity-duration knots and stumble bands are defined in
`config/injection.yaml`. The filler is a synthetic approximation made from
same-speaker voiced audio. PSOLA may introduce timbre or boundary artifacts.
Praat duration overlap-add output is cached by source audio and transform
settings so repeated builds in this workspace reuse identical samples; a fresh
cache's first native Praat realization is not guaranteed byte-identical across
machines. LibriSpeech audiobook readings are treated as clean fluent reference
speech by volunteers, not as professional orator performances.

Each ideal passage has identity PSOLA resynthesis, lossy codec round trip, gain
change, and additive noise controls. Gain sign alternates deterministically
between passages and is peak-limited. Noise SNR is 30.0 dB.
The available installed MP3 encoder is used for lossy round trips.

## Splits and Counts

Counts below are derived from `data/labels/manifest.csv`.

- Unique passages: 16
- Ideal recordings: 16
- Injected recordings: 80
- Control recordings: 64
- Total recordings: 160
- Dev ideal recordings by gender: M=4, F=4
- Test ideal recordings by gender: M=4, F=4
- Alignment quality flags: None

Speakers are disjoint across dev and test. Dev is intended for threshold tuning;
test is held for final reporting. Passage speaker, chapter, utterance IDs,
transcript, and split are recorded under `data/labels/passages/`.
