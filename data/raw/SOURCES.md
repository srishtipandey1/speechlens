# Test Clip Sources

- Dataset: LibriSpeech ASR corpus, dev-clean
- Download URL: https://openslr.trmal.net/resources/12/dev-clean.tar.gz
- License listed on the OpenSLR SLR12 page: CC BY 4.0
- Required citation: Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur, 'LibriSpeech: an ASR corpus based on public domain audio books,' ICASSP 2015.
- Speaker ID: 1272
- Chapter ID: 128104
- Utterance IDs, in joined order: 1272-128104-0001, 1272-128104-0002, 1272-128104-0003
- Selection rule: lexicographically first eligible chapter; earliest shortest
  consecutive transcript window with at least 25 seconds of source audio and
  a joined duration within the configured clip bounds.
