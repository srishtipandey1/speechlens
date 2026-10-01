"""Pure selection and parsing tests for the LibriSpeech fixture fetcher."""

from pathlib import Path

from scripts.fetch_librispeech import CHAPTER_PATTERN


def test_chapter_transcript_filename_extracts_speaker_and_chapter() -> None:
    """LibriSpeech transcript filenames contain speaker and chapter IDs."""
    match = CHAPTER_PATTERN.match("1272-128104.trans.txt")

    assert match is not None
    assert match.groups() == ("1272", "128104")
    assert Path("1272-128104.trans.txt").name.endswith(".trans.txt")