"""Fetch and deterministically assemble a local LibriSpeech alignment clip."""

import re
import sys
import tarfile
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import soundfile as sf
import yaml
from scipy.signal import resample_poly


ARCHIVE_URL = "https://openslr.trmal.net/resources/12/dev-clean.tar.gz"
ARCHIVE_NAME = "dev-clean.tar.gz"
ARCHIVE_SIZE_LIMIT = 2**40
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
CHAPTER_PATTERN = re.compile(r"^(\d+)-(\d+)\.trans\.txt$")
REQUIRED_CITATION = (
    "Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur, "
    "'LibriSpeech: an ASR corpus based on public domain audio books,' "
    "ICASSP 2015."
)


@dataclass(frozen=True)
class ArchiveMetadata:
    """Metadata verified from the OpenSLR archive endpoint."""

    url: str
    size_bytes: int


@dataclass(frozen=True)
class Utterance:
    """One ordered LibriSpeech transcript/audio pair and its duration."""

    utterance_id: str
    transcript: str
    audio_path: Path
    sample_count: int
    sample_rate: int

    @property
    def duration_s(self) -> float:
        """Return the exact source duration in seconds."""
        return self.sample_count / self.sample_rate


def load_config(project_root: Path) -> dict:
    """Load the shared audio and test-clip settings."""
    config_path = project_root / "config" / "default.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def verify_archive_url(url: str = ARCHIVE_URL) -> ArchiveMetadata:
    """Verify the published endpoint and size with an HTTP HEAD request."""
    request = Request(url, method="HEAD")
    with urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(f"archive HEAD request returned HTTP {response.status}")
        content_length = response.headers.get("Content-Length")
        if content_length is None:
            raise RuntimeError("archive server did not provide Content-Length")
        size_bytes = int(content_length)
        if size_bytes <= 0 or size_bytes > ARCHIVE_SIZE_LIMIT:
            raise RuntimeError(f"unexpected archive size: {size_bytes} bytes")
        return ArchiveMetadata(response.geturl(), size_bytes)


def _response_start(response, offset: int) -> tuple[int, int]:
    """Return the write offset and total size represented by a GET response."""
    status = response.status
    if offset and status == 206:
        content_range = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
        if match is None or int(match.group(1)) != offset:
            raise RuntimeError("server returned an invalid Content-Range for resume")
        return offset, int(match.group(3))
    if status == 200:
        return 0, int(response.headers.get("Content-Length", "0"))
    raise RuntimeError(f"archive GET request returned HTTP {status}")


def download_archive(metadata: ArchiveMetadata, archive_path: Path) -> None:
    """Download the archive with byte-range resume and a terminal progress bar."""
    if archive_path.is_file():
        print(f"Archive already exists; skipping download: {archive_path}")
        return

    partial_path = archive_path.with_name(f"{archive_path.name}.part")
    offset = partial_path.stat().st_size if partial_path.exists() else 0
    request_headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = Request(metadata.url, headers=request_headers)

    try:
        response_context = urlopen(request, timeout=60)
    except HTTPError as error:
        if error.code == 416 and offset == metadata.size_bytes:
            partial_path.replace(archive_path)
            return
        raise
    except URLError as error:
        raise RuntimeError(f"could not download LibriSpeech: {error}") from error

    with response_context as response:
        write_offset, response_total = _response_start(response, offset)
        if response_total and response_total != metadata.size_bytes:
            raise RuntimeError(
                f"download size changed: HEAD={metadata.size_bytes}, GET={response_total}"
            )
        mode = "ab" if write_offset else "wb"
        downloaded = write_offset
        last_reported = -1
        with partial_path.open(mode) as archive_file:
            while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                archive_file.write(chunk)
                downloaded += len(chunk)
                if downloaded // DOWNLOAD_CHUNK_BYTES != last_reported:
                    last_reported = downloaded // DOWNLOAD_CHUNK_BYTES
                    _print_progress(downloaded, metadata.size_bytes)
        print()

    if partial_path.stat().st_size != metadata.size_bytes:
        raise RuntimeError(
            f"download incomplete: received {partial_path.stat().st_size} "
            f"of {metadata.size_bytes} bytes; run again to resume"
        )
    partial_path.replace(archive_path)


def _print_progress(downloaded: int, total: int) -> None:
    """Print a compact in-place download progress indicator."""
    ratio = min(1.0, downloaded / total)
    bar_width = 30
    filled = int(bar_width * ratio)
    bar = "=" * filled + " " * (bar_width - filled)
    print(
        f"\r[{bar}] {ratio:6.1%} "
        f"{downloaded / 1_000_000:.1f}/{total / 1_000_000:.1f} MB",
        end="",
        flush=True,
    )


def extract_archive(archive_path: Path, extraction_root: Path) -> None:
    """Safely extract the verified tar archive beneath the raw-data folder."""
    extraction_root.mkdir(parents=True, exist_ok=True)
    root_resolved = extraction_root.resolve()
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            member_path = (extraction_root / member.name).resolve()
            if not member_path.is_relative_to(root_resolved):
                raise RuntimeError(f"archive contains an unsafe path: {member.name}")
        archive.extractall(extraction_root, members=members, filter="data")


def _read_chapter(transcript_path: Path) -> tuple[str, str, list[Utterance]]:
    """Read one chapter transcript and resolve its ordered FLAC utterances."""
    match = CHAPTER_PATTERN.match(transcript_path.name)
    if match is None:
        raise ValueError(f"unexpected LibriSpeech transcript filename: {transcript_path}")
    speaker_id, chapter_id = match.group(1), match.group(2)
    utterances: list[Utterance] = []
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        utterance_id, separator, transcript = line.partition(" ")
        if not separator or not transcript.strip():
            continue
        audio_path = transcript_path.parent / f"{utterance_id}.flac"
        if not audio_path.is_file():
            continue
        audio_info = sf.info(audio_path)
        utterances.append(
            Utterance(
                utterance_id=utterance_id,
                transcript=transcript.strip(),
                audio_path=audio_path,
                sample_count=audio_info.frames,
                sample_rate=audio_info.samplerate,
            )
        )
    return speaker_id, chapter_id, utterances


def _select_window(
    utterances: list[Utterance],
    minimum_consecutive_audio_s: float,
    minimum_duration_s: float,
    maximum_duration_s: float,
    silence_s: float,
) -> list[Utterance] | None:
    """Choose the earliest shortest consecutive window satisfying duration limits."""
    for start_index in range(len(utterances)):
        selected: list[Utterance] = []
        raw_duration_s = 0.0
        for utterance in utterances[start_index:]:
            selected.append(utterance)
            raw_duration_s += utterance.duration_s
            joined_duration_s = raw_duration_s + silence_s * (len(selected) - 1)
            if joined_duration_s > maximum_duration_s:
                break
            if (
                len(selected) >= 2
                and raw_duration_s >= minimum_consecutive_audio_s
                and minimum_duration_s <= joined_duration_s <= maximum_duration_s
            ):
                return selected
    return None


def find_selection(
    extraction_root: Path,
    config: dict,
) -> tuple[str, str, list[Utterance]]:
    """Select the first eligible speaker/chapter/window by sorted path order."""
    settings = config["test_clip"]
    transcripts = sorted(extraction_root.rglob("*.trans.txt"))
    for transcript_path in transcripts:
        speaker_id, chapter_id, utterances = _read_chapter(transcript_path)
        selected = _select_window(
            utterances,
            minimum_consecutive_audio_s=float(settings["min_consecutive_audio_s"]),
            minimum_duration_s=float(settings["minimum_duration_s"]),
            maximum_duration_s=float(settings["maximum_duration_s"]),
            silence_s=float(settings["silence_between_utterances_s"]),
        )
        if selected:
            return speaker_id, chapter_id, selected
    raise RuntimeError("no chapter has a consecutive utterance window matching the clip limits")


def _load_mono_16k(utterance: Utterance, sample_rate_hz: int) -> np.ndarray:
    """Read an utterance as mono float32 at the configured sample rate."""
    samples, actual_sample_rate = sf.read(
        utterance.audio_path,
        dtype="float32",
        always_2d=True,
    )
    mono = samples.mean(axis=1, dtype=np.float32)
    if actual_sample_rate != sample_rate_hz:
        divisor = gcd(actual_sample_rate, sample_rate_hz)
        mono = resample_poly(
            mono,
            up=sample_rate_hz // divisor,
            down=actual_sample_rate // divisor,
        ).astype(np.float32)
    return np.ascontiguousarray(mono, dtype=np.float32)


def assemble_clip(
    selected: list[Utterance],
    sample_rate_hz: int,
    silence_s: float,
) -> tuple[np.ndarray, str]:
    """Join selected audio and exact source transcript text with silence gaps."""
    gap = np.zeros(round(silence_s * sample_rate_hz), dtype=np.float32)
    chunks: list[np.ndarray] = []
    for index, utterance in enumerate(selected):
        if index:
            chunks.append(gap)
        chunks.append(_load_mono_16k(utterance, sample_rate_hz))
    waveform = np.concatenate(chunks).astype(np.float32, copy=False)
    transcript = " ".join(item.transcript for item in selected).lower()
    return waveform, transcript


def write_sources(
    output_path: Path,
    archive_url: str,
    speaker_id: str,
    chapter_id: str,
    selected: list[Utterance],
) -> None:
    """Write the required LibriSpeech attribution and deterministic selection."""
    utterance_ids = ", ".join(item.utterance_id for item in selected)
    source_text = f"""# Test Clip Sources

- Dataset: LibriSpeech ASR corpus, dev-clean
- Download URL: {archive_url}
- License listed on the OpenSLR SLR12 page: CC BY 4.0
- Required citation: {REQUIRED_CITATION}
- Speaker ID: {speaker_id}
- Chapter ID: {chapter_id}
- Utterance IDs, in joined order: {utterance_ids}
- Selection rule: lexicographically first eligible chapter; earliest shortest
  consecutive transcript window with at least 25 seconds of source audio and
  a joined duration within the configured clip bounds.
"""
    output_path.write_text(source_text, encoding="utf-8")


def run(project_root: Path) -> tuple[str, str, list[Utterance]]:
    """Fetch, extract, select, and write the local test fixture."""
    raw_root = project_root / "data" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    config = load_config(project_root)
    sample_rate_hz = int(config["audio"]["sample_rate_hz"])
    clip_settings = config["test_clip"]
    archive_path = raw_root / ARCHIVE_NAME

    metadata = verify_archive_url()
    print(f"Verified archive URL: {metadata.url}")
    print(f"Archive size (HEAD Content-Length): {metadata.size_bytes:,} bytes")
    download_archive(metadata, archive_path)

    extraction_root = raw_root / "librispeech_dev_clean"
    extracted_transcripts = list(extraction_root.rglob("*.trans.txt"))
    if not extracted_transcripts:
        extract_archive(archive_path, extraction_root)

    speaker_id, chapter_id, selected = find_selection(extraction_root, config)
    waveform, transcript = assemble_clip(
        selected,
        sample_rate_hz,
        float(clip_settings["silence_between_utterances_s"]),
    )
    duration_s = waveform.size / sample_rate_hz
    if not (
        float(clip_settings["minimum_duration_s"])
        <= duration_s
        <= float(clip_settings["maximum_duration_s"])
    ):
        raise RuntimeError(f"assembled clip duration is outside configured bounds: {duration_s}")

    sf.write(raw_root / "test_clip.wav", waveform, sample_rate_hz, subtype="PCM_16")
    (raw_root / "test_clip.txt").write_text(f"{transcript}\n", encoding="utf-8")
    write_sources(
        raw_root / "SOURCES.md",
        metadata.url,
        speaker_id,
        chapter_id,
        selected,
    )
    print(f"Selected speaker {speaker_id}, chapter {chapter_id}")
    print(f"Utterances: {', '.join(item.utterance_id for item in selected)}")
    print(f"Clip duration: {duration_s:.3f} seconds")
    print(f"Wrote {raw_root / 'test_clip.wav'}")
    print(f"Wrote {raw_root / 'test_clip.txt'}")
    print(f"Wrote {raw_root / 'SOURCES.md'}")
    return speaker_id, chapter_id, selected


def main() -> None:
    """Run the fetcher relative to the repository root."""
    project_root = Path(__file__).resolve().parents[1]
    try:
        run(project_root)
    except (HTTPError, OSError, RuntimeError, URLError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()