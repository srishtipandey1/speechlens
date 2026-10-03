"""Precompute one complete DEV-passage API demo locally without network access."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from speechlens.api.service import analyze_audio_bytes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = PROJECT_ROOT / "data" / "processed" / "demo"
CONTROL_PRIORITY = ("identity", "mild_noise", "gain", "lossy", "mp3")


def _select_passage(rows: list[dict[str, str]], requested_id: str | None) -> tuple[str, list[dict[str, str]]]:
    """Select a DEV passage containing its ideal, severity ladder, and two controls."""
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["passage_id"], []).append(row)
    passage_ids = [requested_id] if requested_id else sorted(grouped)
    for passage_id in passage_ids:
        if passage_id not in grouped:
            continue
        passage_rows = grouped[passage_id]
        ideal = next((row for row in passage_rows if row["kind"] == "ideal"), None)
        injected = {
            int(row["severity_level"]): row
            for row in passage_rows
            if row["kind"] == "injected"
        }
        controls = [row for row in passage_rows if row["kind"] == "control"]
        ordered_controls = []
        for key in CONTROL_PRIORITY:
            ordered_controls.extend(
                row for row in controls
                if key in row["recording_id"].lower()
                and row not in ordered_controls
            )
        if ideal and all(level in injected for level in range(1, 6)) and len(ordered_controls) >= 2:
            selected = [ideal, *(injected[level] for level in range(1, 6)), *ordered_controls[:2]]
            return passage_id, selected
    raise RuntimeError(
        "No DEV passage has an ideal, the full injected severity ladder, and two recognized controls"
    )


def precompute_demo(project_root: Path = PROJECT_ROOT, passage_id: str | None = None) -> list[Path]:
    """Write stable per-recording JSON for one paired DEV passage."""
    project_root = project_root.resolve()
    manifest_path = project_root / "data" / "labels" / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError("DEV manifest is missing; build the dataset first")
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        dev_rows = [row for row in csv.DictReader(stream) if row.get("split") == "dev"]
    selected_passage, rows = _select_passage(dev_rows, passage_id)
    passage_path = project_root / "data" / "labels" / "passages" / f"{selected_passage}.json"
    passage = json.loads(passage_path.read_text(encoding="utf-8"))
    if passage.get("split") != "dev":
        raise ValueError(f"selected passage is not DEV: {selected_passage}")
    transcript = str(passage["transcript"])
    ideal_row = next(row for row in rows if row["kind"] == "ideal")
    ideal_bytes = (project_root / ideal_row["path"]).read_bytes()
    demo_dir = project_root / "data" / "processed" / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for row in rows:
        recording_id = row["recording_id"]
        audio_bytes = (project_root / row["path"]).read_bytes()
        result = analyze_audio_bytes(
            audio_bytes,
            transcript,
            ideal_bytes,
            project_root=project_root,
        )
        result.update({
            "recording_id": recording_id,
            "passage_id": selected_passage,
            "kind": row["kind"],
            "severity_level": int(row["severity_level"]),
        })
        output_path = demo_dir / f"{recording_id}.json"
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        output_paths.append(output_path)
    print(f"Precomputed {len(output_paths)} DEV demo results for passage {selected_passage}")
    for path in output_paths:
        print(path.relative_to(project_root).as_posix())
    return output_paths


def main() -> None:
    """Precompute the requested DEV passage or the first complete passage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passage-id", help="DEV passage ID; defaults to first complete passage")
    args = parser.parse_args()
    precompute_demo(passage_id=args.passage_id)


if __name__ == "__main__":
    main()
