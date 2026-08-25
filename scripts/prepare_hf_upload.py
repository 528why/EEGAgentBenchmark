#!/usr/bin/env python3
"""Build a Hugging Face dataset upload tree without duplicating signal data."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT.parent / "EEGAgentBenchmark-HF-upload"
METADATA_FILES = ("files.jsonl",)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stage_file(source: Path, destination: Path, copy_files: bool) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise ValueError(f"wrong-size staged file: {destination}")
        return "existing"
    if copy_files:
        shutil.copy2(source, destination)
        return "copied"
    try:
        os.link(source, destination)
        return "linked"
    except OSError as error:
        raise OSError(
            f"cannot hard-link {source} to {destination}; rerun with --copy"
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy signal files instead of using same-filesystem hard links.",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output = args.output.resolve()
    manifest_dir = source_root / "benchmark" / "manifests"
    file_rows = read_jsonl(manifest_dir / "files.jsonl")

    expected_count = 1034
    expected_bytes = 24_244_274_188
    if len(file_rows) != expected_count:
        raise ValueError(f"expected {expected_count} files, found {len(file_rows)}")
    total_bytes = sum(int(row["size_bytes"]) for row in file_rows)
    if total_bytes != expected_bytes:
        raise ValueError(f"expected {expected_bytes} bytes, found {total_bytes}")

    counts = {"linked": 0, "copied": 0, "existing": 0}
    for row in file_rows:
        relative_path = Path(str(row["data_path"]))
        source = source_root / relative_path
        if not source.is_file() or source.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"missing or wrong-size source file: {source}")
        action = stage_file(source, output / relative_path, args.copy)
        counts[action] += 1

    metadata_dir = output / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for filename in METADATA_FILES:
        shutil.copy2(manifest_dir / filename, metadata_dir / filename)

    print(
        f"Prepared {output}: {len(file_rows)} signal files, {total_bytes} bytes "
        f"({counts['linked']} linked, {counts['copied']} copied, "
        f"{counts['existing']} existing)."
    )


if __name__ == "__main__":
    main()
