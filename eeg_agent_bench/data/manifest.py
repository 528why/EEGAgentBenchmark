"""Dataset manifest: indexing records, labels, reports."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ManifestEntry:
    """A single record in the dataset manifest."""
    record_id: str
    dataset: str
    dataset_version: str = ""
    label: str = ""
    split: str = ""  # train | eval | test
    subject_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    report_text: str = ""
    finding_tags: list[str] = field(default_factory=list)
    file_path: str = ""
    checksum: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "label": self.label,
            "split": self.split,
            "subject_id": self.subject_id,
            "metadata": self.metadata,
            "report_text": self.report_text,
            "finding_tags": self.finding_tags,
            "file_path": self.file_path,
            "checksum": self.checksum,
        }


class DatasetManifest:
    """Collection of manifest entries for a dataset."""

    def __init__(self, dataset: str, version: str = ""):
        self.dataset = dataset
        self.version = version
        self.entries: dict[str, ManifestEntry] = {}

    def add(self, entry: ManifestEntry) -> None:
        self.entries[entry.record_id] = entry

    def get(self, record_id: str) -> ManifestEntry:
        if record_id not in self.entries:
            raise KeyError(f"Record '{record_id}' not found in manifest for {self.dataset}.")
        return self.entries[record_id]

    def list_record_ids(self) -> list[str]:
        return list(self.entries.keys())

    def filter_by_label(self, label: str) -> list[ManifestEntry]:
        return [e for e in self.entries.values() if e.label == label]

    def filter_by_split(self, split: str) -> list[ManifestEntry]:
        return [e for e in self.entries.values() if e.split == split]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for entry in self.entries.values():
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "DatasetManifest":
        path = Path(path)
        manifest = cls(dataset="", version="")
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                entry = ManifestEntry(**{k: v for k, v in d.items() if k in ManifestEntry.__dataclass_fields__})
                if not manifest.dataset:
                    manifest.dataset = entry.dataset
                    manifest.version = entry.dataset_version
                manifest.add(entry)
        return manifest

    def __len__(self) -> int:
        return len(self.entries)
