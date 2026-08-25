#!/usr/bin/env python3
"""Prepare EEGdenoiseNet → T2-Artifact semi-synthetic contamination set.

Builds the deterministic semi-synthetic T2 signal set:

  x_noisy = x_eeg + lambda * x_artifact
  lambda  = RMS(x_eeg) / (max(RMS(x_artifact), eps) * 10^(SNR_dB / 10))

Three balanced classes (clean / ocular_contaminated / muscle_contaminated);
contaminated classes are stratified over the EEGdenoiseNet SNR grid
{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2} dB.  Everything is driven by a
single fixed seed so the manifest is bit-reproducible.

Outputs (signals are regenerable local derivatives → git-ignored):
  workspace/processed/eegdenoisenet/signals/<record_id>.npy   (µV, float32)
  workspace/processed/eegdenoisenet/metadata/<record_id>.json
  workspace/candidate_manifests/eegdenoisenet_build.jsonl
  workspace/candidate_manifests/eegdenoisenet_provenance.json
  data/eegdenoisenet/source_metadata.json

Run:
  python3 scripts/prepare_eegdenoisenet.py
  python3 scripts/prepare_eegdenoisenet.py --quick   # tiny set for debugging
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/eegdenoisenet/source/data"
PROC_DIR = ROOT / "data/eegdenoisenet"
SIG_DIR = PROC_DIR / "signals"
META_DIR = PROC_DIR / "metadata"
MANIFEST = ROOT / "workspace/candidate_manifests/eegdenoisenet_build.jsonl"
PROVENANCE = ROOT / "workspace/candidate_manifests/eegdenoisenet_provenance.json"
LOCAL_PROV = ROOT / "data/eegdenoisenet/source_metadata.json"

SFREQ = 256.0
EPOCH_SAMPLES = 512
DURATION_SEC = EPOCH_SAMPLES / SFREQ  # 2.0
SEED = 20260622
EPS = 1e-12
SNR_GRID = tuple(range(-7, 3))

# (clean_n, per-SNR contaminated_n)  →  membership flags
POOL = {"clean": 100, "per_snr": 10}      # candidate: 100 + 100 + 100 = 300
FROZEN = {"clean": 40, "per_snr": 4}      # 40 + 40 + 40 = 120
SMOKE = {"clean": 10, "per_snr": 1}       # 10 + 10 + 10 = 30

SOURCES = {
    "clean": "EEG_all_epochs.npy",
    "ocular": "EOG_all_epochs.npy",
    "muscle": "EMG_all_epochs.npy",
}


def _snr_tag(snr: int) -> str:
    return f"m{abs(snr)}" if snr < 0 else f"p{snr}"


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _qc_ok(x: np.ndarray) -> bool:
    """Reject NaN/Inf, all-zero/near-flat, and absurd amplitude epochs."""
    if x.shape[0] != EPOCH_SAMPLES:
        return False
    if not np.all(np.isfinite(x)):
        return False
    if _rms(x) < EPS:
        return False
    return True


def _qc_indices(arr: np.ndarray) -> list[int]:
    return [i for i in range(arr.shape[0]) if _qc_ok(arr[i])]


def load_sources() -> dict[str, np.ndarray]:
    out = {}
    for key, fname in SOURCES.items():
        p = RAW_DIR / fname
        if not p.exists():
            raise FileNotFoundError(
                f"Missing source {p}.  Download EEGdenoiseNet first "
                "Download EEGdenoiseNet from its official repository first."
            )
        out[key] = np.load(p).astype(np.float64)
    return out


def build_records(quick: bool = False) -> list[dict]:
    clean_n = 6 if quick else POOL["clean"]
    per_snr = 2 if quick else POOL["per_snr"]

    rng = np.random.default_rng(SEED)
    src = load_sources()
    eeg, eog, emg = src["clean"], src["ocular"], src["muscle"]

    eeg_ok = _qc_indices(eeg)
    eog_ok = _qc_indices(eog)
    emg_ok = _qc_indices(emg)
    print(f"[qc] clean {len(eeg_ok)}/{eeg.shape[0]}  "
          f"ocular {len(eog_ok)}/{eog.shape[0]}  "
          f"muscle {len(emg_ok)}/{emg.shape[0]}")

    n_contam = per_snr * len(SNR_GRID)          # per artifact source
    n_eeg_needed = clean_n + 2 * n_contam       # distinct clean bases
    if n_eeg_needed > len(eeg_ok):
        raise ValueError(f"Need {n_eeg_needed} clean epochs, have {len(eeg_ok)}")

    eeg_pick = rng.permutation(eeg_ok)[:n_eeg_needed].tolist()
    eog_pick = rng.permutation(eog_ok)[:n_contam].tolist()
    emg_pick = rng.permutation(emg_ok)[:n_contam].tolist()

    records: list[dict] = []
    cur = 0

    # ── clean ──────────────────────────────────────────────────────
    for seq, ci in enumerate(eeg_pick[:clean_n]):
        rid = f"edn_clean_{seq:04d}"
        records.append({
            "record_id": rid, "label": "clean",
            "clean_index": int(ci), "artifact_source": None,
            "artifact_index": None, "snr_db": None, "lambda": None,
            "_stratum": "clean", "_rank": seq,
        })
    cur = clean_n

    # ── contaminated (ocular / muscle) ─────────────────────────────
    def add_contam(source_key, source_arr, picks, label, tag):
        nonlocal cur
        pi = 0
        for snr in SNR_GRID:
            for seq in range(per_snr):
                ci = eeg_pick[cur]; cur += 1
                ai = picks[pi]; pi += 1
                x_eeg = eeg[ci]
                x_art = source_arr[ai]
                lam = _rms(x_eeg) / (max(_rms(x_art), EPS) * (10.0 ** (snr / 10.0)))
                rid = f"edn_{tag}_{_snr_tag(snr)}_{seq:04d}"
                records.append({
                    "record_id": rid, "label": label,
                    "clean_index": int(ci), "artifact_source": source_key,
                    "artifact_index": int(ai), "snr_db": int(snr),
                    "lambda": float(lam),
                    "_stratum": f"{tag}_{_snr_tag(snr)}", "_rank": seq,
                })

    add_contam("EOG", eog, eog_pick, "ocular_contaminated", "eog")
    add_contam("EMG", emg, emg_pick, "muscle_contaminated", "emg")

    # ── membership flags (smoke ⊂ frozen ⊂ candidate) ──────────────
    fr_clean = 4 if quick else FROZEN["clean"]
    fr_snr = 1 if quick else FROZEN["per_snr"]
    sm_clean = 2 if quick else SMOKE["clean"]
    sm_snr = 1 if quick else SMOKE["per_snr"]
    for r in records:
        is_clean = r["label"] == "clean"
        rank = r["_rank"]
        fr_cap = fr_clean if is_clean else fr_snr
        sm_cap = sm_clean if is_clean else sm_snr
        r["in_frozen"] = rank < fr_cap
        r["in_smoke"] = rank < sm_cap

    return records, src


def synthesize(records: list[dict], src: dict[str, np.ndarray]) -> None:
    SIG_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    for p in SIG_DIR.glob("edn_*.npy"):
        p.unlink()
    for p in META_DIR.glob("edn_*.json"):
        p.unlink()
    eeg = src["clean"]
    art_map = {"EOG": src["ocular"], "EMG": src["muscle"]}

    for r in records:
        x = eeg[r["clean_index"]].astype(np.float64)
        if r["artifact_source"] is not None:
            x = x + r["lambda"] * art_map[r["artifact_source"]][r["artifact_index"]]
        sig = x.astype(np.float32)
        np.save(SIG_DIR / f"{r['record_id']}.npy", sig)

        meta = {
            "dataset": "EEGdenoiseNet",
            "record_id": r["record_id"],
            "label": r["label"],
            "sampling_rate": SFREQ,
            "n_samples": EPOCH_SAMPLES,
            "duration_sec": DURATION_SEC,
            "channels": 1,
            "channel_names": ["EEG"],
            "clean_index": r["clean_index"],
            "artifact_source": r["artifact_source"],
            "artifact_index": r["artifact_index"],
            "snr_db": r["snr_db"],
            "lambda": r["lambda"],
            "seed": SEED,
        }
        (META_DIR / f"{r['record_id']}.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")


def write_manifest(records: list[dict]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        for r in records:
            row = {
                "record_id": r["record_id"],
                "label": r["label"],
                "artifact_source": r["artifact_source"],
                "snr_db": r["snr_db"],
                "lambda": r["lambda"],
                "clean_index": r["clean_index"],
                "artifact_index": r["artifact_index"],
                "sampling_rate": SFREQ,
                "n_samples": EPOCH_SAMPLES,
                "duration_sec": DURATION_SEC,
                "signal_path": f"data/eegdenoisenet/signals/{r['record_id']}.npy",
                "in_frozen": r["in_frozen"],
                "in_smoke": r["in_smoke"],
                "seed": SEED,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_provenance() -> None:
    repo = RAW_DIR.parent
    files = {}
    for key, fname in SOURCES.items():
        p = RAW_DIR / fname
        arr = np.load(p, mmap_mode="r")
        files[fname] = {
            "role": key, "shape": list(arr.shape), "dtype": str(arr.dtype),
            "sha256": _sha256(p),
        }
    prov = {
        "dataset": "EEGdenoiseNet",
        "source_url": "https://gin.g-node.org/NCClab/EEGdenoiseNet",
        "mirror_url": "https://github.com/ncclabsustech/EEGdenoiseNet",
        "paper": "https://arxiv.org/abs/2009.11662",
        "license": "CC0 1.0 Public Domain Dedication",
        "commit": _git_commit(repo),
        "download_date": "2026-06-22",
        "registered_date": date.today().isoformat(),
        "sampling_rate": SFREQ,
        "epoch_samples": EPOCH_SAMPLES,
        "duration_sec": DURATION_SEC,
        "seed": SEED,
        "snr_grid_db": list(SNR_GRID),
        "synthesis": "x_noisy = x_eeg + lambda * x_artifact ; "
                     "lambda = rms_eeg / (rms_artifact * 10^(snr/10))",
        "source_files": files,
        "missing_files_backfilled_from_github_raw": [],
    }
    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    LOCAL_PROV.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_PROV.write_text(json.dumps(prov, indent=2), encoding="utf-8")


def summarise(records: list[dict]) -> None:
    from collections import Counter
    lab = Counter(r["label"] for r in records)
    snr = Counter((r["label"], r["snr_db"]) for r in records if r["snr_db"] is not None)
    print(f"[done] {len(records)} records")
    print("  labels:", dict(lab))
    print("  by snr:", {f"{k[0]}@{k[1]}": v for k, v in sorted(snr.items())})
    print(f"  frozen={sum(r['in_frozen'] for r in records)}  "
          f"smoke={sum(r['in_smoke'] for r in records)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny debug set")
    args = ap.parse_args()

    records, src = build_records(quick=args.quick)
    synthesize(records, src)
    write_manifest(records)
    write_provenance()
    summarise(records)
    print(f"  manifest:   {MANIFEST.relative_to(ROOT)}")
    print(f"  provenance: {PROVENANCE.relative_to(ROOT)}")
    print(f"  signals:    {SIG_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
