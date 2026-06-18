"""
Dataset downloader for Phase 6.

Usage:
  python pipeline/main.py --download toothfairy2   # recommended, auto
  python pipeline/main.py --download toothfairy3   # requires registration
  python pipeline/main.py --list-datasets
"""
from __future__ import annotations
import sys
import urllib.request
import zipfile
from pathlib import Path

DATASETS: dict[str, dict] = {
    "toothfairy2": {
        "description": "480 CBCT scans, 42 classes (teeth + jaw). Main training data.",
        "hf_repo": "ditto-biomed/toothfairy2",
        "zenodo_url": "https://zenodo.org/records/8386688/files/ToothFairy2_dataset.zip",
        "requires_registration": False,
        "size_gb": 15,
    },
    "toothfairy3": {
        "description": "Extended labels. Requires free registration.",
        "hf_repo": None,
        "zenodo_url": None,
        "manual_url": "https://ditto.ing.unimore.it/toothfairy3/",
        "requires_registration": True,
        "size_gb": 25,
    },
    "han_seg": {
        "description": "42 scans, mandible + soft tissue OARs. Used for Phase 5 validation.",
        "hf_repo": None,
        "zenodo_url": "https://zenodo.org/records/7443354/files/HaN-Seg.zip",
        "requires_registration": False,
        "size_gb": 8,
    },
}


def download(dataset: str, out_dir: Path) -> bool:
    info = DATASETS.get(dataset)
    if not info:
        print(f"Unknown dataset: {dataset}. Use --list-datasets.")
        return False

    if info.get("requires_registration"):
        print(f"\n[{dataset}] requires manual registration:")
        print(f"  1. Register at: {info['manual_url']}")
        print(f"  2. Download and extract to: {out_dir}")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading {dataset} (~{info['size_gb']} GB) to {out_dir}")

    if info.get("hf_repo"):
        try:
            return _from_hf(info["hf_repo"], out_dir)
        except Exception as e:
            print(f"  HuggingFace failed ({e}), trying Zenodo...")

    if info.get("zenodo_url"):
        try:
            return _from_zenodo(info["zenodo_url"], out_dir, dataset)
        except Exception as e:
            print(f"  Zenodo failed ({e})")

    print(f"\nAuto-download failed. Manual steps:")
    url = info.get("manual_url") or info.get("zenodo_url")
    print(f"  1. Download from: {url}")
    print(f"  2. Extract to: {out_dir}")
    return False


def _from_hf(repo_id: str, out_dir: Path) -> bool:
    from huggingface_hub import snapshot_download
    print(f"  Fetching from HuggingFace: {repo_id}")
    path = snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(out_dir))
    print(f"  Done: {path}")
    return True


def _from_zenodo(url: str, out_dir: Path, name: str) -> bool:
    zip_path = out_dir / f"{name}.zip"
    print(f"  Fetching: {url}")

    def _progress(count, block, total):
        if total > 0:
            pct = min(count * block * 100 // total, 100)
            print(f"\r  {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, str(zip_path), reporthook=_progress)
    print()
    print(f"  Extracting to {out_dir}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    zip_path.unlink()
    print("  Done.")
    return True


def list_datasets():
    print("\nDatasets for Phase 6 training:\n")
    for name, info in DATASETS.items():
        reg = "  ⚠ requires registration" if info.get("requires_registration") else ""
        print(f"  {name}  ({info['size_gb']} GB){reg}")
        print(f"    {info['description']}")
    print()
