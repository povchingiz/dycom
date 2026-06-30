"""
Phase 6 — ML Training Pipeline

Sub-steps (each idempotent, resumes from last completed):
  download  → download ToothFairy2 (~15GB) via HuggingFace or Zenodo
  prepare   → convert to nnU-Net folder structure + dataset.json
  smoke     → 1-batch CPU check before burning GPU time
  train     → nnUNetv2_train on GPU (overnight on L40 ~12h)
  evaluate  → Dice + HD95 on held-out → metrics.json (the artifact)

Required env vars (set in .env or shell):
  nnUNet_raw, nnUNet_preprocessed, nnUNet_results
  (run ml_lab_cbct/experiments/cbct_seg/scripts/00_setup_env.sh to set these)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from pipeline.phases.base import Phase

DATASET_ID = 112
DATASET_NAME = "ToothFairy2"

# ToothFairy2 label map (42 classes: teeth + jaw structures)
TF2_LABELS = {
    "background": 0,
    "mandible": 1,
    "upper_jawbone": 2,
    "lower_left_central_incisor": 3,
    "lower_left_lateral_incisor": 4,
    "lower_left_canine": 5,
    "lower_left_first_premolar": 6,
    "lower_left_second_premolar": 7,
    "lower_left_first_molar": 8,
    "lower_left_second_molar": 9,
    "lower_left_third_molar": 10,
    "lower_right_central_incisor": 11,
    "lower_right_lateral_incisor": 12,
    "lower_right_canine": 13,
    "lower_right_first_premolar": 14,
    "lower_right_second_premolar": 15,
    "lower_right_first_molar": 16,
    "lower_right_second_molar": 17,
    "lower_right_third_molar": 18,
    "upper_left_central_incisor": 19,
    "upper_left_lateral_incisor": 20,
    "upper_left_canine": 21,
    "upper_left_first_premolar": 22,
    "upper_left_second_premolar": 23,
    "upper_left_first_molar": 24,
    "upper_left_second_molar": 25,
    "upper_left_third_molar": 26,
    "upper_right_central_incisor": 27,
    "upper_right_lateral_incisor": 28,
    "upper_right_canine": 29,
    "upper_right_first_premolar": 30,
    "upper_right_second_premolar": 31,
    "upper_right_first_molar": 32,
    "upper_right_second_molar": 33,
    "upper_right_third_molar": 34,
}


class Phase6Train(Phase):
    name = "phase6_ml_training"

    # ── availability ────────────────────────────────────────────────────

    def is_available(self) -> tuple[bool, str]:
        if not shutil.which("nnUNetv2_train"):
            return False, "nnU-Net not installed — run: pip install nnunetv2"
        for var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
            if not os.getenv(var):
                return False, (
                    f"{var} not set — run:\n"
                    "  source ml_lab_cbct/experiments/cbct_seg/scripts/00_setup_env.sh\n"
                    "  (or add nnUNet_raw / nnUNet_preprocessed / nnUNet_results to .env)"
                )
        return True, ""

    def artifacts_exist(self, data_dir: Path) -> bool:
        results = os.getenv("nnUNet_results", "")
        if not results:
            return False
        ds_pattern = f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
        metrics = list(Path(results).glob(f"{ds_pattern}/*/fold_0/validation/summary.json"))
        return bool(metrics)

    # ── main entry ───────────────────────────────────────────────────────

    def run(self, state, data_dir: Path) -> dict:
        raw = Path(os.environ["nnUNet_raw"])
        ds_dir = raw / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"

        steps = {
            "download": self._download,
            "prepare":  self._prepare,
            "smoke":    self._smoke,
            "train":    self._train,
            "evaluate": self._evaluate,
        }

        for step_name, fn in steps.items():
            if self._step_done(state, step_name):
                print(f"[phase6/{step_name}] already done, skipping")
                continue
            print(f"[phase6/{step_name}] starting...")
            result = fn(state, data_dir, ds_dir)
            self._mark_step(state, step_name, result)

        return {"dataset": DATASET_ID, "name": DATASET_NAME, "config": "3d_fullres", "fold": 0}

    # ── step: download ───────────────────────────────────────────────────

    def _download(self, state, data_dir: Path, ds_dir: Path) -> dict:
        raw_download = data_dir / "raw" / "datasets" / "toothfairy2"

        # If already in nnUNet format, skip download
        if ds_dir.exists() and (ds_dir / "dataset.json").exists():
            print(f"[phase6/download] dataset already in nnUNet_raw, skipping download")
            return {"source": "already_present", "path": str(ds_dir)}

        if raw_download.exists() and any(raw_download.rglob("*.nii.gz")):
            print(f"[phase6/download] raw files found in {raw_download}, skipping download")
            return {"source": "already_present", "path": str(raw_download)}

        raw_download.mkdir(parents=True, exist_ok=True)

        # Try HuggingFace (several known repo names for ToothFairy2)
        hf_candidates = [
            "toothfairy/ToothFairy2",
            "ditto-biomed/toothfairy2",
            "toothfairy2/dataset",
        ]
        for repo_id in hf_candidates:
            try:
                print(f"[phase6/download] trying HuggingFace: {repo_id}")
                from huggingface_hub import snapshot_download
                path = snapshot_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    local_dir=str(raw_download),
                )
                print(f"[phase6/download] HuggingFace OK: {path}")
                return {"source": "huggingface", "repo": repo_id, "path": str(raw_download)}
            except Exception as e:
                print(f"  failed ({type(e).__name__})")

        # Try Zenodo (ToothFairy2 challenge dataset)
        zenodo_urls = [
            "https://zenodo.org/records/8386688/files/ToothFairy2_dataset.zip",
            "https://zenodo.org/records/11182955/files/ToothFairy2_dataset.zip",
        ]
        for url in zenodo_urls:
            try:
                print(f"[phase6/download] trying Zenodo: {url}")
                import urllib.request, zipfile
                zip_path = raw_download / "toothfairy2.zip"

                def _progress(count, block, total):
                    if total > 0:
                        pct = min(count * block * 100 // total, 100)
                        print(f"\r  {pct}%", end="", flush=True)

                urllib.request.urlretrieve(url, str(zip_path), reporthook=_progress)
                print()
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(raw_download)
                zip_path.unlink()
                print(f"[phase6/download] Zenodo OK: {raw_download}")
                return {"source": "zenodo", "path": str(raw_download)}
            except Exception as e:
                print(f"  failed ({type(e).__name__}: {e})")

        # Manual fallback — clear instructions
        raise RuntimeError(
            "\n"
            "Auto-download failed (dataset may require registration).\n"
            "\n"
            "Manual steps:\n"
            "  1. Go to https://toothfairy2.grand-challenge.org/  (free registration)\n"
            "  2. Download dataset (~15 GB)\n"
            "  3. Extract to: data/raw/datasets/toothfairy2/\n"
            "  4. Re-run:\n"
            "       python pipeline/main.py --reset-phase 6\n"
            "       python pipeline/main.py --phase 6\n"
            "\n"
            "Alternative: HaN-Seg (8GB, no registration, for soft tissue validation):\n"
            "  python pipeline/main.py --download han_seg"
        )

    # ── step: prepare ────────────────────────────────────────────────────

    def _prepare(self, state, data_dir: Path, ds_dir: Path) -> dict:
        # Data already in nnUNet_raw (e.g. uploaded directly to server)
        if (ds_dir / "imagesTr").exists() and (ds_dir / "dataset.json").exists():
            print("[phase6/prepare] dataset already in nnUNet_raw, skipping conversion")
            self._patch_dataset_json(ds_dir)
            return {"ds_dir": str(ds_dir), "method": "already_in_nnunet_raw"}

        raw_download = data_dir / "raw" / "datasets" / "toothfairy2"

        # If dataset already in nnUNet format (HF download may give this directly)
        if (raw_download / "imagesTr").exists() and (raw_download / "dataset.json").exists():
            print("[phase6/prepare] dataset already in nnUNet format, copying...")
            if ds_dir.exists():
                shutil.rmtree(ds_dir)
            shutil.copytree(raw_download, ds_dir)
            # Ensure dataset.json has correct ID/name
            self._patch_dataset_json(ds_dir)
            return {"ds_dir": str(ds_dir), "method": "copy_from_hf"}

        # Otherwise convert raw NIfTI pairs to nnUNet structure
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        return self._convert_to_nnunet(raw_download, ds_dir)

    def _convert_to_nnunet(self, raw: Path, ds_dir: Path) -> dict:
        import random
        ds_dir.mkdir(parents=True, exist_ok=True)
        img_tr = ds_dir / "imagesTr"
        lab_tr = ds_dir / "labelsTr"
        img_tr.mkdir(exist_ok=True)
        lab_tr.mkdir(exist_ok=True)

        # Find image/label pairs (ToothFairy2 structure)
        pairs = self._find_tf2_pairs(raw)
        if not pairs:
            raise RuntimeError(
                f"No image/label pairs found in {raw}.\n"
                "Expected structure: images/*.nii.gz + labels/*.nii.gz\n"
                "Adjust _find_tf2_pairs() if your download has different layout."
            )

        print(f"[phase6/prepare] found {len(pairs)} pairs, converting...")
        for i, (img, label) in enumerate(pairs):
            case = f"TF2_{i:04d}"
            shutil.copy(img, img_tr / f"{case}_0000.nii.gz")
            shutil.copy(label, lab_tr / f"{case}.nii.gz")

        self._write_dataset_json(ds_dir, len(pairs))
        print(f"[phase6/prepare] done → {ds_dir}")
        print(f"[phase6/prepare] next: nnUNetv2_plan_and_preprocess -d {DATASET_ID}")
        return {"ds_dir": str(ds_dir), "n_cases": len(pairs)}

    def _find_tf2_pairs(self, raw: Path) -> list[tuple[Path, Path]]:
        """Find (image, label) pairs. Handles common ToothFairy2 layouts."""
        images = sorted(raw.rglob("*.nii.gz"))
        pairs = []
        for img in images:
            if "label" in img.name.lower() or "seg" in img.name.lower():
                continue
            # Common layouts: labels/ sibling dir, or _seg suffix
            candidates = [
                img.parent.parent / "labels" / img.name,
                img.parent.parent / "labelsTs" / img.name,
                img.with_name(img.name.replace(".nii.gz", "_seg.nii.gz")),
                img.parent / "labels" / img.name,
            ]
            label = next((c for c in candidates if c.exists()), None)
            if label:
                pairs.append((img, label))
        return pairs

    def _discover_labels(self, ds_dir: Path, ext: str) -> dict:
        """Scan all label files for global max value, declare labels 0..max_label."""
        import SimpleITK as sitk
        label_files = sorted((ds_dir / "labelsTr").glob(f"*{ext}"))
        max_label = 0
        for i, f in enumerate(label_files):
            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(f)))
            max_label = max(max_label, int(arr.max()))
            if i % 50 == 0:
                print(f"[phase6/patch] scanning labels {i}/{len(label_files)}, max so far={max_label}")
        labels = {"background": 0}
        for v in range(1, max_label + 1):
            labels[f"label_{v:03d}"] = v
        print(f"[phase6/patch] {len(labels)} labels declared (0..{max_label})")
        return labels

    def _write_dataset_json(self, ds_dir: Path, n: int):
        (ds_dir / "dataset.json").write_text(json.dumps({
            "channel_names": {"0": "CT"},
            "labels": TF2_LABELS,
            "numTraining": n,
            "file_ending": ".nii.gz",
            "overwrite_image_reader_writer": "SimpleITKIO",
        }, indent=2))

    def _patch_dataset_json(self, ds_dir: Path):
        """Ensure dataset.json has correct labels, file_ending, and numTraining."""
        dj_path = ds_dir / "dataset.json"
        if dj_path.exists():
            dj = json.loads(dj_path.read_text())
        else:
            dj = {}
        dj.setdefault("channel_names", {"0": "CT"})
        dj.setdefault("file_ending", ".nii.gz")
        dj.setdefault("overwrite_image_reader_writer", "SimpleITKIO")
        # Rebuild labels from actual unique values in label files
        # (our earlier patch overwrote the original labels with a wrong mapping)
        labels = self._discover_labels(ds_dir, dj.get("file_ending", ".nii.gz"))
        dj["labels"] = labels
        # Count actual training files and fix numTraining (original dataset.json may have 0)
        ext = dj.get("file_ending", ".nii.gz")
        n_actual = len(list((ds_dir / "imagesTr").glob(f"*{ext}")))
        if n_actual > 0:
            dj["numTraining"] = n_actual
        dj_path.write_text(json.dumps(dj, indent=2))

    # ── step: smoke ──────────────────────────────────────────────────────

    def _smoke(self, state, data_dir: Path, ds_dir: Path) -> dict:
        """Run the existing smoke test script before burning GPU time."""
        smoke_script = (
            Path(__file__).parent.parent.parent
            / "ml_lab_cbct/experiments/cbct_seg/scripts/02_smoke_test.py"
        )
        result = subprocess.run(
            [sys.executable, str(smoke_script), f"--dataset={DATASET_ID}"],
            capture_output=True, text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError("Smoke test failed — fix issues before launching GPU training")
        return {"status": "passed"}

    # ── step: train ──────────────────────────────────────────────────────

    def _train(self, state, data_dir: Path, ds_dir: Path) -> dict:
        # Plan and preprocess first (idempotent)
        print("[phase6/train] running nnUNetv2_plan_and_preprocess...")
        subprocess.run(
            ["nnUNetv2_plan_and_preprocess", "-d", str(DATASET_ID), "-np", "2"],
            check=True,
        )

        # Generate ResEncUNetL plan (planning only, preprocessing already done above)
        subprocess.run(
            ["nnUNetv2_plan_experiment", "-d", str(DATASET_ID), "-pl", "nnUNetResEncUNetLPlanner"],
            check=True,
        )

        print("[phase6/train] launching nnUNetv2_train (this will run overnight)...")
        subprocess.run(
            [
                "nnUNetv2_train", str(DATASET_ID), "3d_fullres", "0",
                "-p", "nnUNetResEncUNetLPlans",
                "--npz",
            ],
            check=True,
        )
        return {"config": "3d_fullres", "plan": "nnUNetResEncUNetLPlans", "fold": 0}

    # ── step: evaluate ───────────────────────────────────────────────────

    def _evaluate(self, state, data_dir: Path, ds_dir: Path) -> dict:
        results_root = Path(os.environ["nnUNet_results"])
        ds_pattern = f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"

        pred_dirs = list(results_root.glob(f"{ds_pattern}/*/fold_0/validation"))
        if not pred_dirs:
            raise RuntimeError("No validation predictions found — training may not be complete")
        pred_dir = pred_dirs[0]

        gt_dir = ds_dir / "labelsTs"  # held-out labels
        if not gt_dir.exists():
            # nnUNet puts held-out in preprocessed
            gt_dir = Path(os.environ["nnUNet_preprocessed"]) / ds_pattern / "gt_segmentations"

        eval_script = (
            Path(__file__).parent.parent.parent
            / "ml_lab_cbct/experiments/cbct_seg/scripts/04_evaluate.py"
        )
        out_metrics = results_root / ds_pattern / "metrics.json"
        subprocess.run(
            [
                sys.executable, str(eval_script),
                "--pred-dir", str(pred_dir),
                "--gt-dir", str(gt_dir),
                "--labels", ",".join(str(v) for v in TF2_LABELS.values() if v > 0),
                "--out", str(out_metrics),
            ],
            check=True,
        )
        metrics = json.loads(out_metrics.read_text())
        print(f"[phase6/evaluate] mean Dice = {metrics.get('mean_dice')} on {metrics.get('n_cases')} cases")
        return {"metrics": str(out_metrics), "mean_dice": metrics.get("mean_dice")}

    # ── state helpers ─────────────────────────────────────────────────────

    def _step_done(self, state, step: str) -> bool:
        return state.phase(self.name).get("steps", {}).get(step) is not None

    def _mark_step(self, state, step: str, result: dict):
        info = state.phase(self.name)
        steps = info.get("steps", {})
        steps[step] = result
        # Keep overall status as "running" until all done
        all_steps = ["download", "prepare", "smoke", "train", "evaluate"]
        status = "complete" if all(steps.get(s) for s in all_steps) else "running"
        state.mark(self.name, status, steps=steps)
