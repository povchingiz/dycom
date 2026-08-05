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
  (run training/scripts/00_setup_env.sh to set these)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from pipeline.phases.base import Phase

# ── run config (env-overridable so ONE codebase runs on server / 3080Ti / Colab) ─
# All values print themselves at launch; override via env, never via input()
# (training runs headless — a prompt would hang it).
#
#   TF2_CLASSES     7  → remap 48→7 grouped classes (demo). 48 → train raw as-is.
#   TF2_MAX_CASES   60 → subset the dataset (full ToothFairy2 is ~110GB). 0 = all.
#   TF2_CONFIG      3d_lowres → nnU-Net config (lowres = fast + low-mem for demo).
#   TF2_HF_REPO     povchingiz/stomato2 → HF source (HF_TOKEN needed if private).
#   TF2_DATASET_ID  auto: 113 when remapping (7-class), 112 when raw (48-class).
N_CLASSES = int(os.getenv("TF2_CLASSES", "7"))
MAX_CASES = int(os.getenv("TF2_MAX_CASES", "60"))
NNUNET_CONFIG = os.getenv("TF2_CONFIG", "3d_lowres")
HF_REPO = os.getenv("TF2_HF_REPO", "povchingiz/stomato2")

_REMAP = N_CLASSES == 7
DATASET_ID = int(os.getenv("TF2_DATASET_ID", "113" if _REMAP else "112"))
DATASET_NAME = "ToothFairy2_grouped" if _REMAP else "ToothFairy2"

# 7-class grouped scheme (see remap_labels.py / project memory). Individual FDI
# teeth (11–28 upper, 31–48 lower) collapse to upper_teeth/lower_teeth; jaws and
# both inferior alveolar canals are kept; everything else → background.
GROUPED_LABELS = {
    "background": 0,
    "mandible": 1,
    "maxilla": 2,
    "left_canal": 3,
    "right_canal": 4,
    "upper_teeth": 5,
    "lower_teeth": 6,
}

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


class _TrainProgress:
    """Turns nnU-Net's verbose per-epoch stdout into one live progress line.

    nnU-Net v2 prints, per epoch, lines like:
        Epoch 5
        train_loss -0.7234
        val_loss -0.6891
        Pseudo dice [0.89, 0.85, 0.83]
        Epoch time: 142.31 s
    We parse those and render a single updating line:
        Epoch  5/1000 [██········] 0.5% | loss -0.72 | dice 0.857 | 142s/ep | ETA 39.2h

    feed(line) returns True if the line was consumed into the bar (caller should
    NOT also print it), False if it's an unrecognized line (caller prints it raw,
    so warnings/errors/tracebacks are never swallowed).
    """
    import re as _re
    # A number, optionally wrapped as np.float32(...) / np.float64(...).
    _NUM     = _re.compile(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?")
    _NUMCAP  = r"(?:np\.float\d*\()?(-?\d+\.?\d*)\)?"
    _EPOCH   = _re.compile(r"(?:^|\s)Epoch\s+(\d+)\s*$")
    _TRAIN   = _re.compile(r"train_loss[:\s]+" + _NUMCAP)
    _VAL     = _re.compile(r"val_loss[:\s]+" + _NUMCAP)
    _DICE    = _re.compile(r"[Pp]seudo dice[:\s]+\[([^\]]*)\]")
    _EPTIME  = _re.compile(r"Epoch time[:\s]+" + _NUMCAP)
    _TOTALEP = _re.compile(r"num_epochs[:\s]+(\d+)")

    def __init__(self, total_epochs: int = 1000):
        self.total = total_epochs
        self.epoch = 0
        self.train_loss = None
        self.val_loss = None
        self.dice = None
        self.ep_time = None
        self._dirty = False

    def feed(self, line: str) -> bool:
        s = line.rstrip("\n")

        m = self._TOTALEP.search(s)
        if m:
            self.total = int(m.group(1)); return False  # let config lines print

        m = self._EPOCH.search(s)
        if m:
            self.epoch = int(m.group(1)); self._dirty = True; self._render(); return True

        m = self._TRAIN.search(s)
        if m:
            self.train_loss = float(m.group(1)); self._dirty = True; self._render(); return True

        m = self._VAL.search(s)
        if m:
            self.val_loss = float(m.group(1)); self._dirty = True; self._render(); return True

        m = self._DICE.search(s)
        if m:
            # Some nnU-Net versions print numpy reprs, e.g.
            # [np.float32(0.5977), np.float32(0.61)]. Strip the np.floatNN( wrapper
            # FIRST (otherwise the "32"/"64" digits get parsed as values), then
            # pull the actual numbers.
            inner = self._re.sub(r"np\.float\d*\(", "(", m.group(1))
            vals = [float(x) for x in self._NUM.findall(inner)]
            if vals:
                self.dice = sum(vals) / len(vals)
            self._dirty = True; self._render(); return True

        m = self._EPTIME.search(s)
        if m:
            self.ep_time = float(m.group(1)); self._dirty = True; self._render(); return True

        return False

    def _render(self):
        pct = (self.epoch / self.total * 100) if self.total else 0
        filled = int(pct / 10)
        bar = "█" * filled + "·" * (10 - filled)
        parts = [f"Epoch {self.epoch:>4}/{self.total} [{bar}] {pct:4.1f}%"]
        if self.train_loss is not None:
            parts.append(f"loss {self.train_loss:+.3f}")
        if self.dice is not None:
            parts.append(f"dice {self.dice:.3f}")
        if self.ep_time is not None:
            parts.append(f"{self.ep_time:.0f}s/ep")
            remaining = (self.total - self.epoch) * self.ep_time
            parts.append(f"ETA {remaining/3600:.1f}h")
        # \r keeps it on one updating line; pad to clear any leftover chars.
        print("\r" + " | ".join(parts).ljust(90), end="", flush=True)

    def close(self):
        if self._dirty:
            print()  # move off the progress line so the final output is clean


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
                    "  source training/scripts/00_setup_env.sh\n"
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

        return {"dataset": DATASET_ID, "name": DATASET_NAME, "config": NNUNET_CONFIG, "fold": 0}

    # ── step: download ───────────────────────────────────────────────────

    def _download(self, state, data_dir: Path, ds_dir: Path) -> dict:
        """Download a subset of the 48-class ToothFairy2 from HF into a raw dir.

        Pulls only MAX_CASES cases (images + labels) from HF_REPO — the full set
        is ~110GB unpacked, unusable on Colab/small disks. Downloads into a
        "toothfairy2_raw" dir (48-class); _prepare then remaps 48→7 into ds_dir.
        HF_TOKEN in the env is used for private repos (harmless if repo is public).
        """
        raw_download = data_dir / "raw" / "datasets" / "toothfairy2_raw"

        # Already remapped into the target nnUNet dataset? nothing to do.
        if ds_dir.exists() and (ds_dir / "dataset.json").exists():
            print("[phase6/download] target dataset already present, skipping")
            return {"source": "already_present", "path": str(ds_dir)}

        # Raw subset already downloaded? skip (idempotent resume).
        existing = list((raw_download / "labelsTr").glob("*.mha")) if raw_download.exists() else []
        if existing:
            print(f"[phase6/download] raw subset present ({len(existing)} cases), skipping")
            return {"source": "already_present", "path": str(raw_download)}

        (raw_download / "imagesTr").mkdir(parents=True, exist_ok=True)
        (raw_download / "labelsTr").mkdir(parents=True, exist_ok=True)

        from huggingface_hub import HfApi, hf_hub_download
        token = os.getenv("HF_TOKEN") or None

        print(f"[phase6/download] listing {HF_REPO} ...")
        files = HfApi().list_repo_files(HF_REPO, repo_type="dataset", token=token)

        # The repo may nest data under a dataset dir (e.g.
        # "Dataset112_ToothFairy2/imagesTr/..."), so auto-detect the prefix from
        # the labelsTr path rather than assuming files sit at the repo root.
        label_files = [f for f in files
                       if "/labelsTr/" in f or f.startswith("labelsTr/")]
        label_files = [f for f in label_files if f.endswith(".mha")]
        if not label_files:
            sample = "\n  ".join(files[:15])
            raise RuntimeError(
                f"No labelsTr/*.mha found in {HF_REPO}. First files:\n  {sample}\n"
                "Check the repo layout / HF_TOKEN access."
            )
        # prefix = everything before "labelsTr/" (empty if at root)
        prefix = label_files[0].split("labelsTr/")[0]  # e.g. "Dataset112_ToothFairy2/"

        # Case ids from label paths. Prefer 'F' cases (full 48-class annotations)
        # over 'P' (partial) for cleaner training.
        cases = sorted(
            {os.path.basename(f)[:-4] for f in label_files},
            key=lambda c: (0 if "F_" in c else 1, c),
        )
        if MAX_CASES > 0:
            cases = cases[:MAX_CASES]
        print(f"[phase6/download] prefix='{prefix}' — fetching {len(cases)} cases")

        for i, case in enumerate(cases, 1):
            for rel in (f"{prefix}imagesTr/{case}_0000.mha",
                        f"{prefix}labelsTr/{case}.mha"):
                local = hf_hub_download(HF_REPO, rel, repo_type="dataset",
                                        local_dir=str(raw_download), token=token)
                # flatten: symlink/copy into raw_download/{imagesTr,labelsTr}/ so
                # downstream (_remap_48_to_7) finds a flat layout regardless of prefix
                sub = "imagesTr" if "imagesTr" in rel else "labelsTr"
                dst = raw_download / sub / os.path.basename(rel)
                if not dst.exists():
                    try:
                        os.symlink(local, dst)
                    except OSError:
                        shutil.copy(local, dst)
            if i % 10 == 0 or i == len(cases):
                print(f"  {i}/{len(cases)}")

        return {"source": "huggingface", "repo": HF_REPO, "n_cases": len(cases),
                "path": str(raw_download)}

    # ── step: prepare ────────────────────────────────────────────────────

    def _prepare(self, state, data_dir: Path, ds_dir: Path) -> dict:
        # Already prepared (e.g. server has Dataset113 built, or a resumed run).
        if (ds_dir / "imagesTr").exists() and (ds_dir / "dataset.json").exists():
            print("[phase6/prepare] target dataset already prepared, skipping")
            return {"ds_dir": str(ds_dir), "method": "already_present"}

        raw_download = data_dir / "raw" / "datasets" / "toothfairy2_raw"

        if _REMAP:
            return self._remap_48_to_7(raw_download, ds_dir)

        # 48-class path: use the raw download directly (no remap).
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        return self._convert_to_nnunet(raw_download, ds_dir)

    def _remap_48_to_7(self, raw: Path, ds_dir: Path) -> dict:
        """Collapse the 48-class ToothFairy2 labels into the 7 grouped classes.

        FDI teeth 11–28 → upper_teeth(5), 31–48 → lower_teeth(6); mandible(1),
        maxilla(2), and both IA canals(3,4) kept; all else → background. Images
        are copied verbatim; label geometry (spacing/origin) is preserved.
        """
        import numpy as np
        import SimpleITK as sitk

        lut = np.zeros(256, dtype=np.uint8)
        lut[1] = 1; lut[2] = 2; lut[3] = 3; lut[4] = 4
        for t in range(11, 29):
            lut[t] = 5           # upper quadrants 1x, 2x
        for t in range(31, 49):
            lut[t] = 6           # lower quadrants 3x, 4x

        (ds_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
        (ds_dir / "labelsTr").mkdir(parents=True, exist_ok=True)

        labels = sorted((raw / "labelsTr").glob("*.mha"))
        if not labels:
            raise RuntimeError(f"No labels in {raw/'labelsTr'} — download step incomplete?")
        print(f"[phase6/prepare] remapping {len(labels)} cases 48→7 classes")

        n = 0
        for i, lp in enumerate(labels, 1):
            stem = lp.stem                                   # ToothFairy2F_001
            img_src = raw / "imagesTr" / f"{stem}_0000.mha"
            if not img_src.exists():
                print(f"  !! missing image for {stem}, skipping")
                continue
            img = sitk.ReadImage(str(lp))
            out = lut[sitk.GetArrayFromImage(img).astype(np.uint8)]
            oi = sitk.GetImageFromArray(out); oi.CopyInformation(img)
            sitk.WriteImage(oi, str(ds_dir / "labelsTr" / lp.name), useCompression=True)
            shutil.copy(img_src, ds_dir / "imagesTr" / img_src.name)
            n += 1
            if i % 10 == 0 or i == len(labels):
                print(f"  {i}/{len(labels)}")

        (ds_dir / "dataset.json").write_text(json.dumps({
            "channel_names": {"0": "CBCT"},
            "labels": GROUPED_LABELS,
            "numTraining": n,
            "file_ending": ".mha",
            "overwrite_image_reader_writer": "SimpleITKIO",
        }, indent=2))
        print(f"[phase6/prepare] done → {ds_dir} ({n} cases, 7 classes)")
        return {"ds_dir": str(ds_dir), "n_cases": n, "classes": 7}

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
            / "training/scripts/02_smoke_test.py"
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

    # Trainer variant → controls the epoch schedule. nnU-Net defaults to 1000
    # epochs (nnUNetTrainer). On this /dev/shm-starved box each epoch is ~600s,
    # so 1000 epochs ≈ 7 days — not viable. nnU-Net ships built-in short trainers
    # (nnUNetTrainer_{250,500,...}epochs) that cut the schedule with no custom
    # code. 250 epochs (~42h here) typically reaches most of the full Dice.
    # Set to "nnUNetTrainer" for the full 1000-epoch run once the box is faster.
    TRAINER = "nnUNetTrainer_250epochs"

    # Data-augmentation worker processes. 0 = augmentation runs in the training
    # process (no forked workers, no torch shm queues, no /dev/shm dependency) —
    # required on the 64MB-shm server. On a box with real /dev/shm (3080 Ti,
    # Colab) set nnUNet_n_proc_DA=2..4 in the env for much faster epochs.
    DA_WORKERS_DEFAULT = "0"

    def _train_env(self) -> dict:
        """Environment hardened against OOM — GPU VRAM, CPU RAM, and /dev/shm.

        MALLOC_TRIM_THRESHOLD_=0 returns freed heap to the OS aggressively — the
        48-class run was OOM-killed at ~13MB over an 8GB cgroup cap; this keeps
        peak RSS from creeping past the limit. nnUNet_n_proc_DA defaults to 0
        (shm-safe) but honors an explicit env override on better machines.
        """
        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("nnUNet_n_proc_DA", self.DA_WORKERS_DEFAULT)
        env.setdefault("MALLOC_TRIM_THRESHOLD_", "0")  # keep peak RSS under cgroup cap
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("TORCHDYNAMO_DISABLE", "1")
        return env

    def _run_streamed(self, cmd: list[str], env: dict) -> tuple[int, str]:
        """Run a long command, streaming output live AND capturing tail.

        Live streaming matters for a 12h run — buffered output would be lost
        if the process is killed. We keep the last 200 lines to detect OOM.
        """
        from collections import deque
        tail: deque[str] = deque(maxlen=200)
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        progress = _TrainProgress()
        for line in proc.stdout:  # type: ignore[union-attr]
            tail.append(line)
            # Feed each line to the progress tracker. If it recognizes an epoch
            # boundary / metric, it renders a compact live bar; otherwise the raw
            # line is printed so nothing is hidden (errors, warnings, etc.).
            #
            # CRITICAL: the progress display is cosmetic and must NEVER take down
            # a 12h training run. Any parser failure degrades to raw printing.
            try:
                consumed = progress.feed(line)
            except Exception:
                consumed = False
            if not consumed:
                print(line, end="", flush=True)
        try:
            progress.close()
        except Exception:
            pass
        proc.wait()
        return proc.returncode, "".join(tail)

    @staticmethod
    def _is_trainer_missing(log: str, trainer: str) -> bool:
        """nnU-Net couldn't find the requested -tr trainer class."""
        low = log.lower()
        not_found = ("could not find" in low or "unable to locate" in low
                     or "no module" in low)
        return not_found and ("trainer" in low or trainer.lower() in low)

    @staticmethod
    def _is_shm_error(log: str) -> bool:
        """/dev/shm exhaustion — distinct from VRAM OOM, fixed by 0 DA workers."""
        low = log.lower()
        return (
            ("shared memory" in low or "shm" in low or "bus error" in low)
            and ("no space left" in low or "unable to allocate" in low or "bus error" in low)
        )

    @staticmethod
    def _is_oom(log: str) -> bool:
        markers = (
            "CUDA out of memory",
            "out of memory",
            "cuda runtime error (2)",
            "DefaultCPUAllocator: not enough memory",
            "Bus error",            # classic /dev/shm exhaustion in workers
            "shared memory",
        )
        low = log.lower()
        return any(m.lower() in low for m in markers)

    def _try_generate_resenc_plan(self, env: dict):
        """Best-effort ResEnc plan generation across nnUNet versions.

        The ResEnc planner class was renamed over nnUNet releases
        (nnUNetPlannerResEncL / nnUNetResEncUNetLPlanner / ...), and each writes a
        plans file whose identifier we need for `nnUNetv2_train -p <id>`. We try
        the known (planner_class, plans_identifier) pairs; the first that exits 0
        wins. If none work, return None and the caller uses the default plan.
        """
        candidates = [
            ("nnUNetPlannerResEncL",     "nnUNetResEncUNetLPlans"),
            ("nnUNetResEncUNetLPlanner", "nnUNetResEncUNetLPlans"),
            ("ResEncUNetLPlanner",       "nnUNetResEncUNetLPlans"),
        ]
        for planner_class, plans_id in candidates:
            print(f"[phase6/train] trying ResEnc planner: {planner_class}")
            rc = subprocess.run(
                ["nnUNetv2_plan_experiment", "-d", str(DATASET_ID), "-pl", planner_class],
                env=env, capture_output=True, text=True,
            )
            if rc.returncode == 0:
                print(f"[phase6/train] ResEnc plan generated → {plans_id}")
                return plans_id
            print(f"  {planner_class} not available in this nnUNet version")
        print("[phase6/train] no ResEnc planner available — using default nnUNetPlans")
        return None

    def _train(self, state, data_dir: Path, ds_dir: Path) -> dict:
        env = self._train_env()

        # Plan + preprocess. Skip if preprocessed data already exists (expensive,
        # ~73 min for this dataset) so resumes don't re-scan 480 volumes.
        prep_root = Path(os.environ["nnUNet_preprocessed"])
        ds_pattern = f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
        already_preprocessed = (prep_root / ds_pattern).exists() and any(
            (prep_root / ds_pattern).glob("nnUNetPlans*")
        )
        if already_preprocessed:
            print("[phase6/train] preprocessed data present, skipping plan_and_preprocess")
        else:
            print("[phase6/train] running nnUNetv2_plan_and_preprocess...")
            subprocess.run(
                ["nnUNetv2_plan_and_preprocess", "-d", str(DATASET_ID),
                 "-np", "2", "--verify_dataset_integrity"],
                check=True, env=env,
            )

        # Try to generate a ResEncL plan (better use of the 48GB L40). This is an
        # OPTIONAL optimization — the planner class name has changed across nnUNet
        # versions, so we try known variants and, if none work, fall through to
        # the default nnUNetPlans (which preprocessing already produced). A failed
        # optional plan must never block training.
        resenc_plan = self._try_generate_resenc_plan(env)

        # Launch order: ResEncL first (if we got it), then always the guaranteed
        # default. If ResEncL OOMs on a smaller card / fragmentation, we fall back
        # rather than losing the run.
        attempts = []
        if resenc_plan:
            attempts.append((resenc_plan, "ResEncL (large, tuned for L40)"))
        attempts.append(("nnUNetPlans", "default 3d_fullres (guaranteed plan)"))
        last_log = ""
        # Resolve the trainer once. Prefer the short-schedule trainer; if this
        # nnU-Net build doesn't ship it, fall back to the default 1000-epoch one
        # so training still runs (just slower). Probed on the first attempt below.
        trainer = self.TRAINER
        for plan, desc in attempts:
            print(f"[phase6/train] launching nnUNetv2_train — plan={plan}, trainer={trainer} ({desc})")
            print("[phase6/train] (streaming live; safe to detach in tmux)")
            cmd = [
                "nnUNetv2_train", str(DATASET_ID), NNUNET_CONFIG, "0",
                "-p", plan, "-tr", trainer,
                "--npz", "--c",   # --c: resume from checkpoint if present
            ]
            rc, last_log = self._run_streamed(cmd, env)
            if rc == 0:
                return {"config": NNUNET_CONFIG, "plan": plan, "fold": 0, "trainer": trainer}

            # Short-schedule trainer not present in this build → retry with the
            # default trainer (guaranteed to exist) before giving up on this plan.
            if self._is_trainer_missing(last_log, trainer) and trainer != "nnUNetTrainer":
                print(f"[phase6/train] trainer {trainer} not found — falling back to nnUNetTrainer (1000 epochs)")
                trainer = "nnUNetTrainer"
                cmd[cmd.index("-tr") + 1] = trainer
                rc, last_log = self._run_streamed(cmd, env)
                if rc == 0:
                    return {"config": NNUNET_CONFIG, "plan": plan, "fold": 0, "trainer": trainer}

            # /dev/shm exhaustion: retry the SAME plan with 0 DA workers, which
            # removes shared-memory use entirely. Only retry once (guard flag).
            if self._is_shm_error(last_log) and env.get("nnUNet_n_proc_DA") != "0":
                print(f"[phase6/train] /dev/shm exhausted — retrying {plan} with nnUNet_n_proc_DA=0 (no workers)")
                env = {**env, "nnUNet_n_proc_DA": "0"}
                self._clear_cuda_cache()
                rc, last_log = self._run_streamed(cmd, env)
                if rc == 0:
                    return {"config": NNUNET_CONFIG, "plan": plan, "fold": 0, "da_workers": 0}

            if self._is_oom(last_log):
                print(f"[phase6/train] OOM detected with {plan} — retrying with smaller plan")
                self._clear_cuda_cache()
                continue
            # Non-OOM failure: don't mask it by silently downgrading.
            raise RuntimeError(
                f"nnUNetv2_train failed (rc={rc}, plan={plan}).\n"
                f"Last log lines:\n{last_log[-1500:]}"
            )

        raise RuntimeError(
            "Training ran out of memory even on the default plan. Reduce further:\n"
            "  - lower batch/patch via a custom plan, or\n"
            "  - export nnUNet_n_proc_DA=1 to cut dataloader RAM/shm, or\n"
            "  - free VRAM (nvidia-smi) — another process may be resident.\n"
            f"Last log lines:\n{last_log[-1500:]}"
        )

    @staticmethod
    def _clear_cuda_cache():
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

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

        # Use the labels ACTUALLY declared in dataset.json (rebuilt by
        # _discover_labels to 0..max), not the static TF2_LABELS which caps at
        # 34 and would silently ignore the higher classes in this dataset.
        dj = json.loads((ds_dir / "dataset.json").read_text())
        label_values = sorted(v for v in dj.get("labels", TF2_LABELS).values() if int(v) > 0)
        file_ending = dj.get("file_ending", ".nii.gz")

        eval_script = (
            Path(__file__).parent.parent.parent
            / "training/scripts/04_evaluate.py"
        )
        out_metrics = results_root / ds_pattern / "metrics.json"
        subprocess.run(
            [
                sys.executable, str(eval_script),
                "--pred-dir", str(pred_dir),
                "--gt-dir", str(gt_dir),
                "--labels", ",".join(str(int(v)) for v in label_values),
                "--file-ending", file_ending,
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
