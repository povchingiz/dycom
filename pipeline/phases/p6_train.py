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
#   TF2_REQUIRE_ALL_CLASSES  1 → keep only cases annotated for every target class.
#   ToothFairy2 ships 63 fully-annotated "F" cases and 417 partial "P" cases; a
#   partial case with no maxilla label teaches the net that maxilla is background.
REQUIRE_ALL_CLASSES = os.getenv("TF2_REQUIRE_ALL_CLASSES", "1") not in ("0", "false", "False")

_REMAP = N_CLASSES == 7
DATASET_ID = int(os.getenv("TF2_DATASET_ID", "113" if _REMAP else "112"))
DATASET_NAME = "ToothFairy2_grouped" if _REMAP else "ToothFairy2"

# 7-class grouped scheme (see remap_labels.py / project memory). Individual FDI
# teeth (11–28 upper, 31–48 lower) collapse to upper_teeth/lower_teeth; jaws and
# both inferior alveolar canals are kept; everything else → background.
def _dataset_case_count(ds_dir: Path) -> int:
    """How many training images an nnU-Net dataset dir actually holds (0 if it is
    not a dataset). Used instead of "the folder exists" so a half-built dataset
    from a crashed run is rebuilt rather than trained on."""
    if not (ds_dir / "dataset.json").exists():
        return 0
    return len(list((ds_dir / "imagesTr").glob("*.mha"))) if (ds_dir / "imagesTr").exists() else 0


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink the image into the nnU-Net dataset instead of copying it.

    ToothFairy2 images are ~400MB each; at 480 cases a verbatim copy adds ~190GB
    of duplicate data on top of the download and the preprocessed set, which is
    what fills the disk mid-run. Images are read-only inputs, so a hardlink is
    equivalent. Falls back to a copy across filesystems."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy(src, dst)


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

    @staticmethod
    def _nnunet_cli(name: str = "nnUNetv2_train") -> str | None:
        """Locate an nnU-Net CLI. shutil.which only checks PATH, but `make train`
        runs `.venv312/bin/python` WITHOUT activating the venv, so the venv's bin/
        isn't on PATH and the CLIs are invisible. Check next to the running
        interpreter (the venv) first, then fall back to PATH."""
        cand = Path(sys.executable).parent / name
        if cand.exists():
            return str(cand)
        return shutil.which(name)

    def is_available(self) -> tuple[bool, str]:
        if not self._nnunet_cli():
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
        raw_download = self._raw_dir(data_dir)

        # Already remapped into the target nnUNet dataset? nothing to do — but only
        # if it holds as many cases as this run asks for. A dataset dir left over
        # from a crashed run is not a finished dataset.
        if _dataset_case_count(ds_dir) >= MAX_CASES > 0:
            print(f"[phase6/download] target dataset already present "
                  f"({_dataset_case_count(ds_dir)} cases), skipping")
            return {"source": "already_present", "path": str(ds_dir)}

        existing = list((raw_download / "labelsTr").glob("*.mha")) if raw_download.exists() else []
        if existing and MAX_CASES > 0 and len(existing) >= MAX_CASES:
            print(f"[phase6/download] raw subset present ({len(existing)} cases), skipping")
            return {"source": "already_present", "path": str(raw_download)}
        # With MAX_CASES=0 ("all") the local count alone proves nothing — an
        # interrupted download leaves a non-empty dir that used to be accepted as
        # complete, and training then started on whatever few cases had landed.
        # The repo listing below is the only way to know what "all" means.

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
        if REQUIRE_ALL_CLASSES:
            cases = self._filter_fully_annotated(cases, prefix, raw_download, token)

        if MAX_CASES > 0:
            cases = cases[:MAX_CASES]

        missing = [c for c in cases
                   if not (raw_download / "labelsTr" / f"{c}.mha").exists()
                   or not (raw_download / "imagesTr" / f"{c}_0000.mha").exists()]
        if not missing:
            print(f"[phase6/download] all {len(cases)} cases already local, skipping")
            return {"source": "already_present", "n_cases": len(cases),
                    "path": str(raw_download)}
        print(f"[phase6/download] prefix='{prefix}' — {len(cases)} cases total, "
              f"fetching {len(missing)} missing")
        cases = missing

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

    def _filter_fully_annotated(self, cases: list[str], prefix: str,
                                raw_download: Path, token: str | None) -> list[str]:
        """Keep only cases annotated for every class we are training on.

        ToothFairy2 is not 480 equivalent scans. 63 are "F" (full annotation) and
        417 are "P" (partial) — a P case may have the mandible and both alveolar
        canals but no maxilla, or only a handful of teeth. nnU-Net cannot tell
        "not annotated" from "not present", so an unlabelled maxilla is learned as
        background and the model gets worse at the class the extra data was
        supposed to improve.

        Labels are ~10x smaller than images, so this downloads all labels first,
        inspects them, and only then pulls images for the cases that survive.
        Set TF2_REQUIRE_ALL_CLASSES=0 to train on everything regardless.
        """
        import numpy as np
        import SimpleITK as sitk
        from huggingface_hub import hf_hub_download

        required = self._required_source_labels()
        print(f"[phase6/filter] checking {len(cases)} label files for classes {sorted(required)} ...")

        keep, dropped = [], []
        for i, case in enumerate(cases, 1):
            dst = raw_download / "labelsTr" / f"{case}.mha"
            if not dst.exists():
                local = hf_hub_download(HF_REPO, f"{prefix}labelsTr/{case}.mha",
                                        repo_type="dataset",
                                        local_dir=str(raw_download), token=token)
                try:
                    os.symlink(local, dst)
                except OSError:
                    shutil.copy(local, dst)
            present = set(np.unique(sitk.GetArrayFromImage(sitk.ReadImage(str(dst)))).tolist())
            missing_groups = [g for g, members in required.items() if not (members & present)]
            if missing_groups:
                dropped.append((case, missing_groups))
            else:
                keep.append(case)
            if i % 25 == 0 or i == len(cases):
                print(f"  {i}/{len(cases)}  kept={len(keep)} dropped={len(dropped)}")

        print(f"[phase6/filter] {len(keep)} fully annotated, {len(dropped)} dropped")
        for case, groups in dropped[:5]:
            print(f"    e.g. {case}: no {', '.join(groups)}")
        if not keep:
            raise RuntimeError(
                "No case carries every required class — check TF2_CLASSES or set "
                "TF2_REQUIRE_ALL_CLASSES=0."
            )
        return keep

    @staticmethod
    def _required_source_labels() -> dict[str, set[int]]:
        """Target class → the source ToothFairy2 labels that can satisfy it.
        A case passes if at least one member of every group is present."""
        if _REMAP:
            return {
                "mandible": {1},
                "maxilla": {2},
                "left_canal": {3},
                "right_canal": {4},
                "upper_teeth": set(range(11, 29)),
                "lower_teeth": set(range(31, 49)),
            }
        # 48-class mode: only demand the structural classes; expecting all 32
        # teeth in every scan would drop nearly the whole dataset.
        return {"mandible": {1}, "maxilla": {2}, "left_canal": {3}, "right_canal": {4}}

    # ── step: prepare ────────────────────────────────────────────────────

    @staticmethod
    def _raw_dir(data_dir: Path) -> Path:
        """Raw-download dir, keyed by case count so a 480-case run never reuses a
        60-case cache (which would silently train on the wrong subset)."""
        subset_tag = "all" if MAX_CASES <= 0 else str(MAX_CASES)
        return data_dir / "raw" / "datasets" / f"toothfairy2_raw_{subset_tag}"

    def _prepare(self, state, data_dir: Path, ds_dir: Path) -> dict:
        # Already prepared (e.g. server has Dataset113 built, or a resumed run)?
        # Only if it covers everything the raw dir holds — otherwise a partial
        # dataset from an aborted run silently becomes the training set.
        raw_download = self._raw_dir(data_dir)
        have = _dataset_case_count(ds_dir)
        want = len(list((raw_download / "labelsTr").glob("*.mha"))) if raw_download.exists() else 0
        if have and have >= max(want, MAX_CASES):
            print(f"[phase6/prepare] target dataset already prepared ({have} cases), skipping")
            return {"ds_dir": str(ds_dir), "method": "already_present", "n_cases": have}
        if have:
            print(f"[phase6/prepare] existing dataset has {have} cases but {max(want, MAX_CASES)} "
                  f"expected — rebuilding")

        if _REMAP:
            return self._remap_48_to_7(raw_download, ds_dir)

        # 48-class path: raw .mha files → nnU-Net dataset (no remap, just rename).
        return self._prepare_48class(raw_download, ds_dir)

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
            _link_or_copy(img_src, ds_dir / "imagesTr" / img_src.name)
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

    def _prepare_48class(self, raw: Path, ds_dir: Path) -> dict:
        """48-class path: copy raw .mha images+labels verbatim into the nnU-Net
        dataset (no remap), and declare labels 0..max found in the data."""
        import numpy as np
        import SimpleITK as sitk

        (ds_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
        (ds_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
        labels = sorted((raw / "labelsTr").glob("*.mha"))
        if not labels:
            raise RuntimeError(f"No labels in {raw/'labelsTr'} — download incomplete?")
        print(f"[phase6/prepare] preparing {len(labels)} cases (48-class, verbatim)")

        n, max_label = 0, 0
        for i, lp in enumerate(labels, 1):
            stem = lp.stem
            img_src = raw / "imagesTr" / f"{stem}_0000.mha"
            if not img_src.exists():
                print(f"  !! missing image for {stem}, skipping")
                continue
            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(lp)))
            max_label = max(max_label, int(arr.max()))
            _link_or_copy(lp, ds_dir / "labelsTr" / lp.name)
            _link_or_copy(img_src, ds_dir / "imagesTr" / img_src.name)
            n += 1
            if i % 20 == 0 or i == len(labels):
                print(f"  {i}/{len(labels)}")

        labels_map = {"background": 0}
        for v in range(1, max_label + 1):
            labels_map[f"label_{v:03d}"] = v
        (ds_dir / "dataset.json").write_text(json.dumps({
            "channel_names": {"0": "CBCT"},
            "labels": labels_map,
            "numTraining": n,
            "file_ending": ".mha",
            "overwrite_image_reader_writer": "SimpleITKIO",
        }, indent=2))
        print(f"[phase6/prepare] done → {ds_dir} ({n} cases, {len(labels_map)} labels)")
        return {"ds_dir": str(ds_dir), "n_cases": n, "classes": len(labels_map)}

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

    # Trainer variant → controls the epoch schedule. Env-overridable via
    # TF2_TRAINER. Default is the 250-epoch short trainer (fast demo); use
    # "nnUNetTrainer" for the full 1000-epoch run, or nnUNetTrainer_{500,...}epochs.
    TRAINER = os.getenv("TF2_TRAINER", "nnUNetTrainer_250epochs")

    # Data-augmentation worker processes. 0 means augmentation runs inside the
    # training process: no forked workers, no torch shm queues, no /dev/shm
    # dependency — the only thing that worked on the 64MB-shm Docker server.
    # It is also brutally slow: single-threaded CPU augmentation starves the GPU,
    # which then idles ~80% of every epoch. Pick from the machine instead of
    # hardcoding the worst case; an explicit nnUNet_n_proc_DA still wins, and
    # _is_shm_error() retries with 0 if the guess turns out to be wrong.
    SHM_PER_WORKER_MB = 512
    MAX_DA_WORKERS = 12

    @classmethod
    def _default_da_workers(cls) -> str:
        try:
            shm = shutil.disk_usage("/dev/shm")
        except OSError:
            return "0"
        by_shm = int(shm.free / (cls.SHM_PER_WORKER_MB * 1024 * 1024))
        by_cpu = max(1, (os.cpu_count() or 2) - 4)
        workers = max(0, min(by_shm, by_cpu, cls.MAX_DA_WORKERS))
        # Below ~2 workers the fork overhead is not worth it — stay in-process.
        return str(workers if workers >= 2 else 0)

    @staticmethod
    def _trainer_epochs(trainer: str) -> int:
        """Epoch budget encoded in the trainer class name.

        nnU-Net's variants are named nnUNetTrainer_{N}epochs; plain nnUNetTrainer
        is 1000. Without this the progress bar renders every run against 1000 and
        a 250-epoch job reports 4x its real ETA."""
        import re
        match = re.search(r"_(\d+)epochs", trainer)
        return int(match.group(1)) if match else 1000

    def _train_env(self) -> dict:
        """Environment hardened against OOM — GPU VRAM, CPU RAM, and /dev/shm.

        MALLOC_TRIM_THRESHOLD_=0 returns freed heap to the OS aggressively — the
        48-class run was OOM-killed at ~13MB over an 8GB cgroup cap; this keeps
        peak RSS from creeping past the limit. nnUNet_n_proc_DA defaults to 0
        (shm-safe) but honors an explicit env override on better machines.
        """
        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("nnUNet_n_proc_DA", self._default_da_workers())
        env.setdefault("MALLOC_TRIM_THRESHOLD_", "0")  # keep peak RSS under cgroup cap
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("TORCHDYNAMO_DISABLE", "1")
        # nnUNetv2_train is a separate Python process writing into our pipe, so it
        # block-buffers its stdout: the live progress bar arrives in 8KB bursts,
        # minutes apart, and a killed run loses whatever was still buffered.
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env

    def _run_streamed(self, cmd: list[str], env: dict,
                      total_epochs: int = 1000) -> tuple[int, str]:
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
        progress = _TrainProgress(total_epochs)
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
                [self._nnunet_cli("nnUNetv2_plan_experiment"), "-d", str(DATASET_ID), "-pl", planner_class],
                env=env, capture_output=True, text=True,
            )
            if rc.returncode == 0:
                print(f"[phase6/train] ResEnc plan generated → {plans_id}")
                return plans_id
            print(f"  {planner_class} not available in this nnUNet version")
        print("[phase6/train] no ResEnc planner available — using default nnUNetPlans")
        return None

    def _plan_configs(self, plans_identifier: str) -> list[str]:
        """Return the config names present in a generated plans JSON (or [])."""
        prep = Path(os.environ["nnUNet_preprocessed"])
        ds_pattern = f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
        pf = prep / ds_pattern / f"{plans_identifier}.json"
        if not pf.exists():
            return []
        try:
            return list(json.loads(pf.read_text()).get("configurations", {}).keys())
        except Exception:
            return []

    def _train(self, state, data_dir: Path, ds_dir: Path) -> dict:
        env = self._train_env()

        # Plan + preprocess. Skip if preprocessed data already exists (expensive,
        # ~73 min for this dataset) so resumes don't re-scan 480 volumes.
        prep_root = Path(os.environ["nnUNet_preprocessed"])
        ds_pattern = f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
        already_preprocessed = (prep_root / ds_pattern).exists() and any(
            (prep_root / ds_pattern).glob("nnUNetPlans*")
        )
        # "Preprocessed" means the actual data folder for our config exists, not
        # just that a plan JSON was written (train needs nnUNetPlans_<config>/).
        cfg_ready = (prep_root / ds_pattern / f"nnUNetPlans_{NNUNET_CONFIG}").exists()
        if cfg_ready:
            print(f"[phase6/train] preprocessed {NNUNET_CONFIG} present, skipping plan_and_preprocess")
        else:
            print(f"[phase6/train] running nnUNetv2_plan_and_preprocess -c {NNUNET_CONFIG} ...")
            # The planner calls torch.set_num_threads(get_allowed_n_proc_DA()),
            # which crashes on nnUNet_n_proc_DA=0 ("expects a positive integer").
            # 0 is only valid for the training loop; the planner needs ≥1.
            # -c <config> ensures the config we train on is actually preprocessed
            # (default run may skip 3d_fullres for some data → train FileNotFound).
            plan_env = {**env, "nnUNet_n_proc_DA": "2"}
            subprocess.run(
                [self._nnunet_cli("nnUNetv2_plan_and_preprocess"), "-d", str(DATASET_ID),
                 "-c", NNUNET_CONFIG, "-np", "2", "--verify_dataset_integrity"],
                check=True, env=plan_env,
            )

        # Try to generate a ResEncL plan (better use of a big GPU like the L40).
        # OPTIONAL — skipped by default because the ResEnc plan only contains
        # 2d/3d_fullres, which clashes with our 3d_lowres config (and it's tuned
        # for a 48GB card, not a T4). Set TF2_RESENC=1 to opt back in.
        resenc_plan = None
        if os.getenv("TF2_RESENC") == "1":
            resenc_plan = self._try_generate_resenc_plan(env)

        # Resolve the config against what the plan actually contains. nnU-Net
        # only generates 3d_lowres when the median image is large enough; for
        # small/low-res data it may produce just ['2d', '3d_fullres']. If our
        # requested config is absent, fall back to 3d_fullres so we still train.
        config = NNUNET_CONFIG
        avail = self._plan_configs("nnUNetPlans")
        if avail and config not in avail:
            fallback = "3d_fullres" if "3d_fullres" in avail else avail[0]
            print(f"[phase6/train] config '{config}' not in plan "
                  f"(available: {avail}) — using '{fallback}'")
            config = fallback

        # Launch order: ResEncL first (if enabled), then always the guaranteed
        # default. If ResEncL OOMs on a smaller card / fragmentation, we fall back
        # rather than losing the run.
        attempts = []
        if resenc_plan:
            attempts.append((resenc_plan, "ResEncL (large, tuned for L40)"))
        attempts.append(("nnUNetPlans", f"default {config} (guaranteed plan)"))
        last_log = ""
        # Resolve the trainer once. Prefer the short-schedule trainer; if this
        # nnU-Net build doesn't ship it, fall back to the default 1000-epoch one
        # so training still runs (just slower). Probed on the first attempt below.
        trainer = self.TRAINER
        for plan, desc in attempts:
            print(f"[phase6/train] launching nnUNetv2_train — plan={plan}, trainer={trainer} ({desc})")
            print("[phase6/train] (streaming live; safe to detach in tmux)")
            cmd = [
                self._nnunet_cli("nnUNetv2_train"), str(DATASET_ID), config, "0",
                "-p", plan, "-tr", trainer,
                "--npz", "--c",   # --c: resume from checkpoint if present
            ]
            rc, last_log = self._run_streamed(cmd, env, self._trainer_epochs(trainer))
            if rc == 0:
                return {"config": config, "plan": plan, "fold": 0, "trainer": trainer}

            # Short-schedule trainer not present in this build → retry with the
            # default trainer (guaranteed to exist) before giving up on this plan.
            if self._is_trainer_missing(last_log, trainer) and trainer != "nnUNetTrainer":
                print(f"[phase6/train] trainer {trainer} not found — falling back to nnUNetTrainer (1000 epochs)")
                trainer = "nnUNetTrainer"
                cmd[cmd.index("-tr") + 1] = trainer
                rc, last_log = self._run_streamed(cmd, env, self._trainer_epochs(trainer))
                if rc == 0:
                    return {"config": config, "plan": plan, "fold": 0, "trainer": trainer}

            # /dev/shm exhaustion: retry the SAME plan with 0 DA workers, which
            # removes shared-memory use entirely. Only retry once (guard flag).
            if self._is_shm_error(last_log) and env.get("nnUNet_n_proc_DA") != "0":
                print(f"[phase6/train] /dev/shm exhausted — retrying {plan} with nnUNet_n_proc_DA=0 (no workers)")
                env = {**env, "nnUNet_n_proc_DA": "0"}
                self._clear_cuda_cache()
                rc, last_log = self._run_streamed(cmd, env, self._trainer_epochs(trainer))
                if rc == 0:
                    return {"config": config, "plan": plan, "fold": 0, "da_workers": 0}

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
