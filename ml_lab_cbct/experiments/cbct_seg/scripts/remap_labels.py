#!/usr/bin/env python3
"""Remap ToothFairy2 (48-class) into a 7-class grouped dataset for fast training.

Collapses individual FDI teeth into upper/lower groups, keeps jaws + IAN canals,
drops non-dental structures (sinuses, pharynx, bridge/crown/implant) to background.

Reads $nnUNet_raw/Dataset112_ToothFairy2, writes $nnUNet_raw/Dataset113_ToothFairy2_grouped.
Labels are .mha (SimpleITK); image geometry (spacing/origin/direction) is preserved.

Run on the server:  python remap_labels.py
Then:               nnUNetv2_plan_and_preprocess -d 113 --verify_dataset_integrity
"""
import os
import glob
import json
import shutil

import numpy as np
import SimpleITK as sitk

SRC_ID = "Dataset112_ToothFairy2"
DST_ID = "Dataset113_ToothFairy2_grouped"

# --- new class scheme (index -> name) ---
NEW_LABELS = {
    "background": 0,
    "mandible": 1,
    "maxilla": 2,
    "left_canal": 3,
    "right_canal": 4,
    "upper_teeth": 5,
    "lower_teeth": 6,
}

# --- source (ToothFairy2 official) -> new index ---
# everything not listed maps to 0 (background): sinuses(5,6), pharynx(7),
# bridge/crown/implant(8,9,10), and any stray index.
def build_lut():
    lut = np.zeros(256, dtype=np.uint8)  # default -> background
    lut[1] = 1  # lower jaw / mandible
    lut[2] = 2  # upper jaw / maxilla
    lut[3] = 3  # left inferior alveolar canal
    lut[4] = 4  # right inferior alveolar canal
    for t in range(11, 29):  # upper quadrants 1x, 2x
        lut[t] = 5
    for t in range(31, 49):  # lower quadrants 3x, 4x
        lut[t] = 6
    return lut


def remap_one(src_path, dst_path, lut):
    img = sitk.ReadImage(src_path)
    arr = sitk.GetArrayFromImage(img)
    out = lut[arr.astype(np.uint8)]
    out_img = sitk.GetImageFromArray(out)
    out_img.CopyInformation(img)  # preserve spacing/origin/direction
    sitk.WriteImage(out_img, dst_path, useCompression=True)


def main():
    raw = os.path.expandvars("$nnUNet_raw")
    src = os.path.join(raw, SRC_ID)
    dst = os.path.join(raw, DST_ID)
    src_lab = os.path.join(src, "labelsTr")
    src_img = os.path.join(src, "imagesTr")
    dst_lab = os.path.join(dst, "labelsTr")
    dst_img = os.path.join(dst, "imagesTr")
    os.makedirs(dst_lab, exist_ok=True)
    os.makedirs(dst_img, exist_ok=True)

    lut = build_lut()
    lab_files = sorted(glob.glob(os.path.join(src_lab, "*.mha")))
    print(f"Remapping {len(lab_files)} label files: {SRC_ID} -> {DST_ID}")

    for i, lp in enumerate(lab_files, 1):
        base = os.path.basename(lp)                 # e.g. ToothFairy2F_001.mha
        stem = base[:-4]                            # ToothFairy2F_001
        # nnU-Net image naming: <case>_0000.mha
        img_src = os.path.join(src_img, f"{stem}_0000.mha")
        if not os.path.isfile(img_src):
            print(f"  !! missing image for {stem}, skipping")
            continue
        remap_one(lp, os.path.join(dst_lab, base), lut)
        # symlink images to avoid duplicating ~GBs; fall back to copy if needed
        img_dst = os.path.join(dst_img, f"{stem}_0000.mha")
        if not os.path.exists(img_dst):
            try:
                os.symlink(img_src, img_dst)
            except OSError:
                shutil.copy2(img_src, img_dst)
        if i % 50 == 0:
            print(f"  {i}/{len(lab_files)}")

    n = len(glob.glob(os.path.join(dst_lab, "*.mha")))
    dataset_json = {
        "channel_names": {"0": "CBCT"},
        "labels": NEW_LABELS,
        "numTraining": n,
        "file_ending": ".mha",
    }
    with open(os.path.join(dst, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)
    print(f"\nDone. {n} cases written to {dst}")
    print(f"dataset.json labels: {NEW_LABELS}")


if __name__ == "__main__":
    main()
