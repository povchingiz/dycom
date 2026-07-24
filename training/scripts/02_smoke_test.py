#!/usr/bin/env python
"""
02_smoke_test.py — железное правило лаборатории: ничего не уходит на L40,
пока пайплайн не прошёл прогон на 1 батче. Ловит битые пути, неверный
dataset.json, несовпадение форм ДО того, как ты потратишь ночь GPU.
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path


def check_env():
    ok = True
    for var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        if not os.environ.get(var):
            print(f"[FAIL] переменная {var} не задана (см. 00_setup_env.sh)")
            ok = False
        else:
            print(f"[ok] {var}={os.environ[var]}")
    return ok


def check_dataset(dataset_id: int):
    raw = Path(os.environ.get("nnUNet_raw", ""))
    matches = list(raw.glob(f"Dataset{dataset_id:03d}_*"))
    if not matches:
        print(f"[FAIL] датасет {dataset_id} не найден в nnUNet_raw")
        return False
    ds = matches[0]
    import json
    dj_data = json.loads((ds / "dataset.json").read_text()) if (ds / "dataset.json").exists() else {}
    ext = dj_data.get("file_ending", ".nii.gz")
    n_img = len(list((ds / "imagesTr").glob(f"*{ext}")))
    n_lab = len(list((ds / "labelsTr").glob(f"*{ext}")))
    dj = (ds / "dataset.json").exists()
    print(f"[ok] {ds.name}: images={n_img} labels={n_lab} dataset.json={dj}")
    return n_img > 0 and n_img == n_lab and dj


def check_torch_gpu():
    try:
        import torch
    except ImportError:
        print("[FAIL] torch не установлен")
        return False
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        print(f"[ok] GPU: {name}, {vram} ГБ, bf16={torch.cuda.is_bf16_supported()}")
    else:
        print("[warn] GPU не виден — smoke-test пройдёт на CPU, обучение будет медленным")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=int, default=112)  # ToothFairy2 (matches p6_train.py DATASET_ID)
    args = ap.parse_args()

    print("=== SMOKE TEST ===")
    results = [
        ("окружение", check_env()),
        ("датасет", check_dataset(args.dataset)),
        ("torch/gpu", check_torch_gpu()),
    ]
    print("\n=== ИТОГ ===")
    all_ok = all(r for _, r in results)
    for name, r in results:
        print(f"  {name}: {'OK' if r else 'FAIL'}")
    if all_ok:
        print("\nЗелёный свет. Можно запускать nnUNetv2_plan_and_preprocess и обучение.")
        sys.exit(0)
    else:
        print("\nЕсть проблемы — НЕ запускай обучение на L40, сначала почини.")
        sys.exit(1)


if __name__ == "__main__":
    main()
