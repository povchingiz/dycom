#!/usr/bin/env python
"""
04_evaluate.py — превращает обученную модель в ТЕСТИРУЕМЫЙ артефакт.
Считает Dice и HD95 на held-out, пишет metrics.json и таблицу абляций.
Без этого шага "обучил модель" != "артефакт".
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def dice(pred, gt, label):
    p, g = pred == label, gt == label
    inter = np.logical_and(p, g).sum()
    denom = p.sum() + g.sum()
    return 1.0 if denom == 0 else 2.0 * inter / denom


def hd95(pred, gt, label, spacing):
    """95-й перцентиль расстояния Хаусдорфа — чувствителен к границам
    (важно для качества будущего FEA-меша)."""
    try:
        from scipy.ndimage import distance_transform_edt as edt
    except ImportError:
        return None
    p, g = pred == label, gt == label
    if p.sum() == 0 or g.sum() == 0:
        return None
    # расстояния от границы одной маски до другой
    dt_g = edt(~g, sampling=spacing)
    dt_p = edt(~p, sampling=spacing)
    surf_p = p & ~_erode(p)
    surf_g = g & ~_erode(g)
    d = np.concatenate([dt_g[surf_p], dt_p[surf_g]])
    return float(np.percentile(d, 95)) if d.size else None


def _erode(mask):
    from scipy.ndimage import binary_erosion
    return binary_erosion(mask)


def load_seg(path):
    """Читает сегментацию в любом формате (.nii.gz, .mha, .nrrd ...) через
    SimpleITK. SimpleITK spacing идёт как (x,y,z), а массив — (z,y,x),
    поэтому spacing разворачиваем, чтобы совпадал с осями массива."""
    import SimpleITK as sitk
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)          # (z, y, x)
    spacing_zyx = tuple(reversed(img.GetSpacing()))
    return arr, spacing_zyx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, required=True, help="папка с предсказаниями")
    ap.add_argument("--gt-dir", type=Path, required=True, help="папка с held-out метками")
    ap.add_argument("--labels", type=str, default="1,2,3", help="классы через запятую")
    ap.add_argument("--file-ending", type=str, default=".nii.gz",
                    help="расширение масок (.nii.gz | .mha | .nrrd ...)")
    ap.add_argument("--configs", nargs="+", default=["baseline"], help="имена конфигов для таблицы")
    ap.add_argument("--out", type=Path, default=Path("metrics.json"))
    args = ap.parse_args()

    labels = [int(x) for x in args.labels.split(",")]
    ext = args.file_ending if args.file_ending.startswith(".") else f".{args.file_ending}"
    preds = sorted(args.pred_dir.glob(f"*{ext}"))
    if not preds:
        print(f"[warn] в {args.pred_dir} не найдено файлов *{ext} — проверь --file-ending")
    results = {}
    per_case = []
    for pred_path in preds:
        gt_path = args.gt_dir / pred_path.name
        if not gt_path.exists():
            continue
        pred, _ = load_seg(pred_path)
        gt, spacing = load_seg(gt_path)
        row = {"case": pred_path.name}
        for lb in labels:
            row[f"dice_{lb}"] = round(dice(pred, gt, lb), 4)
            h = hd95(pred, gt, lb, spacing)
            row[f"hd95_{lb}"] = round(h, 3) if h is not None else None
        per_case.append(row)

    # агрегаты
    for lb in labels:
        ds = [r[f"dice_{lb}"] for r in per_case if r.get(f"dice_{lb}") is not None]
        results[f"mean_dice_{lb}"] = round(float(np.mean(ds)), 4) if ds else None
    results["mean_dice"] = round(
        float(np.mean([v for k, v in results.items() if k.startswith("mean_dice_") and v])), 4
    ) if per_case else None
    results["n_cases"] = len(per_case)
    results["per_case"] = per_case

    args.out.write_text(json.dumps(results, indent=2))
    print(f"Артефакт-метрики -> {args.out}")
    print(f"mean Dice = {results['mean_dice']} на {results['n_cases']} случаях")
    print("\nЗаполни таблицу абляций в ноутбуке: предсказание vs результат.")


if __name__ == "__main__":
    main()
