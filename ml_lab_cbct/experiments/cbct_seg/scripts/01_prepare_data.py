#!/usr/bin/env python
"""
01_prepare_data.py — конвертация скачанного ToothFairy3 в формат nnU-Net v2.

ВАЖНО: скрипт НЕ скачивает данные (нужна регистрация на ditto.ing.unimore.it).
Он берёт уже скачанные NIfTI-тома и раскладывает в структуру nnU-Net.

ToothFairy3: NIfTI, 0.3мм изотропно, Hounsfield Units, многоклассовая разметка.
nnU-Net сам сделает нормализацию CT, ресэмплинг и планирование — наша задача
лишь дать ему правильную структуру папок и dataset.json.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path


def find_pairs(raw: Path):
    """Ищем пары (изображение, метка). ToothFairy3 хранит тома и сегментации;
    имена начинаются с P (Set A) или F (Set B). Подстрой паттерн под то,
    как реально лежит у тебя после скачивания/распаковки."""
    images = sorted(raw.rglob("*.nii.gz"))
    pairs = []
    for img in images:
        # эвристика: метка лежит рядом в папке labels/ или с суффиксом _seg
        cand = [
            img.parent.parent / "labels" / img.name,
            img.with_name(img.stem.replace(".nii", "") + "_seg.nii.gz"),
        ]
        label = next((c for c in cand if c.exists()), None)
        if label:
            pairs.append((img, label))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path, help="папка со скачанным ToothFairy3")
    ap.add_argument("--out", required=True, type=Path, help="nnUNet_raw корень")
    ap.add_argument("--dataset-id", default=301, type=int)
    ap.add_argument("--dataset-name", default="ToothFairy3")
    ap.add_argument("--val-fraction", default=0.1, type=float, help="9:1 как в литературе")
    args = ap.parse_args()

    ds_dir = args.out / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    img_tr = ds_dir / "imagesTr"; lab_tr = ds_dir / "labelsTr"
    for d in (img_tr, lab_tr):
        d.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(args.raw)
    if not pairs:
        raise SystemExit(
            "Пары изображение/метка не найдены. Проверь --raw и подстрой find_pairs() "
            "под реальную структуру распакованного ToothFairy3."
        )
    print(f"Найдено пар: {len(pairs)}")

    # nnU-Net именование: CASE_0000.nii.gz (изображение, модальность 0), CASE.nii.gz (метка)
    for i, (img, label) in enumerate(pairs):
        case = f"TF3_{i:04d}"
        shutil.copy(img, img_tr / f"{case}_0000.nii.gz")
        shutil.copy(label, lab_tr / f"{case}.nii.gz")

    # dataset.json. labels подставь реальные из ToothFairy3 (там десятки классов).
    # Здесь шаблон — отредактируй под нужное подмножество (например только кости+зубы).
    dataset_json = {
        "channel_names": {"0": "CT"},   # CBCT в HU -> nnU-Net CT-нормализация
        "labels": {
            "background": 0,
            "mandible": 1,
            "maxilla": 2,
            "teeth": 3,
            # ... добавь нужные классы из ToothFairy3
        },
        "numTraining": len(pairs),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    (ds_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    print(f"Готово: {ds_dir}")
    print("Дальше: nnUNetv2_plan_and_preprocess -d", args.dataset_id, "--verify_dataset_integrity")


if __name__ == "__main__":
    main()
