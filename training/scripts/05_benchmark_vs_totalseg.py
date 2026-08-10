#!/usr/bin/env python
"""
05_benchmark_vs_totalseg.py — apples-to-apples: наш nnU-Net против TotalSegmentator.

Зачем: Phase 1 сейчас использует TotalSegmentator. Phase 6 хочет его заменить.
Прежде чем жечь ~неделю GPU на полное обучение, нужно доказать, что замена
вообще выигрывает. Меряем ОБЕ модели на одних и тех же held-out кейсах
ToothFairy2 против одной и той же ground truth.

У пациента ground truth нет, поэтому бенчмарк идёт на held-out ToothFairy2
(те же 12 кейсов, что в fold_0 validation нашей модели).

ВАЖНО — смещение в пользу TotalSegmentator: его "teeth" веса обучены на
ToothFairy3, надмножестве ToothFairy2. Эти кейсы он мог видеть при обучении.
Значит сравнение консервативно для нас: если наш nnU-Net всё равно выигрывает,
решение однозначное.

Usage:
  python training/scripts/05_benchmark_vs_totalseg.py \
      --raw   data/raw/datasets/toothfairy2_raw_all \
      --preds data/nnunet/results/Dataset113_ToothFairy2_grouped/*/fold_0/validation \
      --out   data/benchmark
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

# ── 7-классовая схема (та же, что в pipeline/phases/p6_train.py) ──────────────
GROUPED_LABELS = {
    "background": 0,
    "mandible": 1,
    "maxilla": 2,
    "left_canal": 3,
    "right_canal": 4,
    "upper_teeth": 5,
    "lower_teeth": 6,
}
LABEL_NAMES = {v: k for k, v in GROUPED_LABELS.items() if v > 0}


def gt_lut() -> np.ndarray:
    """ToothFairy2 48 классов → наши 7. FDI 11–28 → upper_teeth, 31–48 → lower."""
    lut = np.zeros(256, dtype=np.uint8)
    lut[1] = 1; lut[2] = 2; lut[3] = 3; lut[4] = 4
    for t in range(11, 29):
        lut[t] = 5
    for t in range(31, 49):
        lut[t] = 6
    return lut


def totalseg_lut() -> np.ndarray:
    """TotalSegmentator task="teeth" (77 классов) → наши 7.

    1 lower_jawbone → mandible, 2 upper_jawbone → maxilla,
    3/4 inferior alveolar canals → left/right canal,
    11–26 (FDI 11–28) → upper_teeth, 27–42 (FDI 31–48) → lower_teeth.

    Пульпы (46–77) складываем в родительский зуб: в ToothFairy2 GT пульпа не
    выделена отдельно, она входит в объём зуба. Без этого мы бы вырезали дырки
    внутри зубов TotalSegmentator'а и штрафовали его несправедливо.
    Коронки/мосты/импланты (8/9/10) → фон: в GT ToothFairy2 их нет.

    ВНИМАНИЕ — каналы меняются местами. У TotalSegmentator 3=left, 4=right;
    в разметке ToothFairy2 стороны названы наоборот. Проверено покейсово:
    dice(pred=3, gt=3)=0.00 против dice(pred=3, gt=4)=0.55. Без свопа обе
    стороны дают ровно 0 и бенчмарк врёт в нашу пользу.
    """
    lut = np.zeros(256, dtype=np.uint8)
    lut[1] = 1; lut[2] = 2
    lut[3] = 4; lut[4] = 3          # своп сторон, см. docstring
    for i in range(11, 27):      # верхние зубы FDI 11–28
        lut[i] = 5
    for i in range(27, 43):      # нижние зубы FDI 31–48
        lut[i] = 6
    for i in range(46, 62):      # пульпы верхних зубов
        lut[i] = 5
    for i in range(62, 78):      # пульпы нижних зубов
        lut[i] = 6
    return lut


# ── метрики: переиспользуем 04_evaluate.py (имя модуля с цифры — только importlib) ──
def _load_eval_module():
    path = Path(__file__).parent / "04_evaluate.py"
    spec = importlib.util.spec_from_file_location("eval04", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EV = _load_eval_module()


def read_img(path: Path):
    import SimpleITK as sitk
    img = sitk.ReadImage(str(path))
    return img, sitk.GetArrayFromImage(img)


def write_like(arr: np.ndarray, ref, path: Path):
    import SimpleITK as sitk
    out = sitk.GetImageFromArray(arr.astype(np.uint8))
    out.CopyInformation(ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(path), useCompression=True)


# ── шаги ─────────────────────────────────────────────────────────────────────
def build_gt(raw: Path, cases: list[str], out_dir: Path) -> None:
    """Метки → 7 классов, кэш на диск.

    --raw может указывать и на сырой 48-классовый ToothFairy2, и на уже
    подготовленный nnU-Net датасет (Dataset113/114), где метки уже сгруппированы.
    Прогонять LUT второй раз нельзя: lut[5] и lut[6] равны нулю, так что зубы
    молча исчезают из GT и обе модели получают Dice 0.0000 по зубам.
    """
    lut = gt_lut()
    for case in cases:
        dst = out_dir / f"{case}.mha"
        if dst.exists():
            continue
        src = raw / "labelsTr" / f"{case}.mha"
        if not src.exists():
            print(f"  !! нет GT для {case}, пропуск")
            continue
        ref, arr = read_img(src)
        write_like(arr if _already_grouped(arr) else lut[arr.astype(np.uint8)], ref, dst)
    print(f"[gt] готово → {out_dir}")


def _already_grouped(arr: np.ndarray) -> bool:
    """True, если метки уже в 7-классовой схеме (нет ни одного FDI-класса 11–48)."""
    return int(arr.max()) <= max(GROUPED_LABELS.values())


def run_totalseg(raw: Path, cases: list[str], out_dir: Path, device: str) -> None:
    """TotalSegmentator task="teeth" на тех же кейсах → маппинг в 7 классов."""
    import SimpleITK as sitk
    from totalsegmentator.python_api import totalsegmentator

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from segmentation.run_teeth_seg import normalize_device

    device = normalize_device(device)
    lut = totalseg_lut()
    tmp = out_dir.parent / "_totalseg_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    for i, case in enumerate(cases, 1):
        dst = out_dir / f"{case}.mha"
        if dst.exists():
            print(f"  [{i}/{len(cases)}] {case} — уже есть, пропуск")
            continue
        img_src = raw / "imagesTr" / f"{case}_0000.mha"
        if not img_src.exists():
            print(f"  !! нет изображения для {case}, пропуск")
            continue

        # TotalSegmentator ожидает NIfTI на входе
        nii_in = tmp / f"{case}.nii.gz"
        if not nii_in.exists():
            sitk.WriteImage(sitk.ReadImage(str(img_src)), str(nii_in), useCompression=True)

        ml_out = tmp / f"{case}_teeth.nii.gz"
        t0 = time.time()
        if ml_out.exists():
            # Raw 77-класс вывод кэшируется: переработка маппинга не должна
            # заново гонять сеть по всем кейсам.
            print(f"  [{i}/{len(cases)}] {case} — переиспользую кэш TotalSegmentator", flush=True)
        else:
            print(f"  [{i}/{len(cases)}] {case} — TotalSegmentator teeth ({device}) ...", flush=True)
            totalsegmentator(nii_in, ml_out, task="teeth", ml=True, device=device, quiet=True)

        ref, arr = read_img(ml_out)
        write_like(lut[arr.astype(np.uint8)], ref, dst)
        print(f"      {time.time() - t0:.0f}s → {dst.name}")
    print(f"[totalseg] готово → {out_dir}")


def map_nnunet(pred_dir: Path, cases: list[str], out_dir: Path,
               postprocess: bool = True) -> None:
    """Предсказания nnU-Net уже в 7 классах — копируем под общее имя.

    С postprocess=True стороны каналов переназначаются по средне-сагиттальной
    плоскости. Сеть 3d_fullres находит оба канала (Dice объединения 0.89), но
    сторону определить не может — патч вокруг одного канала неотличим от
    зеркального патча вокруг другого. См. pipeline/postprocess.py.
    """
    if postprocess:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from pipeline.postprocess import split_canal_sides

    for case in cases:
        dst = out_dir / f"{case}.mha"
        if dst.exists():
            continue
        src = pred_dir / f"{case}.mha"
        if not src.exists():
            print(f"  !! нет предсказания nnU-Net для {case}, пропуск")
            continue
        ref, arr = read_img(src)
        write_like(split_canal_sides(arr) if postprocess else arr, ref, dst)
    print(f"[nnunet] готово → {out_dir}"
          f"{' (со сплитом сторон каналов)' if postprocess else ''}")


def score(pred_dir: Path, gt_dir: Path, labels: list[int]) -> dict:
    per_case = []
    for pred_path in sorted(pred_dir.glob("*.mha")):
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            continue
        pred, _ = EV.load_seg(pred_path)
        gt, spacing = EV.load_seg(gt_path)
        row = {"case": pred_path.name}
        for lb in labels:
            row[f"dice_{lb}"] = round(EV.dice(pred, gt, lb), 4)
            h = EV.hd95(pred, gt, lb, spacing)
            row[f"hd95_{lb}"] = round(h, 3) if h is not None else None
        per_case.append(row)

    res: dict = {}
    for lb in labels:
        ds = [r[f"dice_{lb}"] for r in per_case if r.get(f"dice_{lb}") is not None]
        hs = [r[f"hd95_{lb}"] for r in per_case if r.get(f"hd95_{lb}") is not None]
        res[f"mean_dice_{lb}"] = round(float(np.mean(ds)), 4) if ds else None
        res[f"mean_hd95_{lb}"] = round(float(np.mean(hs)), 3) if hs else None
    dices = [res[f"mean_dice_{lb}"] for lb in labels if res.get(f"mean_dice_{lb}") is not None]
    res["mean_dice"] = round(float(np.mean(dices)), 4) if dices else None
    res["n_cases"] = len(per_case)
    res["per_case"] = per_case
    return res


def print_table(nn: dict, ts: dict, labels: list[int]) -> None:
    print()
    print("=" * 74)
    print(f"{'класс':<16}{'nnU-Net':>12}{'TotalSeg':>12}{'Δ Dice':>10}{'  вердикт'}")
    print("-" * 74)
    for lb in labels:
        a, b = nn.get(f"mean_dice_{lb}"), ts.get(f"mean_dice_{lb}")
        if a is None or b is None:
            print(f"{LABEL_NAMES.get(lb, lb):<16}{str(a):>12}{str(b):>12}{'—':>10}")
            continue
        d = a - b
        verdict = "nnU-Net" if d > 0.01 else ("TotalSeg" if d < -0.01 else "паритет")
        print(f"{LABEL_NAMES.get(lb, lb):<16}{a:>12.4f}{b:>12.4f}{d:>+10.4f}  {verdict}")
    print("-" * 74)
    a, b = nn.get("mean_dice"), ts.get("mean_dice")
    if a is not None and b is not None:
        print(f"{'СРЕДНЕЕ':<16}{a:>12.4f}{b:>12.4f}{a - b:>+10.4f}")
    print("=" * 74)
    print(f"кейсов: nnU-Net {nn.get('n_cases')} / TotalSeg {ts.get('n_cases')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True,
                    help="папка с imagesTr/ + labelsTr/ (.mha, 48 классов)")
    ap.add_argument("--preds", type=Path, required=True,
                    help="fold_0/validation нашей модели (.mha, 7 классов)")
    ap.add_argument("--out", type=Path, default=Path("data/benchmark"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cases", default="", help="через запятую; по умолчанию — все из --preds")
    ap.add_argument("--no-postprocess", action="store_true",
                    help="не переназначать стороны каналов геометрически")
    ap.add_argument("--skip-totalseg", action="store_true",
                    help="только пересчитать метрики по уже готовым предсказаниям")
    args = ap.parse_args()

    if args.cases:
        cases = [c.strip().replace(".mha", "") for c in args.cases.split(",") if c.strip()]
    else:
        cases = sorted(p.stem for p in args.preds.glob("*.mha"))
    if not cases:
        print(f"Не найдено кейсов в {args.preds}")
        return 1
    print(f"Held-out кейсы ({len(cases)}): {', '.join(cases)}")

    gt_dir = args.out / "gt7"
    nn_dir = args.out / "pred_nnunet"
    ts_dir = args.out / "pred_totalseg"
    for d in (gt_dir, nn_dir, ts_dir):
        d.mkdir(parents=True, exist_ok=True)

    build_gt(args.raw, cases, gt_dir)
    map_nnunet(args.preds, cases, nn_dir, postprocess=not args.no_postprocess)
    if not args.skip_totalseg:
        run_totalseg(args.raw, cases, ts_dir, args.device)

    labels = sorted(v for v in GROUPED_LABELS.values() if v > 0)
    nn = score(nn_dir, gt_dir, labels)
    ts = score(ts_dir, gt_dir, labels)

    out_json = args.out / "benchmark.json"
    out_json.write_text(json.dumps(
        {"labels": LABEL_NAMES, "nnunet": nn, "totalsegmentator": ts}, indent=2, ensure_ascii=False))
    print_table(nn, ts, labels)
    print(f"\nПодробности → {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
