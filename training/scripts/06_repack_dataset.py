#!/usr/bin/env python
"""
06_repack_dataset.py — ужимает ToothFairy2 со 109 ГБ до ~11 ГБ без потери качества.

Зачем: снимки в исходном зеркале лежат в float64 БЕЗ сжатия — 550 МБ на кейс.
Данные при этом 13-битные: 4476 уникальных значений от -1000 до 3476 на идеально
равномерной сетке с шагом 1.0002234637. То есть целочисленный сигнал, раздутый
вчетверо, поверх которого никто не включил компрессию.

Замер на одном кейсе (снимок 549.5 МБ / метка 68.7 МБ):

    int без сжатия        137.4 МБ  (4.0x)  |  68.7 МБ    (1.0x)
    int + zlib (.mha)      62.6 МБ  (8.8x)  |   0.6 МБ  (118.5x)   ← выбрано
    int + zstd (blosc2)    50.1 МБ (11.0x)  |   0.3 МБ  (227.5x)
    parquet + zstd         63.6 МБ  (8.6x)  |   0.6 МБ  (121.4x)

Взят .mha + zlib, а не самый плотный вариант: SimpleITKIO читает его как есть,
`file_ending: ".mha"` в plans.json не меняется, nnU-Net ничего не замечает.
blosc2 плотнее на 20%, но .b2nd на вход nnU-Net не принимает — пришлось бы
конвертировать при каждом запуске обучения. Паркет теряет spacing/origin/
direction, без которых снимок перестаёт быть медицинским изображением, и при
этом проигрывает по размеру.

Точность: перевод в int16 округляет до целых HU, ошибка не больше 0.4999 HU —
на порядки ниже шума CBCT, и данные всё равно z-нормализуются перед обучением.
Каждый файл после записи читается обратно и сверяется; кейс, не прошедший
проверку, не попадает в выход.

Скрипт идемпотентен: готовые файлы пропускаются, так что после обрыва достаточно
запустить заново.

Usage:
  python training/scripts/06_repack_dataset.py --out data/raw/datasets/toothfairy2_compact
  python training/scripts/06_repack_dataset.py --out ... --upload povchingiz/stomato2-compact
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import pathlib
from pathlib import Path

import numpy as np

SRC_REPO = "povchingiz/stomato2"
MAX_ABS_HU_ERROR = 0.5      # квантование до целых HU
LOG_EVERY = 10


def hf_case_list(repo: str, token: str | None) -> tuple[str, list[str]]:
    """(префикс внутри репо, отсортированный список case id)."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo, repo_type="dataset", token=token)
    labels = [f for f in files if f.endswith(".mha") and "labelsTr/" in f]
    if not labels:
        raise RuntimeError(f"В {repo} не нашлось labelsTr/*.mha")
    prefix = labels[0].split("labelsTr/")[0]
    cases = sorted({os.path.basename(f)[:-4] for f in labels})
    return prefix, cases


def local_or_download(case: str, kind: str, local_root: Path, cache: Path,
                      repo: str, prefix: str, token: str | None) -> Path:
    """Путь к исходнику: сначала локальная копия, иначе тянем с HF в кэш."""
    name = f"{case}_0000.mha" if kind == "imagesTr" else f"{case}.mha"
    local = local_root / kind / name
    if local.exists():
        return local

    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, f"{prefix}{kind}/{name}", repo_type="dataset",
                                local_dir=str(cache), token=token))


def to_int16(arr: np.ndarray) -> np.ndarray:
    """float64-сетка → int16 в HU. Уже целые типы отдаём как есть."""
    if np.issubdtype(arr.dtype, np.integer):
        return arr
    return np.round(arr).astype(np.int16)


def repack_one(src: Path, dst: Path, is_label: bool) -> tuple[int, int, float]:
    """Возвращает (байт до, байт после, максимальная ошибка в HU)."""
    import SimpleITK as sitk

    img = sitk.ReadImage(str(src))
    arr = sitk.GetArrayFromImage(img)
    packed = arr.astype(np.uint8) if is_label else to_int16(arr)
    err = 0.0 if is_label else float(np.abs(packed.astype(np.float64) - arr).max())

    out = sitk.GetImageFromArray(packed)
    out.CopyInformation(img)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mha")
    sitk.WriteImage(out, str(tmp), useCompression=True)

    # Проверяем то, что реально легло на диск, а не то, что собирались записать.
    back = sitk.ReadImage(str(tmp))
    rb = sitk.GetArrayFromImage(back)
    if rb.shape != packed.shape or not np.array_equal(rb, packed):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{src.name}: файл после записи не совпал с исходным массивом")
    if not np.allclose(back.GetSpacing(), img.GetSpacing()) or \
       not np.allclose(back.GetOrigin(), img.GetOrigin()) or \
       not np.allclose(back.GetDirection(), img.GetDirection()):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{src.name}: потеряна геометрия (spacing/origin/direction)")
    if err > MAX_ABS_HU_ERROR:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{src.name}: ошибка квантования {err:.4f} HU > {MAX_ABS_HU_ERROR}")

    tmp.replace(dst)
    return src.stat().st_size, dst.stat().st_size, err


def _ensure_lfs_rule(api, repo: str, pattern: str = "*.mha") -> None:
    """Гарантирует, что .gitattributes в репозитории отправляет pattern в LFS."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    rule = f"{pattern} filter=lfs diff=lfs merge=lfs -text"
    try:
        current = pathlib.Path(hf_hub_download(
            repo, ".gitattributes", repo_type="dataset")).read_text()
    except (EntryNotFoundError, OSError):
        current = ""
    if rule in current:
        return
    api.upload_file(
        path_or_fileobj=(current.rstrip("\n") + "\n" + rule + "\n").lstrip("\n").encode(),
        path_in_repo=".gitattributes", repo_id=repo, repo_type="dataset",
        commit_message=f"Track {pattern} with LFS — plain binary pushes are rejected",
    )
    print(f"[repack] в .gitattributes добавлено правило LFS для {pattern}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="куда писать сжатый датасет")
    ap.add_argument("--local", type=Path,
                    default=Path("data/raw/datasets/toothfairy2_raw_all"),
                    help="локальные исходники (используются вместо скачивания)")
    ap.add_argument("--cache", type=Path, default=Path("data/raw/datasets/_repack_cache"),
                    help="куда качать недостающие исходники")
    ap.add_argument("--repo", default=SRC_REPO)
    ap.add_argument("--limit", type=int, default=0, help="обработать только N кейсов (для проб)")
    ap.add_argument("--upload", default="", help="залить результат в этот приватный репозиторий")
    ap.add_argument("--keep-cache", action="store_true",
                    help="не удалять скачанный исходник после успешной упаковки")
    args = ap.parse_args()

    token = os.getenv("HF_TOKEN") or None
    prefix, cases = hf_case_list(args.repo, token)
    if args.limit:
        cases = cases[:args.limit]
    print(f"[repack] {len(cases)} кейсов из {args.repo} (префикс '{prefix}')", flush=True)

    stats = {"before": 0, "after": 0, "done": 0, "skipped": 0, "failed": [], "max_err": 0.0}
    t0 = time.time()

    for i, case in enumerate(cases, 1):
        for kind, is_label in (("imagesTr", False), ("labelsTr", True)):
            name = f"{case}_0000.mha" if kind == "imagesTr" else f"{case}.mha"
            dst = args.out / kind / name
            if dst.exists():
                stats["skipped"] += 1
                continue
            try:
                src = local_or_download(case, kind, args.local, args.cache,
                                        args.repo, prefix, token)
                before, after, err = repack_one(src, dst, is_label)
                stats["before"] += before
                stats["after"] += after
                stats["max_err"] = max(stats["max_err"], err)
                stats["done"] += 1
                # Скачанный исходник больше не нужен — оригинал остаётся в репо.
                # Локальные файлы не трогаем: на них ссылаются жёсткие ссылки
                # из nnU-Net датасетов.
                if not args.keep_cache and args.cache in src.parents:
                    src.unlink(missing_ok=True)
            except Exception as e:                       # noqa: BLE001
                print(f"  !! {name}: {e}", flush=True)
                stats["failed"].append(name)

        if i % LOG_EVERY == 0 or i == len(cases):
            gb = lambda b: b / 1e9                        # noqa: E731
            ratio = stats["before"] / stats["after"] if stats["after"] else 0
            elapsed = time.time() - t0
            eta = elapsed / i * (len(cases) - i)
            print(f"  {i}/{len(cases)}  упаковано {stats['done']} (пропущено "
                  f"{stats['skipped']})  {gb(stats['before']):.1f} → "
                  f"{gb(stats['after']):.1f} ГБ  ({ratio:.1f}x)  ETA {eta/3600:.1f}ч",
                  flush=True)

    dataset_json = args.out / "dataset.json"
    if not dataset_json.exists():
        dataset_json.parent.mkdir(parents=True, exist_ok=True)
        dataset_json.write_text(json.dumps({
            "name": "ToothFairy2 (repacked)",
            "source_repo": args.repo,
            "file_ending": ".mha",
            "overwrite_image_reader_writer": "SimpleITKIO",
            "note": "Images int16 HU, labels uint8, both zlib-compressed MetaImage. "
                    "Geometry preserved; quantisation error <= 0.5 HU. "
                    "Label semantics unchanged from the source (48 classes).",
        }, indent=2))

    ratio = stats["before"] / stats["after"] if stats["after"] else 0
    print(f"\n[repack] готово: {stats['done']} файлов упаковано, {stats['skipped']} пропущено")
    print(f"[repack] {stats['before']/1e9:.1f} ГБ → {stats['after']/1e9:.1f} ГБ ({ratio:.1f}x)")
    print(f"[repack] максимальная ошибка квантования: {stats['max_err']:.4f} HU")
    if stats["failed"]:
        print(f"[repack] НЕ УПАКОВАНО ({len(stats['failed'])}): {stats['failed'][:10]}")
        print("[repack] залив отменён — сначала разберись с ошибками")
        return 1

    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.upload, repo_type="dataset", private=True, exist_ok=True)
        # Без явного правила HF отклоняет .mha как «просто бинарник»:
        # "Your push was rejected because it contains binary files."
        # Датасет при этом заливается частично — 741 файл из 961, и молча.
        _ensure_lfs_rule(api, args.upload)
        print(f"[repack] заливаю в {args.upload} (приватный) ...", flush=True)
        url = api.upload_folder(folder_path=str(args.out), repo_id=args.upload,
                                repo_type="dataset",
                                commit_message=f"Repacked ToothFairy2: "
                                               f"{stats['before']/1e9:.0f}GB -> "
                                               f"{stats['after']/1e9:.0f}GB, int16 + zlib")
        print(f"[repack] залито: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
