#!/usr/bin/env bash
# 03_train.sh — обучение nnU-Net. Два конфига: baseline и large_patch (абляция под L40).
# Использование: bash scripts/03_train.sh <DATASET_ID> <baseline|large_patch>
set -euo pipefail

DATASET="${1:?укажи DATASET_ID, напр. 301}"
CONFIG="${2:-baseline}"
FOLD="${3:-0}"

echo "==> Планирование и препроцессинг (один раз на датасет)"
nnUNetv2_plan_and_preprocess -d "$DATASET" --verify_dataset_integrity

case "$CONFIG" in
  baseline)
    # Стандартный 3D full-res ResEnc — как у SOTA на RTX 4090.
    echo "==> BASELINE: nnUNetTrainer, 3d_fullres, ResEnc plan"
    nnUNetv2_train "$DATASET" 3d_fullres "$FOLD" \
      -p nnUNetResEncUNetLPlans \
      --npz
    ;;
  large_patch)
    # АБЛЯЦИЯ: твоя L40 (48 ГБ) тянет больше patch, чем 4090 (24 ГБ).
    # Гипотеза: больше контекста -> выше Dice на крупных структурах и нумерации зубов.
    # Кастомный план с увеличенным patch генерируется отдельно (см. configs/large_patch.md).
    echo "==> LARGE_PATCH: кастомный план с увеличенным patch size"
    echo "    (сначала сгенерируй план по инструкции в configs/large_patch.md)"
    nnUNetv2_train "$DATASET" 3d_fullres "$FOLD" \
      -p nnUNetResEncUNetLPlans_largepatch \
      --npz
    ;;
  *)
    echo "Неизвестный конфиг: $CONFIG (ожидается baseline|large_patch)"; exit 1 ;;
esac

echo "==> Обучение запущено. Чекпоинты и логи в nnUNet_results."
echo "==> Метрики val пишутся автоматически; для held-out оценки: scripts/04_evaluate.py"
