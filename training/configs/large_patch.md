# Абляция large_patch — использование запаса VRAM L40

## Гипотеза (записать ДО запуска)
SOTA на ToothFairy2/3 обучают на RTX 4090 (24 ГБ): patch 128×256×256, batch 1.
Победитель ToothFairy2 поднял patch до 160×320×320 + 7 стадий → Dice 0.9253.

Твоя L40 = 48 ГБ, вдвое больше. Гипотеза: дальнейшее увеличение patch size
даёт сети больше пространственного контекста → точнее co-localization зубов
(правильная FDI-нумерация) и крупные кости. Ожидаемый прирост mean Dice: **запиши число до прогона**.

Контр-гипотеза (тоже записать): прирост упрётся не в контекст, а в число/баланс
классов или в качество разметки → patch не поможет, Dice не вырастет. Это
тоже валидный результат, и он важнее «успеха», потому что говорит, где реальное узкое место.

## Как сгенерировать план
nnU-Net v2 позволяет править план вручную:

```bash
# 1. Базовый план уже создан plan_and_preprocess (nnUNetResEncUNetLPlans)
# 2. Скопировать его в новый план и поднять patch size
python - <<'PY'
import json, os
from pathlib import Path
pp = Path(os.environ["nnUNet_preprocessed"])
ds = sorted(pp.glob("Dataset301_*"))[0]
plans = json.loads((ds / "nnUNetResEncUNetLPlans.json").read_text())
# поднимаем patch в 3d_fullres конфиге (сверься, что влезает в 48 ГБ)
cfg = plans["configurations"]["3d_fullres"]
cfg["patch_size"] = [192, 384, 384]   # подбери по факту памяти
plans["plans_name"] = "nnUNetResEncUNetLPlans_largepatch"
(ds / "nnUNetResEncUNetLPlans_largepatch.json").write_text(json.dumps(plans, indent=2))
print("план large_patch создан, patch:", cfg["patch_size"])
PY
```

## Замечание про память
Если CUDA OOM на 192×384×384 — снижай по одной оси, пока влезет. Цель абляции
не «максимальный patch», а **проверить гипотезу о контексте** при честном сравнении
(тот же датасет, тот же сплит, та же длительность обучения).

## Что сравнивать (таблица в ноутбуке)
| Конфиг | patch_size | Предсказание Dice | Факт Dice | HD95 | Сошлось? |
|---|---|---|---|---|---|
| baseline | 160×320×320 | (литература ~0.92) | | | |
| large_patch | 192×384×384 | (твой прогноз) | | | |
