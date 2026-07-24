#!/usr/bin/env bash
# 00_setup_env.sh — окружение для CBCT-сегментации на L40.
# Запускать один раз на твоей машине с GPU.
set -euo pipefail

echo "==> Python venv"
python -m venv .venv
source .venv/bin/activate

echo "==> PyTorch (CUDA). Под свою версию CUDA сверься на pytorch.org"
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "==> nnU-Net v2 + MONAI + утилиты медимиджинга"
pip install nnunetv2 monai nibabel SimpleITK scipy scikit-image

echo "==> Переменные окружения nnU-Net (добавь в ~/.bashrc, чтобы не терять)"
cat <<'EOF'

# --- добавь это в ~/.bashrc ---
export nnUNet_raw="$PWD/data/nnunet/raw"
export nnUNet_preprocessed="$PWD/data/nnunet/preprocessed"
export nnUNet_results="$PWD/data/nnunet/results"
# ------------------------------
EOF

echo "==> Проверка GPU"
python - <<'PY'
import torch
print("CUDA доступна:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM, ГБ:", round(torch.cuda.get_device_properties(0).total_memory/1e9,1))
    print("bf16:", torch.cuda.is_bf16_supported())
PY

echo "==> Готово. Дальше: make train (pipeline Phase 6) из корня репозитория."
