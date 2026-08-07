#!/usr/bin/env bash
# Sequential training queue — runs multiple Phase-6 configs back-to-back so the
# GPU never sits idle. Each run uses a distinct TF2_DATASET_ID so runs never
# clobber each other's data/checkpoints. If one run fails, the queue logs it and
# CONTINUES to the next (a broken run shouldn't block the others).
#
# Usage:   ./run_training_queue.sh
# Detach:  tmux new -s train './run_training_queue.sh'   (survives disconnect)
# Logs:    logs/queue_<name>.log  per run
set -u

cd "$(dirname "$0")"
mkdir -p logs
set -a; . ./.env; set +a          # load nnUNet paths + HF settings

# each entry: NAME | env overrides for that run
# (all 480 cases = TF2_MAX_CASES=0; full 1000 epochs = TF2_TRAINER=nnUNetTrainer)
run() {
  local name="$1"; shift
  echo "=================================================="
  echo "[queue] START $name  ($(date))"
  echo "[queue] env: $*"
  echo "=================================================="
  # run each override as KEY=VAL prefix to make train
  env "$@" .venv312/bin/python pipeline/main.py --phase 6 \
      2>&1 | tee "logs/queue_${name}.log"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then
    echo "[queue] ✅ DONE $name ($(date))"
  else
    echo "[queue] ❌ FAILED $name rc=$rc ($(date)) — continuing to next"
  fi
}

# ── the queue ──────────────────────────────────────────────────────────
# 1) 7 classes, all 480 cases, full 1000 epochs  (improve the demo model)
run "7cls_480_1000ep" \
    TF2_CLASSES=7 TF2_MAX_CASES=0 TF2_DATASET_ID=114 \
    TF2_CONFIG=3d_fullres TF2_TRAINER=nnUNetTrainer

# 2) 48 classes, all 480 cases, full 1000 epochs  (per-tooth, max)
run "48cls_480_1000ep" \
    TF2_CLASSES=48 TF2_MAX_CASES=0 TF2_DATASET_ID=112 \
    TF2_CONFIG=3d_fullres TF2_TRAINER=nnUNetTrainer

echo "[queue] all runs finished ($(date))"
