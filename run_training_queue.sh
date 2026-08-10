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

# All 480 ToothFairy2 cases are ~200GB raw, and nnU-Net's fullres preprocessing
# writes a comparable amount again. Dying on ENOSPC three days into a run is the
# expensive failure mode, so refuse to start without headroom.
MIN_FREE_GB=${MIN_FREE_GB:-450}
FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
if [ "$FREE_GB" -lt "$MIN_FREE_GB" ]; then
  echo "[queue] ABORT: ${FREE_GB}GB free, need ${MIN_FREE_GB}GB."
  echo "[queue] Free space or lower MIN_FREE_GB if you know the run fits."
  exit 1
fi
echo "[queue] disk check ok: ${FREE_GB}GB free"

# each entry: NAME | env overrides for that run
# (all 480 cases = TF2_MAX_CASES=0; full 1000 epochs = TF2_TRAINER=nnUNetTrainer)
run() {
  local name="$1"; shift
  echo "=================================================="
  echo "[queue] START $name  ($(date))"
  echo "[queue] env: $*"
  echo "=================================================="
  # Reset phase-6 state first. The pipeline records phase6 as "complete" in the
  # shared state file (keyed by phase name, NOT by dataset id), so without a
  # reset every run after the first sees "already done" and exits instantly.
  env "$@" .venv312/bin/python pipeline/main.py --reset-phase 6 >/dev/null 2>&1
  # run each override as KEY=VAL prefix to make train
  # -u matters: stdout is a pipe into tee, so without it Python block-buffers and
  # the log stays empty for hours while the run is actually progressing.
  env "$@" .venv312/bin/python -u pipeline/main.py --phase 6 \
      2>&1 | tee "logs/queue_${name}.log"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then
    echo "[queue] ✅ DONE $name ($(date))"
  else
    echo "[queue] ❌ FAILED $name rc=$rc ($(date)) — continuing to next"
  fi
}

# ── the queue ──────────────────────────────────────────────────────────
# Target to beat, measured on the same 12 held-out cases by
# training/scripts/05_benchmark_vs_totalseg.py (data/benchmark/benchmark.json):
#
#            TotalSegmentator   current 60-case/lowres model
#   mean Dice        0.8401                        0.8307
#   canals    0.771 / 0.777                 0.625 / 0.671
#   canal HD95  5.2 / 4.5 mm                  31.8 / 31.0 mm
#
# The incumbent still wins overall, and it wins on the canals by a wide margin —
# the alveolar nerve is exactly the structure 3d_lowres cannot resolve. That is
# what this queue is for. Success = beat 0.8401 mean AND beat 0.77 on canals.

# 1) 7 classes, all 480 cases, 250 epochs — first honest shot at the target.
#    ~2 days on a 3090 instead of ~a week, so the benchmark can be re-run early.
run "7cls_480_250ep" \
    TF2_CLASSES=7 TF2_MAX_CASES=0 TF2_DATASET_ID=114 \
    TF2_CONFIG=3d_fullres TF2_TRAINER=nnUNetTrainer_250epochs

# 2) Same data, full 1000 epochs — squeezes out the last few points.
#    Runs second on purpose: if (1) already clears the bar, this is optional.
run "7cls_480_1000ep" \
    TF2_CLASSES=7 TF2_MAX_CASES=0 TF2_DATASET_ID=115 \
    TF2_CONFIG=3d_fullres TF2_TRAINER=nnUNetTrainer

# NB: the 48-class run was dropped. Per-tooth labels do not change the predicted
# face — the simulation is driven by the mandible surface, not by which tooth is
# which — and it costs another week of GPU. Re-add it when the product needs
# per-tooth output (implant planning), not before.

echo "[queue] all runs finished ($(date))"
