# Training — CBCT bone/teeth segmentation (nnU-Net v2)

Trains nnU-Net on **ToothFairy2** (480 CBCT scans, teeth + jaw structures) to
produce the segmentation model that replaces TotalSegmentator in Phase 1 of the
FaceSim pipeline.

Orchestration lives in **`pipeline/phases/p6_train.py`** — the canonical entry
point (checkpointed, resumable, OOM-hardened). This folder holds the standalone
helper scripts `p6_train.py` calls, plus configs and the guide notebook.

## Run it (canonical path)

From the repo root, on the GPU server:

```bash
make setup-gpu     # venv + PyTorch CUDA + nnU-Net + writes nnUNet_* paths into .env
make train         # pipeline Phase 6: download → prepare → smoke → train → evaluate
```

`make train` == `python pipeline/main.py --phase 6`. Every sub-step is idempotent
and resumes from the last completed one, so re-running after an interruption is safe.

Artifact = checkpoint + `metrics.json` (Dice/HD95 on held-out). Without a held-out
metric it is not an artifact. Target: mean Dice > 0.85.

## What p6_train.py does

| Sub-step | Action |
|---|---|
| download | Fetch ToothFairy2 (HuggingFace/Zenodo) — skipped if already in `nnUNet_raw` |
| prepare  | Convert to nnU-Net layout + rebuild `dataset.json` (labels `0..max` auto-discovered) |
| smoke    | `scripts/02_smoke_test.py` — env/dataset/GPU check before burning GPU time |
| train    | `nnUNetv2_train` with OOM guards; auto-falls back ResEncL → default plan on OOM |
| evaluate | `scripts/04_evaluate.py` — Dice + HD95 → `metrics.json` |

## OOM safety (built into p6_train.py `_train`)

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — prevents VRAM fragmentation.
- `nnUNet_n_proc_DA=2` — caps data-aug workers (guards the small `/dev/shm` box).
- Two-tier plan: tries `nnUNetResEncUNetLPlans` (tuned for L40 48GB); on OOM it
  clears the CUDA cache and retries the smaller default `nnUNetPlans`.
- `--c` resume + live-streamed logs, so a killed 12h run isn't lost.

## Scientific loop (write these down before the run)
1. **Hypothesis on paper**: L40S (48GB) vs the RTX 4090 (24GB) SOTA baselines.
   Prediction: larger patch size → more spatial context → higher Dice on large
   structures and better tooth numbering. Record expected gain *before* running.
2. **Metric prediction**: baseline nnU-Net ResEnc ≈ 0.90–0.92 Dice (ToothFairy2
   literature). Ablation: expected +X.
3. **Experiment**: baseline + 1 ablation on L40.
4. **Compare**: did the gain match the prediction? If not, why (context vs data/classes)?
5. **Record**: table «config → prediction → Dice → matched?».

## Metric (what makes the artifact testable)
- **Dice** per-class and mean on held-out (9:1 split).
- **HD95** (Hausdorff 95%) — boundary-sensitive, matters for downstream mesh quality.

## Hardware notes
SOTA teams: RTX 4090 (24GB), patch 128×256×256, batch 1, nnU-Net ResEnc.
L40S (48GB) ≈ 2× memory headroom → first meaningful ablation = larger patch/batch.

## Contents
- `scripts/00_setup_env.sh` — reference env setup (Makefile `setup-gpu` is the real one)
- `scripts/02_smoke_test.py` — 1-shot env/dataset/GPU sanity check
- `scripts/04_evaluate.py` — Dice/HD95 on held-out → `metrics.json` (format-agnostic: .nii.gz/.mha)
- `configs/large_patch.md` — notes for the larger-patch ablation
- `notebooks/cbct_seg_guide.ipynb` — hypothesis → run → compare guide

> Data: ToothFairy2 is CC-BY-NC-SA and requires registration. The download
> sub-step tries known mirrors; if they fail it prints manual instructions.
