# Handover — where everything lives and how to restore it

This machine was wiped after the work below. Nothing here depends on it. Every
artifact is either in this repository or in a private HuggingFace repo.

## Where things are

| what | where | notes |
|---|---|---|
| Code | this repository, branch `feature/phase6-model-and-pipeline-fixes` | 12 commits ahead of `main` |
| Model weights (3 checkpoints) | `povchingiz/toothfairy2-7class-model` (HF, private) | 1.5 GB, includes `postprocess.py` and all training logs |
| Training data | `povchingiz/stomato2-compact` (HF, private) | ToothFairy2 repacked, 109.5 GB → 15.5 GB |
| Patient scan + meshes | `povchingiz/dycom-patient-data` (HF, private) | CBCT + the 58 STLs from the original Phase 1 run |
| Secrets | **nowhere** | `.env` was never committed. Recreate from `.env.example` |

`.env` needs: `DEMO_PASSWORD`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`,
`SEGMENTATION_DEVICE`, and the three `nnUNet_*` paths. The Telegram token is
reissued through BotFather if lost.

## Restoring on a new machine

```bash
git clone -b feature/phase6-model-and-pipeline-fixes https://github.com/povchingiz/dycom
cd dycom && make setup-gpu
cp .env.example .env          # then fill it in
huggingface-cli login

huggingface-cli download povchingiz/toothfairy2-7class-model \
    --local-dir $nnUNet_results/Dataset115_ToothFairy2_grouped
huggingface-cli download povchingiz/dycom-patient-data --repo-type dataset --local-dir data/

python tests/selftest_phase5.py     # smoke test: should print PASS
```

To retrain, add the dataset:

```bash
huggingface-cli download povchingiz/stomato2-compact --repo-type dataset \
    --local-dir data/raw/datasets/toothfairy2_compact
./run_training_queue.sh             # needs 450 GB free; it checks
```

## State of play

**Works and is measured.** Phases 0–5 run end to end. The segmentation model
beats the incumbent on every class — 0.9240 mean Dice against TotalSegmentator's
0.8506 on 37 held-out cases, method in
[ROADMAP](ROADMAP.md#phase-6--ml-training--model-beats-the-incumbent-integration-pending).

**Works but is not validated.** The face prediction. Soft-tissue response uses
population-mean soft:hard ratios from the orthognathic literature, not
patient-specific measurements. Phase 5 can measure the real error the moment a
paired post-op scan exists; until then no number in this repository proves the
predicted face is right. This is the single most valuable thing missing, and it
is data, not code.

**Known incomplete.**

1. **The model is not wired into Phase 1.** It is better, and unused. Phase 1
   still calls TotalSegmentator. The obstacle is real: the 7-class model does not
   produce the sinuses and individual teeth that `pipeline/anatomy.py` derives
   the anatomical frame from, and deriving the frame from the grouped classes
   alone is 30° off on the superior axis (measured, rejected). The answer is a
   hybrid — nnU-Net for the mandible and canals, TotalSegmentator for the frame.
2. **Renders are not shown in the browser.** The server produces before/after
   meshes and ships them in the ZIP; the UI only offers a download.
3. **Out-of-domain robustness.** The model degrades on whole-head scans.
   `clean_prediction()` treats the symptom. Untested hypothesis worth half a day:
   crop the input to the dental FOV before inference.
4. **Canal side convention.** ToothFairy2 and TotalSegmentator disagree on which
   canal is left. One is anatomically wrong. Harmless until a side is displayed
   to a clinician; safety-relevant after that.
5. **No CI.** One self-test exists, no runner. Three of the bugs fixed in this
   branch were silent — wrong simulation axes, a double-applied label LUT, and
   decimation by random face dropping. None raised an error.

## Bugs that were silent

Recorded because each looked like working software.

| bug | effect |
|---|---|
| `np.random.choice(faces)` as "decimation" | punched holes in every mesh; the skin surface came out as 96519 components. This is why FEBio was abandoned |
| Phase 3 used mesh axis 1 and 2 directly | simulated a jaw *setback* while anchoring the left side of the head |
| `pydicom==2.4.4` pin | TotalSegmentator could not import at all |
| non-empty download cache trusted as complete | training started on 4 cases and died on `n_splits=5 > n_samples=4` |
| 48→7 LUT applied twice | teeth silently vanished from ground truth; both models scored 0.0000 |
| `nnUNet_n_proc_DA=0` pinned for an old Docker box | 3.6× slower training, GPU at 20% |

## Reproducing the headline numbers

```bash
python training/scripts/05_benchmark_vs_totalseg.py \
    --raw data/raw/Dataset115_ToothFairy2_grouped \
    --preds $nnUNet_results/Dataset115_*/*/fold_0/validation \
    --out data/benchmark
```
