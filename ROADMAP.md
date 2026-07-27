# ROADMAP — FaceSim Pipeline

## Research Question
Open-source pipeline for predicting aesthetic facial changes (before→after) following orthognathic surgery from a single pre-op CBCT. No commercial software.

---

## Stack

| Layer | Tool | Status |
|---|---|---|
| Segmentation | TotalSegmentator (ToothFairy3 weights) | ✅ |
| Mesh processing | Open3D, trimesh, meshio | ✅ |
| Soft tissue sim | Laplacian displacement (scipy sparse) | ✅ |
| Render before/after | Phase 4 pipeline | ✅ |
| ML segmentation training | nnUNet v2 3d_fullres fold 0 | 🔄 |
| Pipeline orchestration | Python state machine phases 1–6 | ✅ |
| Notifications | Telegram bot | ✅ |
| Patient data transfer | HuggingFace private repo | ✅ |

---

## Phase 0 — Setup ✅ COMPLETE

- [x] Patient archive unpacked (Baytemirova_S.zip)
- [x] CBCT verified: Planmeca ProMax, 409×409×409 @ 0.4mm, full head
- [x] Soft tissue confirmed visible in CBCT (skin HU -500 to -200)
- [x] DICOM anonymization done
- [x] Repo: git + DVC + Docker skeleton

---

## Phase 1 — Segmentation ✅ COMPLETE

- [x] TotalSegmentator → 56/77 non-empty masks (jaw, teeth, skin, sinuses)
- [x] Soft tissue threshold segmentation (-700 to +200 HU)
- [x] 58 STL files exported
- [x] Patient data uploaded to HuggingFace → downloaded to L40 server

---

## Phase 2 — Meshing ✅ COMPLETE

- [x] Open3D mesh processing and cleanup
- [x] Before/after mesh directory structure created for Phase 4

---

## Phase 3 — Simulation ✅ COMPLETE

**Approach:** Laplacian displacement propagation (FEBio abandoned — skin STL not watertight, gmsh crashed)

- [x] Jaw-adjacent skin nodes detected via cKDTree (JAW_RADIUS_MM = 15mm)
- [x] Skull base nodes fixed (top 5% by Z)
- [x] Sparse Laplacian matrix built from face adjacency (fully vectorized, numpy)
- [x] 300 sparse matrix multiply iterations (~60 sec total)
- [x] Outputs: `data/sim/deformed.vtk`, `data/mesh/before/`, `data/mesh/after/`

**Tunable parameters in [p3_sim.py](pipeline/phases/p3_sim.py):**

| Param | Default | Effect |
|---|---|---|
| `SCENARIO_MM` | 5.0 | Jaw advancement in mm |
| `JAW_RADIUS_MM` | 15.0 | Skin influence radius around jaw |
| `N_ITER` | 300 | Smoothing iterations (more = smoother) |

---

## Phase 4 — Render ✅ COMPLETE

- [x] Before/after side-by-side renders generated from Phase 3 meshes

---

## Phase 5 — Validation ✅ COMPLETE (placeholder)

- [x] Validation step passed
- [ ] Real validation requires paired pre/post-op scan (not available yet)

---

## Phase 6 — ML Training 🔄 IN PROGRESS

**Goal:** Train nnUNet on ToothFairy2 (480 CBCT scans, 42 dental structures) to replace TotalSegmentator for Phase 1.

**Server paths:**
- Dataset: `/home/kaiyr/chin/dycom/data/toothfairy2/Dataset112_ToothFairy2/`
- Preprocessed: `/home/kaiyr/chin/dycom/data/nnunet_preprocessed/`
- Results: `/home/kaiyr/chin/dycom/data/nnunet_results/`

**Progress:**
- [x] Dataset on server: 480 images + 480 labels (.mha format)
- [x] Smoke test: L40S 47.7GB VRAM, bf16 ✅
- [x] nnUNet preprocessing: 480/480 done (~73 min, `-np 2`)
- [x] dataset.json patched: numTraining=480, correct file_ending=.mha
- [ ] **BLOCKER:** CUDA index-out-of-bounds during training loss computation
  - Root cause: label values in .mha files exceed declared num_classes in dataset.json
  - Fix: discover true max label from all 480 files, declare 0..max

**To fix and start training on server:**
```bash
# 1. Find actual max label
python3 -c "
import SimpleITK as sitk, glob, os
files = glob.glob(os.environ['nnUNet_raw'] + '/Dataset112_ToothFairy2/labelsTr/*')
mx = max(int(sitk.GetArrayFromImage(sitk.ReadImage(f)).max()) for f in files)
print('max label:', mx)
"

# 2. Patch dataset.json
python3 -c "
import json, os
path = os.environ['nnUNet_raw'] + '/Dataset112_ToothFairy2/dataset.json'
dj = json.loads(open(path).read())
mx = 48  # replace with actual value from step 1
dj['labels'] = {'background': 0, **{f'label_{v:03d}': v for v in range(1, mx+1)}}
open(path, 'w').write(json.dumps(dj, indent=2))
print('done:', len(dj['labels']), 'labels')
"

# 3. Launch training in tmux (no preprocessing needed — already done)
tmux new -s training
set -a; source .env; set +a
TORCHDYNAMO_DISABLE=1 .venv312/bin/nnUNetv2_train 112 3d_fullres 0 --npz
# Ctrl+B D to detach
```

**After training (~12h on L40S):**
```bash
cat $nnUNet_results/Dataset112_ToothFairy2/*/fold_0/validation/summary.json
```
Target: mean Dice > 0.85. To improve: train all 5 folds or use XL plan.

---

## Remaining Work

| Task | Priority | Status |
|---|---|---|
| Fix Phase 6 label OOB + launch training | 🔴 High | Patch ready, needs server run |
| Wait for training overnight (~12h) | 🔴 High | Blocked on above |
| Show before/after renders in web UI | 🟡 Medium | FastAPI server exists at :8000 |
| Pre-loaded demo mode (existing patient) | 🟡 Medium | Fastest path to shareable demo |
| Validate simulation vs real post-op | 🟡 Medium | Need paired scan |
| Scale to multiple patients | 🟢 Low | Not needed yet |

---

## Server Info

| | |
|---|---|
| GPU | NVIDIA L40S, 47.7 GB VRAM |
| Path | `/home/kaiyr/chin/dycom/` |
| Venv | `.venv312` |
| Docker shm | 64MB (fixed) — use `TORCHDYNAMO_DISABLE=1` |
| Run pipeline | `make pipeline` |
| Run training only | `make train` |
| Check status | `make status` |

---

## Known Issues / Workarounds

| Issue | Workaround |
|---|---|
| Skin STL not watertight → gmsh/FEBio fails | Replaced with Laplacian displacement |
| Docker /dev/shm too small (~64MB) → training dies with "unable to allocate shared memory(shm): No space left" | `nnUNet_n_proc_DA=0` (augmentation in main process, no shm) — forced in `p6_train.py`. Note: `=2` was **not** enough. Alt: restart container with `--shm-size=8g` and re-enable workers. |
| ToothFairy2 dataset.json has wrong numTraining (0) | `_patch_dataset_json` counts actual files |
| ToothFairy2 labels missing "background" key | Auto-discovered from actual .mha files |
| fast-simplification target_reduction validation | Replaced with open3d simplification |
| Smoke test hardcoded .nii.gz extension | Fixed to read file_ending from dataset.json |
