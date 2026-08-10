# ROADMAP — FaceSim Pipeline

## Research Question
Open-source pipeline for predicting aesthetic facial changes (before→after) following orthognathic surgery from a single pre-op CBCT. No commercial software.

---

## Stack

| Layer | Tool | Status |
|---|---|---|
| Segmentation | TotalSegmentator (ToothFairy3 weights) | ✅ |
| Mesh processing | Open3D, trimesh, meshio | ✅ |
| Soft tissue sim | Ratio-scaled prescribed displacement + Laplacian (scipy sparse) | ✅ |
| Render before/after | Phase 4 pipeline | ✅ |
| ML segmentation training | nnUNet v2 3d_fullres fold 0 | 🔄 |
| Pipeline orchestration | Python state machine phases 1–6 | ✅ |
| Notifications | Telegram bot | ✅ |
| Patient data transfer | HuggingFace private repo | ✅ |
| Model benchmarking | nnU-Net vs TotalSegmentator, same held-out set | ✅ |
| Prediction validation | Surface distance vs post-op scan + do-nothing control | ⏸ needs paired scan |

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
- [x] Soft tissue segmentation: body envelope above the air threshold, closed,
      hole-filled, largest connected component
- [x] 58 STL files exported (Taubin smoothing → quadric decimation → binary STL)
- [x] Patient data uploaded to HuggingFace → downloaded to L40 server

**Three bugs fixed here, each of which broke everything downstream:**

| Was | Effect | Now |
|---|---|---|
| `requirements.txt` pinned `pydicom==2.4.4` | `dicom2nifti` imports `pydicom.pixels` (3.x only) → TotalSegmentator failed to import at all | `pydicom>=3.0.0` |
| `SEGMENTATION_DEVICE=cuda` | TotalSegmentator accepts only `gpu`/`cpu`/`mps` → `ValueError` on every run | `normalize_device()` translates torch-style names |
| Decimation was `np.random.choice(faces)` | Dropping random triangles punches holes; skin mesh came out as 96 519 components with 70 235 non-manifold edges. This is why gmsh/FEBio "could not handle" the mesh | Quadric edge-collapse (`fast_simplification`), teeth now `watertight=True`, skin one component at 99.6% of faces |

The soft-tissue mask was also the raw HU band `-700…-200` written straight to disk —
the fat/air transition shell plus all the scanner noise. It is now the body envelope
(`> -500 HU`, closed, filled, largest component). Volume in cc used a hardcoded
0.4 mm spacing; it now reads the scanner's own spacing.

---

## Phase 2 — Meshing ✅ COMPLETE

- [x] Open3D mesh processing and cleanup
- [x] Before/after mesh directory structure created for Phase 4

---

## Phase 3 — Simulation ✅ COMPLETE

**Approach:** prescribed-displacement boundary conditions + Laplacian propagation
on the skin surface. FEBio was abandoned because the skin STL was not watertight —
that turned out to be a bug in mesh export (random face dropping, see Phase 1), not
a property of the data, so the FEM route is open again if needed.

- [x] Anatomical frame derived from the segmentation ([anatomy.py](pipeline/anatomy.py)),
      not assumed from mesh axis order
- [x] Surgical plan is a rigid transform of the mandible: translation on all three
      anatomical axes + pitch about the lateral axis (Rodrigues)
- [x] Jaw-adjacent skin nodes detected via cKDTree, each driven by its nearest bone
      node's motion scaled by a region-dependent soft:hard ratio
- [x] Skull vault nodes fixed (top 5% along the true superior axis)
- [x] Sparse Laplacian propagation, 300 iterations (~60 sec total)
- [x] Outputs: `data/sim/deformed.vtk`, `data/mesh/before/`, `data/mesh/after/`
      (the "after" jaw mesh is the operated jaw, not a copy of the pre-op one)

**Two bugs fixed here, both silent:** the old code pushed along mesh axis 1 and
anchored on axis 2, which for this scanner's export means "move the jaw backwards"
and "fix the left side of the head". And jaw-adjacent skin moved 1:1 with bone
everywhere, which no soft tissue does.

**Parameters (env-overridable, or passed per request by the server):**

| Env var | Default | Effect |
|---|---|---|
| `FACESIM_ADVANCE_MM` | 5.0 | Mandible advancement, anterior + (setback = negative) |
| `FACESIM_VERTICAL_MM` | 0.0 | Vertical movement, superior + |
| `FACESIM_LATERAL_MM` | 0.0 | Asymmetry correction, patient-left + |
| `FACESIM_PITCH_DEG` | 0.0 | Rotation about the lateral axis, chin anterior at + |
| `FACESIM_JAW_RADIUS_MM` | 15.0 | Skin influence radius around jaw |
| `FACESIM_ITERS` | 300 | Propagation iterations |
| `FACESIM_RATIO_PROFILE` | see code | Soft:hard ratio by height over the mandible |

Default ratio profile: 0.95 at menton/pogonion → 0.90 at B point → 0.70 at the
lower lip → 0.35 at the commissure → 0.05 preauricular. These are population means
from orthognathic outcome studies, **not** patient-specific measurements.

---

## Phase 4 — Render ✅ COMPLETE

- [x] Before/after side-by-side renders generated from Phase 3 meshes

---

## Phase 5 — Validation ✅ IMPLEMENTED (waiting on data)

Measures the only claim that matters: how close the predicted face is to the real
post-op face. See [p5_validate.py](pipeline/phases/p5_validate.py).

- [x] Post-op scan segmented through the same code path as the pre-op one
- [x] Rigid registration on the unoperated face (above the mandible: forehead,
      orbits, nose, zygomas) — the standard orthognathic superimposition region
- [x] Surface distance predicted → real: mean / RMS / median / p90 / p95 / max,
      plus the fraction within 2 mm
- [x] **Do-nothing control:** the same distance from the untouched pre-op face.
      Reported as `improvement_pct`. Without it a good-looking number proves nothing
- [x] Self-test ([tests/selftest_phase5.py](tests/selftest_phase5.py)) feeds a known
      rigid transform back in and asserts the validator recovers it (<0.5 mm)
- [ ] **Blocked:** needs a real paired pre/post-op scan. Drop it in `data/post_op/`

---

## Phase 6 — ML Training ✅ MODEL BEATS THE INCUMBENT (integration pending)

**Goal:** replace TotalSegmentator in Phase 1 with our own nnU-Net.

Model is trained and measurably better in-domain (numbers below). It is **not
wired into Phase 1**, and the reason is not plumbing — it is that the in-domain
win does not transfer to the patient scan.

### What ran on the patient

Inference on the Planmeca ProMax scan takes ~60 s. Raw output, compared against
TotalSegmentator (neither is ground truth here — this measures agreement, not
correctness):

| structure | nnU-Net raw | after cleanup | TotalSegmentator | Dice raw → clean |
|---|---|---|---|---|
| mandible | 103.3 cc | 58.9 cc | 52.5 cc | 0.604 → **0.814** |
| maxilla | 18.5 cc | 12.9 cc | 22.9 cc | 0.599 → 0.674 |

The raw mandible came out as **310 connected components** totalling twice the
plausible volume: the real mandible plus ~37 cc of temporal bone and cervical
spine sitting far posterior. ToothFairy2 is dental-FOV CBCT; this scan is a whole
head with neck, full of bone the model has never seen and confidently mislabels.
`clean_prediction()` in [postprocess.py](pipeline/postprocess.py) drops the extra
components and recovers most of it.

**A held-out score on the training distribution says nothing about a scan whose
field of view is twice as large.** 0.987 in-domain, 0.814 agreement out-of-domain,
and that only after post-processing.

The residual maxilla gap is a label-definition difference, not an error:
ToothFairy2's `maxilla` covers less bone than TotalSegmentator's `upper_jawbone`.

### The anatomical frame cannot come from 7 classes

[anatomy.py](pipeline/anatomy.py) derives the patient frame from the maxillary
sinuses and individual FDI teeth, which the grouped model does not produce.
Re-deriving it from the grouped classes alone was tried and measured against the
validated 58-structure frame:

| axis | error | cue used |
|---|---|---|
| superior | **30.2°** | maxilla − mandible centroid |
| anterior | **28.6°** | cross product, sign from teeth vs mandible |
| lateral | 10.5° | canal centroids (sign needs a convention constant) |

30° on the superior axis would aim the surgical plan into the wrong part of the
face. The maxilla centroid is the culprit — it is skewed by exactly the label-extent
difference above.

**Conclusion: hybrid, not replacement.** Keep TotalSegmentator for the anatomical
frame and the auxiliary structures it segments well; use nnU-Net for the mandible
surface that drives Phase 3 and for the canals (0.88–0.89 vs 0.79–0.81, and they
are the surgical-safety structure). Each tool where it is actually validated.

### Result: the incumbent is beaten on every class

Measured by [05_benchmark_vs_totalseg.py](training/scripts/05_benchmark_vs_totalseg.py),
both models on the same 37 held-out cases against the same ground truth
(`data/benchmark114/benchmark.json`). Model: 3d_fullres, 182 cases.

| class | 250 epochs | 1000 epochs | TotalSegmentator | Δ (1000ep) |
|---|---|---|---|---|
| mandible | 0.9871 | **0.9890** | 0.8882 | +0.101 |
| maxilla | 0.7916 | **0.8584** | 0.7601 | +0.098 |
| left_canal | 0.8899 | **0.8991** | 0.8092 | +0.090 |
| right_canal | 0.8827 | **0.8760** | 0.7942 | +0.082 |
| upper_teeth | 0.9432 | **0.9461** | 0.9123 | +0.034 |
| lower_teeth | 0.9659 | **0.9755** | 0.9396 | +0.036 |
| **mean** | 0.9101 | **0.9240** | **0.8506** | **+0.073** |

The 1000-epoch run paid for itself almost entirely through the maxilla
(0.792 → 0.858) — the one class flagged as having real headroom. Everything else
moved by hundredths. Both runs use the same 37-case split, so the columns are
directly comparable. Ship the 1000-epoch checkpoint (Dataset115).

Success criterion (beat the mean **and** beat 0.77 on both canals): met.

For reference, the previous 60-case / 250ep / **3d_lowres** model scored 0.8307
mean against TotalSegmentator's 0.8401 on its own 12-case split — it lost. The
jump came from three things: 3× more data (60 → 182 fully-annotated cases),
3d_fullres instead of 3d_lowres, and the canal post-process below.

Caveat in TotalSegmentator's favour, unchanged: its teeth weights are trained on
ToothFairy3, a superset of ToothFairy2, so it may have seen these cases. Its
numbers are optimistic, which makes our margin conservative.

### The canal classes needed a post-process, not more epochs

Raw model output scored **0.607 / 0.281** on the canals — worse than the lowres
model. Diagnosis (per-case, on the held-out set):

```
canal union (3|4) Dice        0.8865      ← both canals found, almost perfectly
predicted canal volume        ~1.0x GT    ← nothing missing, nothing extra
the model's own side labels   0.607 / 0.281
```

The network detects both canals and then dumps ~80% of both into class 3. It is
not a resolution failure and more epochs cannot fix it: a 3d_fullres patch around
one canal is indistinguishable from the mirrored patch around the other, and the
patch is far too small to see which side of the head it sits on. The 3d_lowres
model got *balanced* canals for exactly that reason — its patches covered enough
skull to infer the side. Higher resolution traded away the global context that
left/right depends on.

Left/right is geometry, not texture. [pipeline/postprocess.py](pipeline/postprocess.py)
splits the predicted canal mask by the mid-sagittal plane (placed at the mandible
centroid, no ground truth involved), which recovers the union score on both sides:

```
after geometric side split    0.890 / 0.883
```

The lateral axis is derived per case from the two canal components, never
hardcoded — the array axis order depends on the export.

### Weights

| where | what |
|---|---|
| `povchingiz/toothfairy2-7class-model` (HF, **private**) | `nnUNetTrainer__nnUNetPlans__3d_fullres/` — the 1000-epoch model to use, plus `plans.json`/`dataset.json`, `postprocess.py`, the benchmark JSON and a model card. The `3d_lowres` folder is the old 60-case model, kept for provenance only |
| `data/nnunet/results/Dataset115_ToothFairy2_grouped/` | same checkpoints, local |
| `data/nnunet/results/Dataset114_ToothFairy2_grouped/` | the 250-epoch run |

Pull them on another machine:

```bash
huggingface-cli login          # or export HF_TOKEN
huggingface-cli download povchingiz/toothfairy2-7class-model \
    --local-dir $nnUNet_results/Dataset115_ToothFairy2_grouped
```

`postprocess.py` is uploaded **with** the weights on purpose: raw predictions score
0.607/0.281 on the canals, so shipping the checkpoint without it ships a broken
model. Anyone who downloads the weights gets the fix in the same folder.

Repo is private. `huggingface-cli repo visibility ... --public` opens it — check
the ToothFairy2 licence terms first, the weights are derived from that dataset.

### Gotcha: "480 cases" is 182 usable ones

ToothFairy2 ships 63 fully-annotated **F** cases and 417 partially-annotated **P**
cases. A P case may have the mandible and both canals but no maxilla. nnU-Net cannot
distinguish "not annotated" from "not present", so an unlabelled maxilla is learned
as background — the extra data degrades the very class it was supposed to improve.

`_filter_fully_annotated` downloads labels first (~10× smaller than images), keeps
only cases carrying every target class, and pulls images only for those. Result:
**182 cases**, three times the current model's training set. Disable with
`TF2_REQUIRE_ALL_CLASSES=0`.

### Canal side naming conflict

TotalSegmentator labels 3/4 as left/right inferior alveolar canal; the ToothFairy2
ground truth uses the opposite convention. Verified per case: `dice(pred=3, gt=3)`
is exactly 0.000 on all 12 while `dice(pred=3, gt=4)` is 0.55–0.88. The benchmark
swaps sides to compare fairly. **One of the two conventions is anatomically wrong**
and it is worth resolving before any surgical output shows a side label.

### Running it

```bash
./run_training_queue.sh                        # or, detached:
tmux new -s train './run_training_queue.sh'
tail -f logs/queue_7cls_480_250ep.log
```

The queue refuses to start below 450 GB free — dying on ENOSPC three days into a run
is the expensive failure mode. Queue contents:

1. `7cls_480_250ep` — 7 classes, all fully-annotated cases, 3d_fullres, 250 epochs.
   ~2 days on a 3090, so the benchmark can be re-run early.
2. `7cls_480_1000ep` — same data, full 1000 epochs. Optional if (1) clears the bar.

The 48-class run was dropped: per-tooth labels do not change the predicted face
(the simulation is driven by the mandible surface), and it costs another GPU week.
Re-add it when the product needs per-tooth output for implant planning.

### After training

```bash
.venv312/bin/python training/scripts/05_benchmark_vs_totalseg.py \
    --raw data/raw/datasets/toothfairy2_raw_all \
    --preds data/nnunet/results/Dataset114_*/*/fold_0/validation \
    --out data/benchmark
```

---

## Remaining Work

| Task | Priority | Status |
|---|---|---|
| ~~Finish 182-case 3d_fullres training~~ | 🔴 High | ✅ Done, mean Dice 0.9101 |
| ~~Re-run benchmark against TotalSegmentator~~ | 🔴 High | ✅ Beaten on every class (+0.060 mean) |
| Wire the model into Phase 1 | 🔴 High | Blocked on the anatomy-frame dependency above |
| ~~1000-epoch run (Dataset115)~~ | 🟡 Medium | ✅ Done — mean 0.9240, maxilla 0.792 → 0.858 |
| Get a paired post-op scan → close Phase 5 | 🔴 High | Only thing that can prove the prediction |
| Resolve the canal left/right convention | 🟡 Medium | Safety-relevant once sides are displayed |
| Show before/after renders in web UI | 🟡 Medium | Server now returns them in the ZIP; not drawn in the browser yet |
| Patient-specific soft:hard ratios instead of population means | 🟡 Medium | Needs validated cases first |
| Pre-loaded demo mode (existing patient) | 🟡 Medium | Fastest path to shareable demo |
| FEM (tetra + FEBio) now that meshes are clean | 🟢 Low | Only worth it if the ratio model fails validation |
| Scale to multiple patients | 🟢 Low | Not needed yet |

---

## Server Info

| | |
|---|---|
| GPU (current box) | NVIDIA RTX 3090, 24 GB VRAM, 16 cores, 62 GB RAM, 32 GB /dev/shm |
| GPU (earlier, rented) | NVIDIA L40S, 47.7 GB VRAM, Docker with 64 MB shm |
| Venv | `.venv312` |
| Run pipeline | `make pipeline` |
| Run training only | `make train` |
| Check status | `make status` |
| Watch a training run | `tail -f data/nnunet/results/Dataset*/*/fold_0/training_log_*.txt` |

The 64 MB-shm Docker box is why `nnUNet_n_proc_DA` was pinned to 0. On this
machine that pin cost **3.6x**: single-threaded augmentation left the GPU at 20%
utilisation and epochs took 240 s. Workers are now sized from actual `/dev/shm`
and core count (12 here) → 66 s/epoch at 99.7% GPU. `_is_shm_error()` still falls
back to 0 workers if the guess is wrong on some other machine.

---

## Known Issues / Workarounds

| Issue | Workaround |
|---|---|
| ~~Skin STL not watertight → gmsh/FEBio fails~~ | **Root cause found:** random-face decimation in `masks_to_stl.py`, not the data. Fixed; the FEM route is open again |
| Docker /dev/shm too small (~64MB) → training dies with "unable to allocate shared memory(shm): No space left" | `nnUNet_n_proc_DA=0` (augmentation in main process, no shm) — forced in `p6_train.py`. Note: `=2` was **not** enough. Alt: restart container with `--shm-size=8g` and re-enable workers. |
| ToothFairy2 dataset.json has wrong numTraining (0) | `_patch_dataset_json` counts actual files |
| ToothFairy2 labels missing "background" key | Auto-discovered from actual .mha files |
| Interrupted download accepted as complete (`TF2_MAX_CASES=0` trusted any non-empty cache) | Training silently started on 4 cases → `n_splits=5 > n_samples=4`. Now the repo listing decides what "all" means |
| Leftover dataset dir from a crashed run reused as-is | `_dataset_case_count()` — a half-built dataset is rebuilt, not trained on |
| `_remap_48_to_7` copied every image verbatim | +190 GB of duplicates at 480 cases; now hardlinked |
| `pipeline/main.py` block-buffered through `tee` | Queue logs stayed empty for hours; `python -u` in `run_training_queue.sh` |
| fast-simplification target_reduction validation | Primary path again (open3d kept as fallback) |
| Smoke test hardcoded .nii.gz extension | Fixed to read file_ending from dataset.json |
